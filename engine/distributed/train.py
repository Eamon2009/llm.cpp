import os
import sys
import time
from pathlib import Path
import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

SCRIPT_DIR = Path(__file__).resolve().parent

# -------------------------------------
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ.get("RANK"))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK"))
    ddp_world_size = int(os.environ.get("WORLD_SIZE"))
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # Fallback to local single-device execution
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

if master_process:
    print("[llm] - Distributed Cluster Mode Enabled")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Total Cluster GPUs (World Size): {ddp_world_size if ddp else 1}")

start = time.time()
cleaned_path = Path(os.environ.get("data", SCRIPT_DIR / "input.txt"))
train_split = 0.9
seed = 1337
batch_size = 2
block_size = 20
max_iters = 10000
eval_interval = 50
learning_rate = 3e-4
eval_iters = 20
n_embd = 6
n_head = 4
n_layer = 4
dropout = 0.0
torch.manual_seed(seed + ddp_rank)


def get_miniq_tokenizer(encoding_name="o200k_base"):
    miniq_tokenizer = tiktoken.get_encoding(encoding_name)
    miniq_vocab_size = miniq_tokenizer.n_vocab
    return miniq_tokenizer, miniq_vocab_size


def miniq_encode(text, tokenizer):
    return tokenizer.encode(text)


def miniq_decode(tokens, tokenizer):
    return tokenizer.decode(tokens)


# All processes load the dataset
with open(cleaned_path, "r", encoding="utf-8") as f:
    text = f.read()

miniq_tokenizer, vocab_size = get_miniq_tokenizer("o200k_base")
encoded_data = miniq_encode(text, miniq_tokenizer)
data = torch.tensor(encoded_data, dtype=torch.long)
n = int(train_split * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i: i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    raw_model.eval()  # Use un-wrapped model instance references
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = miniq_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    raw_model.train()
    return out


class MiniQuadtrixHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MiniQuadtrixMHA(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [MiniQuadtrixHead(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class MiniQuadtrixFFN(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MiniQuadtrixBlock(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MiniQuadtrixMHA(n_head, head_size)
        self.ffwd = MiniQuadtrixFFN(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniQuadtrix(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                MiniQuadtrixBlock(n_embd, n_head=n_head)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


miniq_model = MiniQuadtrix().to(device)
raw_model = miniq_model

if ddp:
    miniq_model = DDP(miniq_model, device_ids=[ddp_local_rank])
    raw_model = miniq_model.module

miniq_n_params = sum(p.numel() for p in raw_model.parameters())
miniq_optimizer = torch.optim.AdamW(miniq_model.parameters(), lr=learning_rate)

if master_process:
    print("CONFIG")
    print(f"Seed Configuration: {seed}")
    print(f"Per-GPU Batch size: {batch_size}")
    print(f"Effective Cluster Batch size: {batch_size * ddp_world_size}")
    print(f"Block size: {block_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Parameters: {miniq_n_params:,}")
    print(f"Train tokens: {len(train_data):,}")
    print()

best_val_loss = float("inf")
train_start = time.time()

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - train_start
        total_norm = 0
        for p in miniq_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        tokens_per_sec = (
            (iter + 1) * batch_size * ddp_world_size * block_size / elapsed
            if elapsed > 0
            else 0
        )
        if master_process:
            is_best = losses["val"] < best_val_loss
            if is_best:
                best_val_loss = losses["val"]
                torch.save(raw_model.state_dict(), "llm.pt")

            print(
                f"step {iter} | loss: {losses['train']:.6f} | val_loss: {losses['val']:.6f} | norm: {total_norm:.4f} | dt: {elapsed*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            sys.stdout.flush()

    xb, yb = get_batch("train")
    logits, loss = miniq_model(xb, yb)
    miniq_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    miniq_optimizer.step()

total_time = time.time() - train_start

if master_process:
    print("\n" + "=" * 50)
    print(
        f"Training Duration: {int(total_time // 60)}m {int(total_time % 60):02d}s")
    print(f"Best validation loss recorded: {best_val_loss:.4f}")
    print("=" * 50 + "\n")
    try:
        raw_model.load_state_dict(
            torch.load("llm.pt", map_location=device, weights_only=True)
        )
        raw_model.eval()

        print("INFERENCE TERMINAL CHAT MODE")
        print("Type 'exit' to terminate session.\n")

        while True:
            prompt = input("user > ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                print("\nSession ended.")
                break
            if not prompt:
                continue

            encoded_prompt = miniq_encode(prompt, miniq_tokenizer)
            context = torch.tensor(
                [encoded_prompt], dtype=torch.long, device=device
            )

            with torch.no_grad():
                output_ids = raw_model.generate(context, max_new_tokens=100)

            new_tokens = output_ids[0][len(encoded_prompt):].tolist()
            response = miniq_decode(new_tokens, miniq_tokenizer).strip()

            print(f"\nModel > {response}\n")

    except (KeyboardInterrupt, FileNotFoundError):
        print("\nInference interrupted or weight metrics file bypassed.")

    wall_clock = time.time() - start
    print(
        f"\nTotal process lifecycle runtime: {int(wall_clock // 60)}m {int(wall_clock % 60):02d}s\n")

# Shutdown distributed worker nodes gracefully
if ddp:
    dist.destroy_process_group()
