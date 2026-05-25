import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import sys
import os
from pathlib import Path
import tiktoken
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.txt"


def log(msg: str = ""):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


cleaned_path = Path(os.environ.get("data", SCRIPT_DIR / "input.txt"))
train_split = 0.9
seed = 1337

batch_size = 32
block_size = 543
max_iters = 25000
eval_interval = 1
sample_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 10
n_layer = 10
dropout = 0.0

torch.manual_seed(seed)


def get_tokenizer(encoding_name="gpt2"):
    tokenizer = tiktoken.get_encoding(encoding_name)
    vocab_size = tokenizer.n_vocab
    return tokenizer, vocab_size


def encode(text, tokenizer): return tokenizer.encode(text)
def decode(tokens, tokenizer): return tokenizer.decode(tokens)


with open(cleaned_path, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer, vocab_size = get_tokenizer("o200k_base")
data = torch.tensor(encode(text, tokenizer), dtype=torch.long)
n = int(train_split * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
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


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
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
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def print_sample(step, max_new_tokens=1000):
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output_ids = model.generate(context, max_new_tokens=max_new_tokens)
    sample = decode(output_ids[0].tolist(), tokenizer).strip()
    log("")
    log(f"sample at step {step}:")
    log("-" * 60)
    log(sample)
    log("-" * 60)
    log("")
    model.train()


model = GPTLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

log("")
log("quadtrix v1.0")
log(f"device: {device}")
log(f"number of parameters: {n_params:,}")
log(f"train tokens: {len(train_data):,}  |  val tokens: {len(val_data):,}")
log(f"batch size: {batch_size}  |  block size: {block_size}  |  lr: {learning_rate:.0e}")
log(f"layers: {n_layer}  |  heads: {n_head}  |  embd: {n_embd}  |  dropout: {dropout}")
log("")
log(f"training for {max_iters} steps, eval every {eval_interval}, sample every {sample_interval}")
log("")

best_val_loss = float('inf')
best_step = 0
prev_train = None
train_start = time.time()

for iter_num in range(max_iters):

    # eval
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - train_start

        # gradient norm
        total_norm = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        # loss delta
        delta_str = "    n/a" if prev_train is None else f"{losses['train'].item() - prev_train:+.4f}"
        prev_train = losses['train'].item()

        # throughput + ETA
        steps_done = iter_num + 1
        tok_per_sec = int(steps_done * batch_size *
                          block_size / elapsed) if elapsed > 0 else 0
        steps_left = max_iters - steps_done
        eta_sec = int(steps_left * elapsed /
                      steps_done) if steps_done > 0 else 0
        eta_str = f"{eta_sec // 60}m {eta_sec % 60:02d}s"

        is_best = losses['val'] < best_val_loss
        if is_best:
            best_val_loss = losses['val']
            best_step = iter_num
            torch.save(model.state_dict(), 'best_model.pt')

        marker = " (best)" if is_best else ""
        log(
            f"step {iter_num:>5}/{max_iters} |"
            f" train loss {losses['train'].item():.4f} |"
            f" val loss {losses['val'].item():.4f} |"
            f" delta {delta_str} |"
            f" norm {total_norm:.3f} |"
            f" {tok_per_sec:,} tok/s |"
            f" eta {eta_str}"
            f"{marker}"
        )
        sys.stdout.flush()
    if iter_num > 0 and iter_num % sample_interval == 0:
        print_sample(iter_num)
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

total_time = time.time() - train_start
log("")
log(f"training done in {int(total_time // 60)}m {int(total_time % 60):02d}s")
log(f"best val loss {best_val_loss:.4f} at step {best_step}")
log(f"checkpoint saved to best_model.pt")
log("")
model.load_state_dict(torch.load(
    'best_model.pt', map_location=device, weights_only=True))
model.eval()
log(f"restored best_model.pt (val {best_val_loss:.4f})")
log("")
log("final sample from best checkpoint:")
print_sample("final")
log("inference  |  type 'quit' to exit")
log("")

try:
    while True:
        prompt = input("you > ").strip()
        log(f"you > {prompt}")

        if prompt.lower() in ("quit", "exit", "q"):
            log("session ended.")
            break
        if not prompt:
            continue

        context = torch.tensor([encode(prompt, tokenizer)],
                               dtype=torch.long, device=device)
        with torch.no_grad():
            output_ids = model.generate(context, max_new_tokens=200)

        new_tokens = output_ids[0][context.shape[1]:].tolist()
        response = decode(new_tokens, tokenizer).strip()
        log("")
        log(f"model > {response}")
        log("")

except KeyboardInterrupt:
    log("")
    log("interrupted.")

log("")
log(f"total training time: {int(total_time // 60)}m {int(total_time % 60):02d}s")
log(f"log written to: {LOG_PATH}")
log("")
