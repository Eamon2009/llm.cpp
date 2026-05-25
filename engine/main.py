import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import sys
import os
from pathlib import Path
import tiktoken

#
#  LOGGING UTILITIES

W = 78
DOUBLE = "=" * W
SINGLE = "-" * W
TICK = "best"
ARROW = ">"

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.txt"
SCRIPT_DIR = Path(__file__).resolve().parent


def log(message=""):
    line = "" if message == "" else f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{line}\n")


def header(title, subtitle=""):
    log()
    log(DOUBLE)
    log(f"  {title}")
    if subtitle:
        log(f"  {subtitle}")
    log(DOUBLE)


def row(label, value="", unit="", note=""):
    label_col = f"  {label:<28}"
    value_col = f"{str(value):<20}"
    unit_col = f"{unit:<8}"
    note_col = f"  {note}" if note else ""
    log(f"{label_col}{value_col}{unit_col}{note_col}")


def rule():   log(f"  {SINGLE}")
def blank():  log()
def info(msg):    log(f"  {ARROW}  {msg}")
def success(msg): log(f"  ok  {msg}")


#  SESSION


log(f"{'Quadtrix':^{W}}")
blank()
start = time.time()

#  CONFIGURATION


cleaned_path = Path(os.environ.get("data", SCRIPT_DIR / "input.txt"))
train_split = 0.9
seed = 1337

batch_size = 16
block_size = 32
max_iters = 6000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1

torch.manual_seed(seed)


# tokenizer

def get_tokenizer(encoding_name="gpt2"):
    tokenizer = tiktoken.get_encoding(encoding_name)
    vocab_size = tokenizer.n_vocab
    return tokenizer, vocab_size


def encode(text, tokenizer): return tokenizer.encode(text)
def decode(tokens, tokenizer): return tokenizer.decode(tokens)


#  DATA
with open(cleaned_path, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer, vocab_size = get_tokenizer("gpt2")
encoded_data = encode(text, tokenizer)

data = torch.tensor(encoded_data, dtype=torch.long)
n = int(train_split * len(data))
train_data = data[:n]
val_data = data[n:]

#  Batch and LOSS


def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


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

# model


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


class FeedFoward(nn.Module):
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
        self.ffwd = FeedFoward(n_embd)
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


#  INITIALISE
model = GPTLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ── num_activations: real count via forward hooks on a dummy batch ───────────


def count_activations(m, bs, seq_len, dev):
    total = 0
    hooks = []

    def _hook(module, inp, out):
        nonlocal total
        if isinstance(out, torch.Tensor):
            total += out.numel()
    for mod in m.modules():
        hooks.append(mod.register_forward_hook(_hook))
    dummy = torch.zeros(bs, seq_len, dtype=torch.long, device=dev)
    with torch.no_grad():
        m(dummy)
    for h in hooks:
        h.remove()
    return total


_num_activations = count_activations(model, batch_size, block_size, device)

# ── dataset batch counts ─────────────────────────────────────────────────────
_train_batches = len(train_data) // (batch_size * block_size)
_val_batches = len(val_data) // (batch_size * block_size)

row("Batch size", batch_size)
row("Block size", block_size)
row("Parameters", f"{n_params:,}")
blank()

# training
header("TRAINING",
       f"{max_iters:,} steps | eval every {eval_interval} | checkpoint on improvement")
blank()

best_val_loss = float('inf')
train_start = time.time()

for iter in range(max_iters):

    # ── val + checkpoint at eval_interval ────────────────────────────────────
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        is_best = losses['val'] < best_val_loss
        if is_best:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), 'best_model.pt')
        log(f"  val loss {losses['val']:.6f}")
        sys.stdout.flush()

    # ── forward + backward ───────────────────────────────────────────────────
    step_start = time.time()

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # real grad norm (after backward, before step)
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.detach().data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    optimizer.step()

    # real per-step dt and tok/sec
    step_dt_ms = (time.time() - step_start) * 1000
    tok_per_sec = (batch_size * block_size) / (step_dt_ms / 1000.0)

    # real current lr from optimizer state
    cur_lr = optimizer.param_groups[0]['lr']

    line = (f"step {iter} | loss: {loss.item():.6f} | lr {cur_lr:.4e} | norm: {grad_norm:.4f} | dt: {step_dt_ms:.2f}ms | tok/sec: {tok_per_sec:.2f}")
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    # ── sample 100 tokens every 50 steps ─────────────────────────────────────
    if (iter + 1) % 50 == 0:
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            sample_ids = model.generate(context, max_new_tokens=100)
        sample_text = decode(sample_ids[0].tolist(), tokenizer)
        model.train()
        blank()
        log(f"  [sample @ step {iter + 1}]")
        log(f"  {'-' * 60}")
        log(f"  {sample_text.strip()}")
        log(f"  {'-' * 60}")
        blank()

total_time = time.time() - train_start
blank()
rule()
row("Duration",       f"{int(total_time // 60)}m {int(total_time % 60):02d}s")
row("Best val loss",  f"{best_val_loss:.4f}", "", TICK)
row("Checkpoint",     "best_model.pt",        "", TICK)
rule()


#  RESTORE CHECKPOINT
blank()
model.load_state_dict(torch.load(
    'best_model.pt', map_location=device, weights_only=True))
model.eval()
success(f"Restored best_model.pt | val loss {best_val_loss:.4f}")

#  INFERENCE


header("INFERENCE", "quit / exit / q -> end session")
blank()

try:
    while True:
        prompt = input(f"  user  {ARROW} ").strip()
        log(f"  You  {ARROW} {prompt}")

        if prompt.lower() in ("quit", "exit", "q"):
            blank()
            success("Session ended.")
            break

        if not prompt:
            continue

        encoded_prompt = encode(prompt, tokenizer)
        context = torch.tensor(
            [encoded_prompt], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = model.generate(context, max_new_tokens=200)

        new_tokens = output_ids[0][len(encoded_prompt):].tolist()
        response = decode(new_tokens, tokenizer).strip()

        blank()
        log(f"  Model {ARROW} {response}")
        blank()

except KeyboardInterrupt:
    blank()
    success("Interrupted.")


end = time.time()
wall_clock = end - start

blank()
rule()
row("Training",     f"{int(total_time // 60)}m {int(total_time % 60):02d}s")
row("Total",
    f"{int(wall_clock // 60)}m {int(wall_clock % 60):02d}s", "", TICK)
rule()
blank()
log(f"{DOUBLE}\n")
