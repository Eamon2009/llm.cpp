'''
 * llm.cpp - LMGNU Organization
 * Copyright (C) 2026 Eamon2009
 * https://github.com/LMGNU/llm.cpp
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 
 '''
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import sys
import os
import glob
import random
import numpy as np
import math
import re
from pathlib import Path
import tiktoken

# ----------------------------------------------------
# --- Nanochat Style Constants & Helpers ---
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

RANK = int(os.environ.get("RANK", 0))  # Process rank for multi-GPU training

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

# --- Setup Log Directory & Logging File ---
log_dir = SCRIPT_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_file_path, "w", encoding="utf-8")


def strip_ansi(text):
    """Removes ANSI color codes for clean plain-text log files."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def log_print(text="", end="\n"):
    """Prints to terminal (Rank 0 only) and writes clean output to log file."""
    if RANK == 0:
        print(text, end=end)
        sys.stdout.flush()
    # Strip ANSI colors before saving to text file
    log_file.write(strip_ansi(text) + end)
    log_file.flush()


def print_banner():
    banner = r"""
██      ██      ███▄███▄         ▄████ ████▄ ████▄
██      ██      ██ ██ ██        ██     ██ ██ ██ ██
██      ██      ██ ██ ██   ▄    ▀████  ████▀ ████▀
██████  ██████  ██ ██ ██   ▀           ██    ██   
                                       ▀▀    ▀▀   
"""
    log_print(f"{CYAN}{BOLD}{banner}{RESET}")


start = time.time()
train_split = 0.9
seed = 1337
batch_size = 2
block_size = 20
max_iters = 10000

# --- Logging & Eval Config ---
eval_interval = 50  # Run validation evaluation every 100 iterations
log_interval = 1     # Print per-step training progress
eval_iters = 20      # Averages loss over 20 evaluation steps

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 10
n_head = 4
n_layer = 4
dropout = 0.0

# --- Learning Rate Hyperparameters ---
learning_rate = 3e-4
max_lr = learning_rate
min_lr = max_lr / 10.0
warmup_iters = 500
lr_decay_iters = max_iters

torch.manual_seed(seed)


# --- 1. TOKENIZER CONFIGURATION (GPT-2 MATCHED TO SHARDS) ---
def get_miniq_tokenizer(encoding_name="gpt2"):
    miniq_tokenizer = tiktoken.get_encoding(encoding_name)
    miniq_vocab_size = miniq_tokenizer.n_vocab  # 50,257
    return miniq_tokenizer, miniq_vocab_size


miniq_tokenizer, vocab_size = get_miniq_tokenizer("gpt2")


def miniq_encode(text, tokenizer): return tokenizer.encode_ordinary(text)
def miniq_decode(tokens, tokenizer): return tokenizer.decode(tokens)


# --- 2. BINARY SHARD DATA LOADING LOGIC ---
shard_pattern = os.path.join(DATA_DIR, 'shards', 'shard_*.bin')
shard_files = sorted(glob.glob(shard_pattern))

if not shard_files:
    # Fallback search if pattern differs slightly
    shard_pattern = os.path.join(DATA_DIR, 'shards', 'shard*.bin')
    shard_files = sorted(glob.glob(shard_pattern))

if not shard_files:
    raise ValueError(
        f"No binary shard files found at '{shard_pattern}'. Please run your streaming data tokenization script.")

total_tokens = sum(os.path.getsize(
    f) // 4 for f in shard_files)  # 4 bytes per int32
train_tokens = int(train_split * total_tokens)
val_tokens = total_tokens - train_tokens
tokens_per_batch = batch_size * block_size


def get_batch(split):
    shard_file = random.choice(shard_files)
    # Memory-map 32-bit signed integers written by shard generator
    data = np.memmap(shard_file, dtype=np.int32, mode='r')

    n = int(train_split * len(data))
    data_split = data[:n] if split == 'train' else data[n:]

    ix = torch.randint(len(data_split) - block_size, (batch_size,))

    x = torch.stack(
        [torch.from_numpy((data_split[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
        (data_split[i + 1:i + block_size + 1]).astype(np.int64)) for i in ix])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    miniq_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = miniq_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    miniq_model.train()
    return out


# --- Cosine Learning Rate Scheduler with Warmup ---
def get_lr(it):
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


class MiniQuadtrixHead(nn.Module):
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


class MiniQuadtrixMHA(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [MiniQuadtrixHead(head_size) for _ in range(num_heads)])
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
            *[MiniQuadtrixBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
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


miniq_model = MiniQuadtrix().to(device)
miniq_n_params = sum(p.numel() for p in miniq_model.parameters())
miniq_optimizer = torch.optim.AdamW(miniq_model.parameters(), lr=learning_rate)

# --- Print Nanochat Banner ---
print_banner()

dev_str = f"{'CUDA' if torch.cuda.is_available() else 'CPU'}"
if torch.cuda.is_available():
    dev_str += f" ({torch.cuda.get_device_name(0)})"

table_str = f"""
+---------------------------------------------------------------------------------------+
| SYSTEM & HYPERPARAMETER CONFIGURATION                                                 |
+---------------------------------------+-----------------------------------------------+
| Device          : {dev_str:<18} | PyTorch        : {torch.__version__:<28} |
| Seed            : {seed:<18} | Batch Size     : {batch_size:<28} |
| Block Size      : {block_size:<18} | Max LR         : {max_lr:<28.4e} |
| Layers          : {n_layer:<18} | Min LR         : {min_lr:<28.4e} |
| Embedding Dim   : {n_embd:<18} | Warmup Iters   : {warmup_iters:<28} |
| Parameters      : {f'{miniq_n_params:,}':<18} | Vocab Size     : {vocab_size:<28} |
| Train Tokens    : {f'{train_tokens:,}':<18} | Val Tokens     : {f'{val_tokens:,}':<28} |
| Data Source     : {"shards/shard_*.bin":<18} | Log Path       : {str(log_file_path.name):<28} |
+---------------------------------------+-----------------------------------------------+
"""
log_print(table_str)

best_val_loss = float('inf')
current_val_loss = float('inf')
train_start = time.time()

for iter in range(max_iters):

    # 1. Update Learning Rate
    lr = get_lr(iter)
    for param_group in miniq_optimizer.param_groups:
        param_group['lr'] = lr

    # 2. Validation Loss Check
    if iter % eval_interval == 0 or iter == max_iters - 1:
        eval_losses = estimate_loss()
        current_val_loss = eval_losses['val']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(miniq_model.state_dict(), 'llm.pt')

    # 3. Time Per-step Execution
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    step_t0 = time.time()

    xb, yb = get_batch('train')
    logits, loss = miniq_model(xb, yb)
    miniq_optimizer.zero_grad(set_to_none=True)
    loss.backward()

    total_norm = 0.0
    for p in miniq_model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    miniq_optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    step_t1 = time.time()

    dt = step_t1 - step_t0
    tokens_per_sec = tokens_per_batch / dt if dt > 0 else 0.0

    # 4. Print Per-step Logs
    if iter % log_interval == 0:
        pct = ((iter + 1) / max_iters) * 100
        tokens_seen = (iter + 1) * tokens_per_batch
        current_epoch = tokens_seen / train_tokens if train_tokens > 0 else 0.0
        val_loss_str = f"{current_val_loss:.4f}" if current_val_loss != float(
            'inf') else "N/A"

        log_print(
            f"step {iter}/{max_iters}({pct:.1f}%) | "
            f"epoch {current_epoch:.3f} | "
            f"loss: {loss.item():.4f} | "
            f"val loss: {val_loss_str} | "
            f"lr: {lr:.4e} | "
            f"norm: {total_norm:.4f} | "
            f"dt: {dt*1000:.2f}ms | "
            f"tok/sec: {tokens_per_sec:.0f}"
        )

total_time = time.time() - train_start

log_print()
log_print(f"Duration: {int(total_time // 60)}m {int(total_time % 60):02d}s")
log_print(f"Best val loss: {best_val_loss:.4f}")
log_print()

miniq_model.load_state_dict(torch.load(
    'llm.pt', map_location=device, weights_only=True))
miniq_model.eval()

log_print(f"{CYAN}{BOLD}INFERENCE TERMINAL CHAT{RESET}")
log_print("Type 'exit' or 'q' to terminate session.\n")

try:
    while True:
        prompt = input("user > ").strip()
        log_file.write(f"user > {prompt}\n")
        if prompt.lower() in ("quit", "exit", "q"):
            log_print("\nSession ended.")
            break
        if not prompt:
            continue

        encoded_prompt = miniq_encode(prompt, miniq_tokenizer)
        context = torch.tensor(
            [encoded_prompt], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = miniq_model.generate(context, max_new_tokens=200)

        new_tokens = output_ids[0][len(encoded_prompt):].tolist()
        response = miniq_decode(new_tokens, miniq_tokenizer).strip()

        log_print(f"\nModel > {response}\n")

except KeyboardInterrupt:
    log_print("\nInterrupted.")

wall_clock = time.time() - start
log_print()
log_print(f"Total time: {int(wall_clock // 60)}m {int(wall_clock % 60):02d}s")
log_file.close()
