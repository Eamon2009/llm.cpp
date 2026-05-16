import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import tiktoken
from pathlib import Path

# Training configuration
script_dir = Path(__file__).parent
file_path = script_dir / "input.txt"
model_path=script_dir / 'best_model.pt'
batch_size    = 16
block_size    = 32
max_iters     = 2000
eval_interval = 100
learning_rate = 1e-3
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters    = 20
n_embd        = 64
n_head        = 4
n_layer       = 4
dropout       = 0.1

# Tokenizer setup
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

# Model definition (minimal GPT)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
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
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

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
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# Data loading function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Load your dataset - REPLACE 'input.txt' with your actual data file
print("Loading data...")
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Encode using tiktoken
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Dataset: {len(data):,} tokens | Train: {len(train_data):,} | Val: {len(val_data):,}")

# Initialize model
print(f"Initializing model with vocab_size={vocab_size}")
model = GPTLanguageModel().to(device)

# Load existing weights
print("Loading weights from best_model.pt...")
checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict):
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
else:
    model.load_state_dict(checkpoint)
print(f"Weights loaded successfully")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
print(f"\nStarting fine-tuning on {device}")
print(f"{'Step':<10} {'Train Loss':<12} {'Val Loss':<12} {'Time (ms)':<12} {'Tok/s':<10}")
print("-" * 66)

start_time = time.time()
for iter in range(max_iters):
    
    # Evaluate
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        elapsed = (time.time() - start_time) * 1000
        tokens_per_sec = (batch_size * block_size * eval_interval) / ((time.time() - start_time) if iter > 0 else 1)
        
        print(f"{iter:<10} {losses['train']:.6f}     {losses['val']:.6f}     {elapsed:<12.2f} {tokens_per_sec:<10.0f}")
        start_time = time.time()
    
    # Training step
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save fine-tuned model
print("\nSaving fine-tuned model...")
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iter': max_iters,
    'config': {
        'vocab_size': vocab_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'block_size': block_size,
        'dropout': dropout,
    }
}, 'finetuned_model.pt')
print(" Model saved to finetuned_model.pt")