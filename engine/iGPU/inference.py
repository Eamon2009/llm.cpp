import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

try:
    import torch_directml
except ImportError:
    torch_directml = None


W = 78
DOUBLE = "=" * W
SINGLE = "-" * W
ARROW = "->"

block_size = 32
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1


def header(title, subtitle=""):
    print(f"\n{DOUBLE}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(DOUBLE)


def row(label, value="", unit="", note=""):
    label_col = f"  {label:<28}"
    value_col = f"{str(value):<20}"
    unit_col = f"{unit:<8}"
    note_col = f"  {note}" if note else ""
    print(f"{label_col}{value_col}{unit_col}{note_col}")


def rule():
    print(f"  {SINGLE}")


def blank():
    print()


def get_device():
    if torch_directml is not None:
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_tokenizer(encoding_name="gpt2"):
    tokenizer = tiktoken.get_encoding(encoding_name)
    return tokenizer, tokenizer.n_vocab


def encode(text, tokenizer):
    return tokenizer.encode(text)


def decode(tokens, tokenizer):
    return tokenizer.decode(tokens)


tokenizer, vocab_size = get_tokenizer("gpt2")
device = get_device()


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
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
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
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
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def default_checkpoint_path():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "best_model.pt",
        Path.cwd() / "best_model.pt",
        Path.cwd() / "iGPU" / "best_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return script_dir / "best_model.pt"


def load_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train first with iGPU/main.py, or pass --checkpoint path/to/best_model.pt"
        )

    model = GPTLanguageModel().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_response(model, prompt, max_new_tokens, temperature, top_k):
    encoded_prompt = encode(prompt, tokenizer)
    context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    new_tokens = output_ids[0][len(encoded_prompt):].tolist()
    return decode(new_tokens, tokenizer).strip()


def chat(model, args):
    header("INFERENCE", "quit / exit / q -> end session")
    blank()

    while True:
        prompt = input(f"  user  {ARROW} ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            blank()
            print("  Session ended.")
            break
        if not prompt:
            continue

        response = generate_response(
            model,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
        )
        blank()
        print(f"  Model {ARROW} {response}")
        blank()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference from an iGPU trained .pt checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="Path to the .pt file generated by iGPU/main.py.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Generate once from this prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    print(f"{'Quadtrix-v1.0':^{W}}")
    blank()
    row("Started", time.strftime("%Y-%m-%d  %H:%M:%S"))
    row("Device", str(device))
    row("PyTorch", torch.__version__)
    row("Checkpoint", args.checkpoint)
    rule()

    model = load_model(args.checkpoint)

    if args.prompt:
        response = generate_response(
            model,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
        )
        blank()
        print(response)
    else:
        chat(model, args)

    blank()
    row("Total", f"{time.time() - start:.2f}s")
    print(DOUBLE)


if __name__ == "__main__":
    main()
