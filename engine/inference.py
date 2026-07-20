import os
import sys
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters (Must match your trained model exactly)
BLOCK_SIZE = 20
N_EMBD = 6
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0  # Set to 0 for deterministic inference evaluation
MODEL_WEIGHTS_PATH = "llm.pt"
# -----------------------------------
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class MiniQuadtrixHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )
        self.dropout = nn.Dropout(DROPOUT)

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
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

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
            nn.Dropout(DROPOUT),
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

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[
                MiniQuadtrixBlock(N_EMBD, n_head=N_HEAD)
                for _ in range(N_LAYER)
            ]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        # idx shape: (Batch_Size, Context_Length)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits = self(idx_cond)
            # Extract last token slice for all batches
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def load_tokenizer(encoding_name="o200k_base"):
    tokenizer = tiktoken.get_encoding(encoding_name)
    return tokenizer, tokenizer.n_vocab


@torch.inference_mode()
def main():
    print("=" * 50)
    print("llm Inference Engine")
    print(f"Target Device: {DEVICE.upper()}")
    print("=" * 50)

    tokenizer, vocab_size = load_tokenizer("o200k_base")
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(
            f"Error: Weights file '{MODEL_WEIGHTS_PATH}' not found. Please train the model first."
        )
        sys.exit(1)

    model = MiniQuadtrix(vocab_size).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    try:
        # Fuses operations and parallelizes execution sub-graphs natively
        model = torch.compile(model)
        print("Graph compilation successful (torch.compile applied).")
    except Exception:
        print("Proceeding without torch.compile graph optimizations.")

    print("\nEntering Chat Mode. Type 'exit' or 'quit' to stop.")
    print(
        "Parallel Generation Mode: Returns 3 alternative paths simultaneously.\n"
    )

    while True:
        try:
            prompt = input("User > ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                print("\nGoodbye.")
                break
            if not prompt:
                continue

            # Tokenize input
            tokens = tokenizer.encode(prompt)

            # Leverage Batch Parallelism: Replicate the prompt tensor across dimension 0
            # This forces the GPU/CPU to execute 3 generations in parallel operations
            num_parallel_samples = 3
            context = (
                torch.tensor([tokens], dtype=torch.long, device=DEVICE)
                .repeat(num_parallel_samples, 1)
            )

            output_ids = model.generate(context, max_new_tokens=60)

            print(
                f"\nModel Responses ({num_parallel_samples} Parallel Paths):")
            for i in range(num_parallel_samples):
                new_tokens = output_ids[i][len(tokens):].tolist()
                response = tokenizer.decode(new_tokens).strip()
                print(f"  Path {i+1} -> {response}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break


if __name__ == "__main__":
    main()
