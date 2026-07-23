import os
import sys
import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
BLOCK_SIZE = 20
N_EMBD = 6
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0
MODEL_WEIGHTS_PATH = "llm.pt"
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
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337 + ddp_rank)


class MiniQuadtrixHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(
            torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
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
            [MiniQuadtrixHead(head_size) for _ in range(num_heads)])
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
            *[MiniQuadtrixBlock(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.inference_mode()
def main():
    tokenizer = tiktoken.get_encoding("o200k_base")
    vocab_size = tokenizer.n_vocab
    model = MiniQuadtrix(vocab_size).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH,
                          map_location=device, weights_only=True))
    model.eval()

    if master_process:
        print(f"Cluster inference online. Total GPUs active: {ddp_world_size}")

    while True:
        prompt_str = ""
        if master_process:
            try:
                prompt_str = input("user > ").strip()
                if not prompt_str:
                    prompt_str = "IGNORE_EMPTY"
            except (KeyboardInterrupt, EOFError):
                prompt_str = "EXIT_ENGINE"
        if ddp:
            str_len = torch.tensor(
                [len(prompt_str)], dtype=torch.long, device=device)
            dist.broadcast(str_len, src=0)
            char_tensor = torch.tensor(
                [ord(c) for c in prompt_str], dtype=torch.long, device=device)
            if not master_process:
                char_tensor = torch.zeros(
                    str_len.item(), dtype=torch.long, device=device)
            dist.broadcast(char_tensor, src=0)
            prompt_str = "".join([chr(x) for x in char_tensor.tolist()])

        if prompt_str in ("exit", "quit", "q", "EXIT_ENGINE"):
            break
        if prompt_str == "IGNORE_EMPTY":
            continue

        tokens = tokenizer.encode(prompt_str)
        context = torch.tensor([tokens], dtype=torch.long, device=device)
        output_ids = model.generate(context, max_new_tokens=100)
        new_tokens = output_ids[0][len(tokens):].tolist()
        local_response = tokenizer.decode(new_tokens).strip()
        if ddp:

            gather_objects = [None] * ddp_world_size
            dist.all_gather_object(gather_objects, local_response)
        else:
            gather_objects = [local_response]

        if master_process:
            print(f"\n--- Outputs across {ddp_world_size} GPUs ---")
            for rank_id, resp in enumerate(gather_objects):
                print(f"[GPU {rank_id}] > {resp}")
            print("-" * 40 + "\n")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
