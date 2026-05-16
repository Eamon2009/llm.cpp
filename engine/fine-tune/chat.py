import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
DEFAULT_CONFIG = {
    'n_embd':      64,
    'n_head':      4,
    'n_layer':     4,
    'block_size':  32,
    'dropout':     0.0,   
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

class Head(nn.Module):
    def __init__(self, head_size, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(DEFAULT_CONFIG['n_embd'], head_size, bias=False)
        self.query = nn.Linear(DEFAULT_CONFIG['n_embd'], head_size, bias=False)
        self.value = nn.Linear(DEFAULT_CONFIG['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, block_size, dropout):
        super().__init__()
        n_embd = DEFAULT_CONFIG['n_embd']
        self.heads   = nn.ModuleList([Head(head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj    = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
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
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_embd     = cfg['n_embd']
        n_head     = cfg['n_head']
        n_layer    = cfg['n_layer']
        block_size = cfg['block_size']
        dropout    = cfg['dropout']

        self.block_size = block_size
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature          # (B, vocab_size)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
def load_model(pt_path: str) -> GPTLanguageModel:
    print(f"Loading checkpoint: {pt_path}")
    checkpoint = torch.load(pt_path, map_location=device)

    if isinstance(checkpoint, dict):
        cfg        = checkpoint.get('config', DEFAULT_CONFIG)
        state_dict = checkpoint.get('model', checkpoint)
    else:
        # Raw state dict saved directly
        cfg        = DEFAULT_CONFIG
        state_dict = checkpoint

    # Merge missing keys with defaults
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    cfg['dropout'] = 0.0          # always off at inference
    cfg['vocab_size'] = vocab_size

    # Update module-level config so layers build correctly
    DEFAULT_CONFIG.update(cfg)

    model = GPTLanguageModel(cfg).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded  |  params: {total_params:,}  |  device: {device}")
    print(f"  block_size={cfg['block_size']}  n_embd={cfg['n_embd']}  "
          f"n_head={cfg['n_head']}  n_layer={cfg['n_layer']}")
    return model
def generate_reply(model, prompt: str, max_new_tokens=200,
                   temperature=0.8, top_k=50, top_p=0.95) -> str:
    tokens = tokenizer.encode(prompt)
    idx    = torch.tensor([tokens], dtype=torch.long, device=device)
    out    = model.generate(idx, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k, top_p=top_p)
    # Return only the newly generated part
    new_tokens = out[0][len(tokens):].tolist()
    return tokenizer.decode(new_tokens)

def chat(model):
    print("\n" + "═" * 60)
    print("  Quadtrix Chat  —  type 'quit' or 'exit' to stop")
    print("  Commands: /temp <0-2>  /tokens <n>  /topk <n>  /topp <0-1>  /reset")
    print("═" * 60 + "\n")

    # Mutable settings
    settings = {
        'temperature': 0.8,
        'max_new_tokens': 200,
        'top_k': 50,
        'top_p': 0.95,
        'context_window': True,   # keep rolling context
    }
    history = ""   # rolling conversation context

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ('quit', 'exit'):
            print("Goodbye!")
            break
        if user_input.startswith('/'):
            parts = user_input.split()
            cmd   = parts[0].lower()
            try:
                if cmd == '/temp'   and len(parts) == 2:
                    settings['temperature'] = float(parts[1])
                    print(f"   temperature set to {settings['temperature']}")
                elif cmd == '/tokens' and len(parts) == 2:
                    settings['max_new_tokens'] = int(parts[1])
                    print(f"  max_new_tokens set to {settings['max_new_tokens']}")
                elif cmd == '/topk'  and len(parts) == 2:
                    settings['top_k'] = int(parts[1])
                    print(f"   top_k set to {settings['top_k']}")
                elif cmd == '/topp'  and len(parts) == 2:
                    settings['top_p'] = float(parts[1])
                    print(f"   top_p set to {settings['top_p']}")
                elif cmd == '/reset':
                    history = ""
                    print("  conversation history cleared")
                elif cmd == '/settings':
                    print(f"  {settings}")
                else:
                    print(f"   Unknown command: {cmd}")
            except ValueError:
                print("  Invalid value")
            continue
        history += user_input + "\n"
        prompt   = history
        reply = generate_reply(
            model, prompt,
            max_new_tokens = settings['max_new_tokens'],
            temperature    = settings['temperature'],
            top_k          = settings['top_k'],
            top_p          = settings['top_p'],
        )

        print(f"\nModel: {reply.strip()}\n")
        history += reply + "\n"
        tokens = tokenizer.encode(history)
        if len(tokens) > model.block_size - 50:
            history = tokenizer.decode(tokens[-(model.block_size - 50):])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Chat with model')
    parser.add_argument('--model', type=str, default='finetuned_model.pt',
                        help='Path to the .pt checkpoint file (default: finetuned_model.pt)')
    parser.add_argument('--max-tokens',  type=int,   default=200,  help='Max new tokens per reply')
    parser.add_argument('--temperature', type=float, default=0.8,  help='Sampling temperature (0.1–2.0)')
    parser.add_argument('--top-k',       type=int,   default=50,   help='Top-k sampling (0 = disabled)')
    parser.add_argument('--top-p',       type=float, default=0.95, help='Top-p nucleus sampling')
    parser.add_argument('--prompt',      type=str,   default=None, help='Single prompt (non-interactive)')
    args = parser.parse_args()

    model = load_model(args.model)

    if args.prompt:
        # One-shot mode
        reply = generate_reply(model, args.prompt,
                                max_new_tokens=args.max_tokens,
                                temperature=args.temperature,
                                top_k=args.top_k if args.top_k > 0 else None,
                                top_p=args.top_p)
        print(reply)
    else:
        chat(model)