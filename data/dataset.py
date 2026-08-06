import os
import numpy as np
import tiktoken
from datasets import load_dataset

# Configuration
output_dir = "data/shards"
shard_size_tokens = 12_500_000  # 12.5M tokens per shard (~50 MB per .bin file)
# ~2.5B tokens (approx equivalent to 10GB raw text)
target_total_tokens = 2_500_00_000

os.makedirs(output_dir, exist_ok=True)

print("Initializing GPT-2 BPE tokenizer...")
enc = tiktoken.get_encoding("gpt2")
eot_token = enc._special_tokens.get("<|endoftext|>", 50256)

print(
    f"Streaming TinyStories and tokenizing directly into binary shards in '{output_dir}'...")

dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

token_buffer = []
shard_idx = 0
total_tokens_written = 0

for entry in dataset:
    if total_tokens_written >= target_total_tokens:
        break

    # Tokenize story + append End-Of-Text token
    tokens = enc.encode_ordinary(entry["text"])
    tokens.append(eot_token)
    token_buffer.extend(tokens)

    # Dump to binary file whenever the token buffer reaches shard size
    while len(token_buffer) >= shard_size_tokens:
        shard_tokens = token_buffer[:shard_size_tokens]
        token_buffer = token_buffer[shard_size_tokens:]

        # Convert to raw 32-bit integer array matching C++ int* memory mapping
        shard_arr = np.array(shard_tokens, dtype=np.int32)
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
        shard_arr.tofile(shard_path)

        total_tokens_written += shard_size_tokens
        print(f"Saved {shard_path} | Total tokens: {total_tokens_written:,}")
        shard_idx += 1

# Write remaining tokens in the buffer as the final shard
if token_buffer and total_tokens_written < target_total_tokens:
    shard_arr = np.array(token_buffer, dtype=np.int32)
    shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
    shard_arr.tofile(shard_path)
    total_tokens_written += len(token_buffer)
    print(f"Saved {shard_path} (final) | Total tokens: {total_tokens_written:,}")

print(
    f"\nDone! Created {shard_idx + (1 if token_buffer else 0)} shard(s) with {total_tokens_written:,} tokens.")
