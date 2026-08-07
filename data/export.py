import struct
import os
import tiktoken


def export_tiktoken_vocab(output_dir="data/shards", encoding_name="gpt2"):
    os.makedirs(output_dir, exist_ok=True)
    enc = tiktoken.get_encoding(encoding_name)
    n_vocab = enc.n_vocab  # 50257 for gpt2

    output_path = os.path.join(output_dir, "tokenizer.bin")
    print(
        f"Exporting {encoding_name} vocab ({n_vocab} tokens) to {output_path}...")

    with open(output_path, "wb") as f:
        # C++ format: magic 'TKV1'
        f.write(struct.pack("<I", 0x544B5631))
        # base_vocab_size = 256 (byte-level foundation)
        f.write(struct.pack("<I", 256))
        # vocab_size = actual vocab count
        f.write(struct.pack("<I", n_vocab))
        # n_vocab entries
        f.write(struct.pack("<I", n_vocab))

        for i in range(n_vocab):
            token_bytes = enc.decode_single_token_bytes(i)
            f.write(struct.pack("<I", len(token_bytes)))
            f.write(token_bytes)

        # 0 merges — we are using pre-tokenized shards, so C++ never needs to re-encode
        # during training. This keeps the file simple and decode() works perfectly.
        f.write(struct.pack("<I", 0))

    print(f"Done. tokenizer.bin ready for C++.")


if __name__ == "__main__":
    export_tiktoken_vocab()
