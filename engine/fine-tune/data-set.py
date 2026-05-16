import os
import sys
import argparse
from datasets import load_dataset

TARGET_MB = 30
OUTPUT_FILE = "input.txt"

DATASETS = {
    "alpaca":      ("yahma/alpaca-cleaned",            None,                  "train"),
    "dolly":       ("databricks/databricks-dolly-15k", None,                  "train"),
    "tinystories": ("roneneldan/TinyStories",          None,                  "train"),
    "wikitext":    ("Salesforce/wikitext",             "wikitext-103-raw-v1", "train"),
    "oasst":       ("OpenAssistant/oasst1",            None,                  "train"),
    "gsm8k":       ("openai/gsm8k",                   "main",                "train"),
}

def row_to_text(row):
    for field in ("text", "code", "content", "response", "output"):
        if row.get(field, "").strip():
            return row[field].strip() + "\n\n"
    parts = []
    if row.get("instruction", "").strip():
        parts.append("### Instruction:\n" + row["instruction"].strip())
    if row.get("input", "").strip():
        parts.append("### Input:\n" + row["input"].strip())
    if row.get("output", "").strip():
        parts.append("### Response:\n" + row["output"].strip())
    if parts:
        return "\n\n".join(parts) + "\n\n"
    for v in row.values():
        if isinstance(v, str) and v.strip():
            return v.strip() + "\n\n"
    return ""

def download(dataset_key, target_mb, output_file):
    target_bytes = target_mb * 1024 * 1024

    if dataset_key not in DATASETS:
        print(f"Unknown dataset '{dataset_key}'. Choose from: {', '.join(DATASETS)}")
        sys.exit(1)

    hf_id, config, split = DATASETS[dataset_key]
    print(f"Downloading '{dataset_key}' → {output_file} (target: {target_mb} MB)")

    load_kwargs = dict(split=split, streaming=True, trust_remote_code=True)
    if config:
        load_kwargs["name"] = config

    ds = load_dataset(hf_id, **load_kwargs)

    written = 0
    rows = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for row in ds:
            text = row_to_text(row)
            if not text:
                continue
            f.write(text)
            written += len(text.encode())
            rows += 1
            if written >= target_bytes:
                break

    print(f"Done. {rows} rows | {written / 1024 / 1024:.2f} MB → {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="alpaca", choices=DATASETS.keys())
    parser.add_argument("--mb", type=int, default=TARGET_MB)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    download(args.dataset, args.mb, args.output)

if __name__ == "__main__":
    main()