from datasets import load_dataset
import os

# Settings
target_size_mb = 20
target_size_bytes = target_size_mb * 1024 * 1024
output_file = "input.txt"

print(f"Streaming TinyStories until {target_size_mb} MB is reached...")

# We use streaming=True so we don't download the whole 2GB+ file at once
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

current_size = 0
with open(output_file, "w", encoding="utf-8") as f:
    for entry in dataset:
        story_text = entry["text"] + "\n\n"

        # Calculate size of this story in bytes
        story_bytes = len(story_text.encode('utf-8'))

        if current_size + story_bytes > target_size_bytes:
            break

        f.write(story_text)
        current_size += story_bytes

print(f"Done! Created '{output_file}' ({current_size / (1024*1024):.2f} MB)")
