"""
Download 30MB of FineWeb dataset from Hugging Face.
FineWeb is a large-scale web text dataset for language model training.
"""

from datasets import load_dataset
import os

def download_fineweb_sample(output_dir="engine", target_size_mb=30):
    """
    Download approximately 30MB of FineWeb dataset.-
    """
    print(f"Downloading ~{target_size_mb}MB of FineWeb dataset...")
    print("This may take a few minutes depending on your connection speed.\n")
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    
    # Download and save data until we reach ~30MB
    target_bytes = target_size_mb * 1024 * 1024
    current_bytes = 0
    samples = []
    
    print("Streaming and collecting samples...")
    for i, sample in enumerate(dataset):
        sample_size = len(sample['text'].encode('utf-8'))
        
        if current_bytes + sample_size > target_bytes:
            break
            
        samples.append(sample)
        current_bytes += sample_size
        
        if (i + 1) % 100 == 0:
            print(f"Collected {i + 1} samples ({current_bytes / (1024*1024):.2f} MB)")
    
    print(f"\nDownloaded {len(samples)} samples ({current_bytes / (1024*1024):.2f} MB)")
    output_file = os.path.join(output_dir, "input.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample['text'])
            f.write('\n\n' + '='*80 + '\n\n')  # Separator between documents
    
    print(f"Data saved to: {output_file}")
    print(f"Total samples: {len(samples)}")
    print(f"Total size: {current_bytes / (1024*1024):.2f} MB")
    
    return output_file

if __name__ == "__main__":
    try:
        download_fineweb_sample()
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"\ Error: {e}")
        print("\nMake sure you have the 'datasets' library installed:")
        print("  pip install datasets")