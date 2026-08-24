import os
import glob
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Configuration
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
PARQUET_DIR = "./fineweb_raw"
OUTPUT_BIN = "train_25b_packed.bin"
TARGET_TOKENS = 25_000_000_000  # 25 Billion Tokens
NUM_WORKERS = max(1, cpu_count() - 2)  # Leave 2 cores free for system background tasks

# Global tokenizer instance for worker processes
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def process_file(file_path: str) -> np.ndarray:
    """Reads a Parquet file, tokenizes the text column, and appends EOS tokens."""
    try:
        table = pq.read_table(file_path, columns=["text"])
        texts = table["text"].to_pylist()
        all_tokens = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            tokens.append(tokenizer.eos_token_id)
            all_tokens.extend(tokens)
            
        return np.array(all_tokens, dtype=np.uint16)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.array([], dtype=np.uint16)

def main():
    parquet_files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {PARQUET_DIR}")

    print(f"🚀 Found {len(parquet_files)} Parquet files.")
    print(f"⚡ Starting multi-core packing with {NUM_WORKERS} CPU workers...")

    # Allocate memory-mapped binary file on disk
    arr = np.memmap(OUTPUT_BIN, dtype=np.uint16, mode='w+', shape=(TARGET_TOKENS,))
    
    token_count = 0
    pbar = tqdm(total=TARGET_TOKENS, unit="tok", unit_scale=True, desc="Packing Data")

    with Pool(NUM_WORKERS) as pool:
        for tokens in pool.imap_unordered(process_file, parquet_files, chunksize=5):
            n_tok = len(tokens)
            if n_tok == 0:
                continue

            # Check if adding this batch exceeds target token limit
            if token_count + n_tok > TARGET_TOKENS:
                n_tok = TARGET_TOKENS - token_count
                arr[token_count:token_count + n_tok] = tokens[:n_tok]
                token_count += n_tok
                pbar.update(n_tok)
                break

            arr[token_count:token_count + n_tok] = tokens
            token_count += n_tok
            pbar.update(n_tok)

            if token_count >= TARGET_TOKENS:
                break

    pbar.close()
    arr.flush()
    print(f"\n✅ Packing complete! Total tokens stored in {OUTPUT_BIN}: {token_count / 1e9:.2f} Billion")

if __name__ == "__main__":
    main()
