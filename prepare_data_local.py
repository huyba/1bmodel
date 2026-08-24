import glob
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
# Directory where the downloaded CC-MAIN-* parquet files are stored
LOCAL_PARQUET_DIR = "./fineweb_raw"  
OUTPUT_FILE = "train_25b_packed.bin"
TOTAL_TOKENS = 25_000_000_000  # Target: 25 Billion Tokens
SEQ_LEN = 8192                 # Context window length (8K)
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b" # Vocab ~50K (fits uint16 safely)

# ==============================================================================
# 2. INITIALIZE TOKENIZER & MEMORY-MAPPED FILE
# ==============================================================================
print(f"[1/3] Loading Tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
EOS_TOKEN_ID = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

print(f"[2/3] Initializing 50GB Memory-Mapped Binary File: {OUTPUT_FILE}")
# Allocate uint16 binary file directly on disk (2 bytes per token * 25B tokens = ~50GB)
fp = np.memmap(OUTPUT_FILE, dtype=np.uint16, mode='w+', shape=(TOTAL_TOKENS,))

# ==============================================================================
# 3. RECURSIVE PARQUET SCANNING & OFFLINE TOKENIZATION
# ==============================================================================
# Recursively match all .parquet files inside CC-MAIN-* subdirectories
search_pattern = os.path.join(LOCAL_PARQUET_DIR, "**", "*.parquet")
parquet_files = sorted(glob.glob(search_pattern, recursive=True))

print(f"[3/3] Found {len(parquet_files)} Parquet files. Starting offline processing...")

buffer = []
write_idx = 0
start_time = time.time()

pbar = tqdm(total=TOTAL_TOKENS, unit="tokens", unit_scale=True, desc="Packing Data Offline")

for file_path in parquet_files:
    if write_idx >= TOTAL_TOKENS:
        break
        
    try:
        # Load only the 'text' column from the parquet file for maximum I/O performance
        df = pd.read_parquet(file_path, columns=["text"])
    except Exception as e:
        print(f"\n⚠️ Skipping corrupt file: {file_path} ({e})")
        continue
    
    for text in df["text"]:
        if write_idx >= TOTAL_TOKENS:
            break
            
        if not text or not isinstance(text, str):
            continue
            
        # Encode raw text to token IDs and append EOS token
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(EOS_TOKEN_ID)
        buffer.extend(tokens)
        
        # Sequence Packing: Group tokens into fixed 8192-length chunks
        while len(buffer) >= SEQ_LEN:
            chunk = buffer[:SEQ_LEN]
            buffer = buffer[SEQ_LEN:]
            
            chunk_len = len(chunk)
            if write_idx + chunk_len > TOTAL_TOKENS:
                chunk_len = TOTAL_TOKENS - write_idx
                chunk = chunk[:chunk_len]
                
            # Write chunk directly to disk via memmap
            fp[write_idx : write_idx + chunk_len] = np.array(chunk, dtype=np.uint16)
            write_idx += chunk_len
            pbar.update(chunk_len)
            
            if write_idx >= TOTAL_TOKENS:
                break

# Flush memory changes to physical disk
fp.flush()
pbar.close()

elapsed_time = time.time() - start_time
print(f"\n✅ PROCESSING COMPLETE!")
print(f"Output File : {OUTPUT_FILE}")
print(f"Total Disk Size : {os.path.getsize(OUTPUT_FILE) / (1024**3):.2f} GB")
print(f"Time Elapsed   : {elapsed_time / 3600:.2f} hours")
