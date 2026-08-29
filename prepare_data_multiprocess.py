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

# Conservative by default so this doesn't hog every core/all RAM and hang the
# machine. Raise it for more throughput: PACK_WORKERS=8 python prepare_data_multiprocess.py
NUM_WORKERS = int(os.environ.get("PACK_WORKERS", max(1, cpu_count() // 2)))

# Documents tokenized per streamed chunk within a file. Bounds peak memory per
# worker to ~this many documents at a time instead of loading/tokenizing an
# entire ~700K-document file in one shot.
CHUNK_DOCS = int(os.environ.get("PACK_CHUNK_DOCS", 5000))

# Global tokenizer instance for worker processes. Each pool worker re-runs this
# line (multiprocessing "spawn" re-executes the module per process), so
# local_files_only avoids a Hub round-trip per worker once the tokenizer is cached.
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)

def process_file(file_path: str):
    """Streams a Parquet file in chunks of CHUNK_DOCS documents, batch-tokenizing
    each chunk and appending EOS tokens. Streaming (rather than loading and
    tokenizing all ~700K documents in a file at once) keeps peak memory per
    worker bounded regardless of file size."""
    try:
        eos_id = tokenizer.eos_token_id
        chunk_arrays = []
        parquet_file = pq.ParquetFile(file_path)

        for batch in parquet_file.iter_batches(batch_size=CHUNK_DOCS, columns=["text"]):
            texts = batch.column("text").to_pylist()
            encoded_docs = tokenizer(texts)["input_ids"]

            chunk_tokens = []
            for doc_tokens in encoded_docs:
                chunk_tokens.extend(doc_tokens)
                chunk_tokens.append(eos_id)
            chunk_arrays.append(np.array(chunk_tokens, dtype=np.uint16))

        if not chunk_arrays:
            return file_path, np.array([], dtype=np.uint16)
        return file_path, np.concatenate(chunk_arrays)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, np.array([], dtype=np.uint16)

def main():
    # Recursive so nested layouts (e.g. fineweb_raw/sample/100BT/*.parquet) are found too
    parquet_files = glob.glob(os.path.join(PARQUET_DIR, "**", "*.parquet"), recursive=True)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {PARQUET_DIR}")

    print(f"🚀 Found {len(parquet_files)} local Parquet files under {PARQUET_DIR}")
    print(f"⚡ Starting multi-core packing with {NUM_WORKERS} CPU workers...")

    # Allocate memory-mapped binary file on disk
    arr = np.memmap(OUTPUT_BIN, dtype=np.uint16, mode='w+', shape=(TARGET_TOKENS,))

    token_count = 0
    files_done = 0
    pbar = tqdm(total=TARGET_TOKENS, unit="tok", unit_scale=True, desc="Packing Data")
    pbar.set_postfix(file=f"0/{len(parquet_files)}", fill="0.0%")

    with Pool(NUM_WORKERS) as pool:
        # chunksize=1: each worker reports back after every single file instead of
        # buffering several files' worth of token arrays in memory before returning.
        for file_path, tokens in pool.imap_unordered(process_file, parquet_files, chunksize=1):
            files_done += 1
            n_tok = len(tokens)

            if n_tok > 0:
                # Check if adding this batch exceeds target token limit
                if token_count + n_tok > TARGET_TOKENS:
                    n_tok = TARGET_TOKENS - token_count
                    arr[token_count:token_count + n_tok] = tokens[:n_tok]
                    token_count += n_tok
                    pbar.update(n_tok)
                else:
                    arr[token_count:token_count + n_tok] = tokens
                    token_count += n_tok
                    pbar.update(n_tok)

            pbar.set_postfix(
                file=f"{files_done}/{len(parquet_files)} ({os.path.basename(file_path)})",
                fill=f"{100 * token_count / TARGET_TOKENS:.1f}%"
            )

            if token_count >= TARGET_TOKENS:
                break

    pbar.close()
    arr.flush()

    fill_pct = 100 * token_count / TARGET_TOKENS
    print(f"\n✅ Packing complete! Processed {files_done}/{len(parquet_files)} files.")
    print(f"   Total tokens stored in {OUTPUT_BIN}: {token_count / 1e9:.2f}B / {TARGET_TOKENS / 1e9:.0f}B target ({fill_pct:.1f}% filled)")
    if token_count < TARGET_TOKENS:
        print(f"   ⚠️  Ran out of source data before reaching the target — the remaining "
              f"{(TARGET_TOKENS - token_count) / 1e9:.2f}B tokens in {OUTPUT_BIN} are still zero-padding.")

if __name__ == "__main__":
    main()
