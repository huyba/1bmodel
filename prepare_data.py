import os
import time
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

OUTPUT_FILE = 'train_25B_packed.bin'
TOTAL_TOKENS = 25_000_000_000 #25B
SEQ_LEN = 8192 #8K
VOCAB_SIZE = 65535 #64K

TOKENIZER_NAME = 'EleutherAI/gpt-neox-20b'

print(f"[1/4] Loading Tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

actual_vocab_size = len(tokenizer)
print(f'vocab size {actual_vocab_size}')
assert actual_vocab_size <= 65535, 'more than vocab size'

# Get ID of EOS - end of each content
EOS_TOKEN_ID = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
if EOS_TOKEN_ID is None:
    EOS_TOKEN_ID = 0  # Fallback safety

print(f"[2/4] Init Memory-Mapped Binary File: {OUTPUT_FILE}")
expected_gb = (TOTAL_TOKENS * 2) / (1024 ** 3)
print(f"      Expected disk capacity needed: {expected_gb:.2f} GB ({TOTAL_TOKENS:,} tokens x 2 bytes)")

# Create empy binary file 50GB on SSD NVMe
fp = np.memmap(OUTPUT_FILE, dtype=np.uint16, mode='w+', shape=(TOTAL_TOKENS,))

# ==============================================================================
# 1. DATA SOURCE CONFIG
# ==============================================================================
# use FineWeb-Edu sample-100BT
DATA_SET_REPO = "HuggingFaceFW/fineweb-edu"
DATA_SET_NAME = "sample-100BT"
DATA_SET_SPLIT = "train"

# ==============================================================================
# 3. STREAMING DATA GENERATOR
# ==============================================================================
def create_text_stream():
    """
    Stream liên tục dữ liệu văn bản từ FineWeb-Edu.
    """
    print("[3/4] Đang kết nối Streaming API tới FineWeb-Edu...")
    dataset = load_dataset(DATA_SET_REPO, name=DATA_SET_NAME, split=DATA_SET_SPLIT, streaming=True)

    for item in dataset:
        text = item.get("text", "")
        if text.strip():
            yield text


# ==============================================================================
# 4. SEQUENCE PACKING & WRITE TO DISK
# ==============================================================================
print("[4/4] Encoding, Sequence Packing (8K) write to disk...")
text_stream = create_text_stream()

buffer = []
write_idx = 0
start_time = time.time()

pbar = tqdm(total=TOTAL_TOKENS, unit="tokens", unit_scale=True, desc="Processing Progress")

for text in text_stream:
    if write_idx >= TOTAL_TOKENS:
        break

    # 1. Text -> Token IDs
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # 2. Sequence Packing: Add EOS token to the end
    tokens.append(EOS_TOKEN_ID)
    buffer.extend(tokens)

    # 3. When buffer gets to 8K (SEQ_LEN) -> Packing and write to disk
    while len(buffer) >= SEQ_LEN:
        chunk = buffer[:SEQ_LEN]
        buffer = buffer[SEQ_LEN:]  # Keep the remaining to next chunk

        chunk_len = len(chunk)

        # Cut if close to 25B tokens
        if write_idx + chunk_len > TOTAL_TOKENS:
            chunk_len = TOTAL_TOKENS - write_idx
            chunk = chunk[:chunk_len]

        # Force to uint16 (2 bytes/token) and write to binary file
        fp[write_idx : write_idx + chunk_len] = np.array(chunk, dtype=np.uint16)

        write_idx += chunk_len
        pbar.update(chunk_len)

        if write_idx >= TOTAL_TOKENS:
            break

# Flush to disk
fp.flush()
pbar.close()

elapsed_hours = (time.time() - start_time) / 3600
print(f"\n✅ COMPLETED PRE-TRAINING!")
print(f"  - File output      : {OUTPUT_FILE}")
print(f"  - Total Tokens     : {write_idx:,}")
print(f"  - File Size        : {os.path.getsize(OUTPUT_FILE) / (1024**3):.2f} GB")
print(f"  - Total Time Elapsed: {elapsed_hours:.2f} hours")

