import os
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

class PretrainBinaryDataset(IterableDataset):
    """
    Read binary data in both Single-process (Mac CPU) and Multi-GPU (DDP).
    """
    def __init__(self, data_dir: str, seq_len: int = 8192):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        total_shards = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.bin')])
        if not files:
            raise FileNotFoundError(f"could not find any .bin files")

        chunk_size = self.seq_len + 1
        for file_path in files:
            data = np.memmap(file_path, dtype=np.uint16, mode='r')
            num_chunks = len(data) // chunk_size

            try:
                for i in range(global_worker_id, num_chunks, total_shards):
                    start = i * chunk_size
                    chunk = torch.from_numpy(data[start : start + chunk_size].astype(np.int64))
                    yield chunk[:-1], chunk[1:]
            finally:
                # Close pointer memmap to free RAM/File Descriptor
                if hasattr(data, '_mmap') and data._mmap is not None:
                    data._mmap.close()
                del data

if __name__ == "__main__":
    print("🧪 [MacBook M1] Testing Dataset Pipeline...")
    dummy_dir = "./dummy_data"
    os.makedirs(dummy_dir, exist_ok=True)
    dummy_file = os.path.join(dummy_dir, "test.bin")

    # Create test data with 100,000 tokens if not exist
    if not os.path.exists(dummy_file):
        tokens = np.random.randint(0, 50280, size=(100000,), dtype=np.uint16)
        tokens.tofile(dummy_file)
        print(f"  - Created dummy binary file: {dummy_file}")

    # Test with small sequence length trên CPU
    ds = PretrainBinaryDataset(data_dir=dummy_dir, seq_len=512)
    dl = DataLoader(ds, batch_size=2)

    x, y = next(iter(dl))
    print(f"  - Output x shape : {x.shape}")  # Expected: [2, 512]
    print(f"  - Output y shape : {y.shape}")  # Expected: [2, 512]
    assert torch.equal(x[:, 1:], y[:, :-1]), "Target y not diff 1 token vs. x!"
    print("✅ DATASET TEST PASSED ON MACBOOK CPU!\n")
