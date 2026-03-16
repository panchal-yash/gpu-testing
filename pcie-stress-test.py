import torch
import time

device = torch.device("cuda:0")

# Pinned CPU memory for fast DMA transfer
size = 256 * 1024 * 1024  # 256M elements = ~1GB per tensor (float32)
cpu_tensor = torch.randn(size, dtype=torch.float32).pin_memory()
gpu_tensor = torch.empty(size, dtype=torch.float32, device=device)

print("Streaming CPU → GPU continuously...")
print("Monitor with: dcgmi dmon -e 1009,1010 -d 1000")

start = time.time()
bytes_transferred = 0

while time.time() - start < 30:  # Run for 30 seconds
    gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    torch.cuda.synchronize()
    bytes_transferred += cpu_tensor.nbytes

elapsed = time.time() - start
gbps = (bytes_transferred / elapsed) / 1e9
print(f"Achieved: {gbps:.2f} GB/s over {elapsed:.1f}s")
