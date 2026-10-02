import sys
import torch

print("=" * 60, flush=True)
print("GPU & CUDA DIAGNOSTICS", flush=True)
print("=" * 60, flush=True)
print(f"PyTorch Version:  {torch.__version__}", flush=True)
cuda_avail = torch.cuda.is_available()
print(f"CUDA Available:   {cuda_avail}", flush=True)
if cuda_avail:
    device_count = torch.cuda.device_count()
    print(f"Device Count:     {device_count}", flush=True)
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"Device {i}:        {props.name}", flush=True)
        print(f"  VRAM Total:     {props.total_memory / (1024**3):.2f} GB", flush=True)
        print(f"  Compute Cap:    {props.major}.{props.minor}", flush=True)
    print(f"CUDA Built With:  {torch.version.cuda}", flush=True)
    print(f"cuDNN Version:    {torch.backends.cudnn.version()}", flush=True)
    
    # Simple tensor operation on GPU
    print("Testing GPU tensor allocation and matmul...", flush=True)
    x = torch.randn(100, 100, device="cuda")
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print(f"GPU Matmul Test:  SUCCESS! Computed on {x.device}", flush=True)
else:
    print("CUDA is NOT available.", flush=True)
print("=" * 60, flush=True)
