import sys
import torch

mods = [
    "pandas", "polars", "pyarrow", "numpy", "scipy",
    "rapidfuzz", "Levenshtein", "jellyfish",
    "sklearn", "xgboost", "lightgbm",
    "torch", "torchvision", "torchaudio",
    "transformers", "accelerate", "sentence_transformers",
    "matplotlib", "plotly", "tqdm", "psutil", "regex", "joblib"
]

print("=" * 65, flush=True)
print("PROJECT VIRTUAL ENVIRONMENT & HARDWARE VERIFICATION", flush=True)
print("=" * 65, flush=True)

all_ok = True
for m in mods:
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "installed")
        print(f"  [OK] {m:25s} v{ver}", flush=True)
    except Exception as e:
        print(f"  [FAIL] {m:23s} {e}", flush=True)
        all_ok = False

print("-" * 65, flush=True)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cuda_ver = torch.version.cuda
    print(f"  CUDA Available:     YES", flush=True)
    print(f"  Device Name:        {gpu_name}", flush=True)
    print(f"  Dedicated VRAM:     {vram_gb:.2f} GB", flush=True)
    print(f"  PyTorch CUDA:       {cuda_ver}", flush=True)
    
    # Simple tensor operation on GPU
    t1 = torch.ones(256, 256, device="cuda")
    t2 = torch.matmul(t1, t1)
    torch.cuda.synchronize()
    print(f"  GPU Matmul Check:   PASSED on {t1.device}", flush=True)
else:
    print("  CUDA Available:     NO (CPU mode only)", flush=True)
print("=" * 65, flush=True)
if all_ok and torch.cuda.is_available():
    print("ALL ENVIRONMENT AND GPU CHECKS PASSED PERFECTLY!", flush=True)
