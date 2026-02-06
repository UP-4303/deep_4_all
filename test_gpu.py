import numpy as np
import torch

print("torch:", getattr(torch, "__version__", "not installed"))
print("cuda available:", torch.cuda.is_available() if hasattr(torch, "cuda") else "unknown")
print("cuda version:", getattr(torch.version, "cuda", None))

torch.__config__.show()

print("NumPy est installé, version :", np.__version__)

# Installer les bibliothèques nécessaires Torch Geometric

print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), "cuda_ver:", torch.version.cuda)

