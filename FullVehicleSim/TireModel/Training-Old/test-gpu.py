import torch
if torch.backends.mps.is_available():
    print("MPS backend is available!")
else:
    print("MPS backend is not available. Please check your PyTorch installation.")