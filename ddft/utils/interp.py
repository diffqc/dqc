import numpy as np
import torch

def searchsorted(a, v, side="left"):
    idx = np.searchsorted(a.detach().numpy(), v.detach().numpy(), side=side)
    return torch.tensor(idx, dtype=torch.long)
