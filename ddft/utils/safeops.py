import torch

eps = 1e-12

def safepow(a, p, eps=1e-12):
    if torch.any(a < 0):
        raise RuntimeError("safepow only works for positive base")
    base = torch.sqrt(a*a + eps*eps) # soft clip
    return base**p
