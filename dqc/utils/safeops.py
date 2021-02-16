import math
import torch
from typing import Union, Optional

eps = 1e-12

def safepow(a: torch.Tensor, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if torch.any(a < 0):
        raise RuntimeError("safepow only works for positive base")
    base = torch.sqrt(a * a + eps * eps)  # soft clip
    return base ** p

def safenorm(a: torch.Tensor, dim: int, eps: float = 1e-15) -> torch.Tensor:
    # calculate the 2-norm safely
    return torch.sqrt(torch.sum(a * a + eps * eps, dim=dim))

def occnumber(a: Union[int, float],
              n: Optional[int] = None,
              dtype: torch.dtype = torch.double,
              device: torch.device = torch.device('cpu')) -> torch.Tensor:
    # returns the occupation number (maxed at 1) where the total sum of the
    # output equals to a with length of the output is n

    # get the ceiling and flooring of a
    if isinstance(a, int):
        ceil_a: int = a
        floor_a: int = a
    else:
        ceil_a = int(math.ceil(a))
        floor_a = int(math.floor(a))

    # get the length of the tensor output
    if n is None:
        nlength = ceil_a
    else:
        nlength = n
        assert nlength >= ceil_a, "The length of occupation number must be at least %d" % ceil_a

    res = torch.zeros(nlength, dtype=dtype, device=device)
    res[:floor_a] = 1
    if ceil_a > floor_a:
        res[-1] = a - floor_a
    return res
