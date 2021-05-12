from typing import Generator, Tuple
import torch

__all__ = ["chunkify", "get_memory", "get_dtype_memsize"]

def chunkify(a: torch.Tensor, dim: int, maxnumel: int) -> \
        Generator[Tuple[torch.Tensor, int, int], None, None]:
    """
    Returns an iterator that splits the tensor ``a`` into several chunks along
    the dimension given by ``dim``.

    Arguments
    ---------
    a: torch.Tensor
        The big tensor to be splitted into chunks.
    dim: int
        The dimension where the tensor would be splitted.
    maxnumel: int
        Maximum number of elements in a chunk.
    """
    dim = a.ndim + dim if dim < 0 else dim

    numel = a.numel()
    dimnumel = a.shape[dim]
    nondimnumel = numel // dimnumel
    if maxnumel < nondimnumel:
        msg = "Cannot split the tensor of shape %s along dimension %s with maxnumel %d" % \
              (a.shape, dim, maxnumel)
        raise RuntimeError(msg)

    csize = min(maxnumel // nondimnumel, dimnumel)
    ioffset = 0
    lslice = (slice(None, None, None),) * dim
    rslice = (slice(None, None, None),) * (a.ndim - dim - 1)
    while ioffset < dimnumel:
        iend = ioffset + csize
        yield a[(lslice + (slice(ioffset, iend, None),) + rslice)], ioffset, iend
        ioffset = iend

def get_memory(a: torch.Tensor) -> int:
    # returns the size of the tensor in bytes
    return a.numel() * get_dtype_memsize(a)

def get_dtype_memsize(a: torch.Tensor) -> int:
    # returns the size of each element in the tensor in bytes
    if a.dtype == torch.float64 or a.dtype == torch.int64:
        size = 8
    elif a.dtype == torch.float32 or a.dtype == torch.int32:
        size = 4
    elif a.dtype == torch.bool:
        size = 1
    else:
        raise TypeError("Unknown tensor type: %s" % a.dtype)
    return size
