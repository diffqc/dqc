from typing import Callable, overload, TypeVar, Any, Mapping, Generator, Tuple
import functools
import torch
import scipy.special

T = TypeVar('T')
K = TypeVar('K')

def memoize_method(fcn: Callable[[Any], T]) -> Callable[[Any], T]:
    # alternative for lru_cache for memoizing a method without any arguments
    # lru_cache can produce memory leak for a method
    # this can be known by running test_ks_mem.py individually

    cachename = "__cch_" + fcn.__name__

    @functools.wraps(fcn)
    def new_fcn(self) -> T:
        if cachename in self.__dict__:
            return self.__dict__[cachename]
        else:
            res = fcn(self)
            self.__dict__[cachename] = res
            return res

    return new_fcn

def get_option(name: str, s: K, options: Mapping[K, T]) -> T:
    # get the value from dictionary of options, if not found, then raise an error
    if s in options:
        return options[s]
    else:
        raise ValueError(f"Unknown {name}: {s}. The available options are: {str(list(options.keys()))}")

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

@overload
def gaussian_int(n: int, alpha: float) -> float:
    ...

@overload
def gaussian_int(n: int, alpha: torch.Tensor) -> torch.Tensor:
    ...

def gaussian_int(n, alpha):
    # int_0^inf x^n exp(-alpha x^2) dx
    n1 = (n + 1) * 0.5
    return scipy.special.gamma(n1) / (2 * alpha ** n1)
