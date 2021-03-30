from typing import Callable, overload
import functools
import torch
import scipy.special

def memoize_method(fcn: Callable) -> Callable:
    # alternative for lru_cache for memoizing a method without any arguments
    # lru_cache can produce memory leak for a method
    # this can be known by running test_ks_mem.py individually

    cachename = "__cch_" + fcn.__name__

    @functools.wraps(fcn)
    def new_fcn(self):
        if cachename in self.__dict__:
            return self.__dict__[cachename]
        else:
            res = fcn(self)
            self.__dict__[cachename] = res
            return res

    return new_fcn

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
