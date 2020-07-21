import torch
import numpy as np
from abc import abstractmethod

"""
This file contains the cumulative sum quadrature functions.
The functions are usually used in solve_poisson method in grid objects.
"""

class BaseCumSumQuad(object):
    @abstractmethod
    def cumsum(y):
        pass

class CumSumQuad(BaseCumSumQuad):
    """
    CumSumQuad class is a class to perform the cumulative sum using quadrature
    rules.
    Example:

    >>> x = torch.linspace(0, 1, 1000)
    >>> y = torch.exp(-x*x/2.0)
    >>> obj = CumSumQuad(x, side="left", method="trapz")
    >>> cumsum = obj.cumsum(y)

    Arguments
    ---------
    * x: torch.tensor with shape (*, nx)
        The location of the integration points
    * side: ("left", "right", "both")
        If "left", then it performs cumulative sum from the left, i.e. int_0^x y(x) dx.
        If "right", it performs from the right, i.e. int_x^inf y(x) dx.
        If "both", then it is right + left.
    * method: str
        Method of quadrature.
    """
    def __init__(self, x, side="left", method="trapz"):
        if method == "trapz":
            self.quad = TrapzCumSumQuad(x, side=side)
        elif method == "simpson":
            self.quad = SimpsonCumSumQuad(x, side=side)
        else:
            raise RuntimeError("Unknown method: %s" % method)

    def cumsum(self, y):
        return self.quad.cumsum(y)

######################## weight-based cumsum quadrature ########################
class WeightBasedCumSumQuad(BaseCumSumQuad):
    def __init__(self, x, side="left"):
        # x: (*, nx)
        # returns: (*, nx, nx)
        self.w = 0
        side = side.lower()
        xshape = x.shape
        nx = xshape[-1]
        x = x.view(-1, nx)
        if side == "left" or side == "both":
            w = self.get_weights(x) # (*, nx, nx)
            self.w = self.w + w
        if side == "right" or side == "both":
            w = torch.flip(self.get_weights(torch.flip(-x, dims=[-1])), dims=[-1,-2]) # (*, nx, nx)
            self.w = self.w + w
        self.w = self.w.view(*xshape[:-1], nx, nx)

    @abstractmethod
    def get_weights(self, x):
        pass

    def cumsum(self, y):
        # y: (*, nx)
        # w: (*, nx, nx)
        # returns: (*, nx)
        return torch.sum(y.unsqueeze(-2) * self.w, dim=-1)

class TrapzCumSumQuad(WeightBasedCumSumQuad):
    def get_weights(self, x):
        return get_trapz_weights(x)

class SimpsonCumSumQuad(WeightBasedCumSumQuad):
    def get_weights(self, x):
        return get_simpson_weights(x)

@torch.jit.script
def get_trapz_weights(x):
    # x: (nb, nx)
    # returns: (nb, nx, nx)
    half_dx = (x[:,1:] - x[:,:-1]) * 0.5 # (nb, nx-1)
    nx = x.shape[-1]
    res = torch.zeros((x.shape[0], nx, nx), dtype=x.dtype, device=x.device)
    for i in range(1,nx):
        res[:,i:,i-1:i+1] += half_dx[:,i-1:i].unsqueeze(-1)
    return res

@torch.jit.script
def get_simpson_weights(x):
    # x: (nb, nx)
    # returns: (nb, nx, nx)
    # ref: https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
    h = x[:,1:] - x[:,:-1] # (nb, nx-1)
    h1 = h[:,1::2] #  (nb, (nx-2)//2)
    h0 = h[:,:-1:2] # (nb, (nx-2)//2)
    h1_2 = h1*h1
    h0_2 = h0*h0
    h1_3 = h1_2*h1
    h0_3 = h0_2*h0
    alpha = (2*h1_3 - h0_3 + 3*h0*h1_2) / (6*h1*(h1+h0)) # (nb, (nx-2)//2)
    eta   = (2*h0_3 - h1_3 + 3*h1*h0_2) / (6*h0*(h1+h0)) # (nb, (nx-2)//2)
    beta  = (h1_3 + h0_3 + 3*h1*h0*(h1+h0)) / (6*h1*h0)  # (nb, (nx-2)//2)
    # last part (for odd parts only)
    hN1 = h[:,2::2] # (nb, (nx-3)//2)
    hN2 = h[:,1:-1:2] # (nb, (nx-3)//2)
    alpha_l = (2*hN1*hN1 + 3*hN1*hN2) / (6*(hN1 + hN2))
    eta_l   = hN1*hN1*hN1 / (6*hN2*(hN1 + hN2))
    beta_l  = (hN1*hN1 + 3*hN1*hN2) / (6*hN2)

    nx = x.shape[-1]
    res = torch.zeros((x.shape[0], nx, nx), dtype=x.dtype, device=x.device)
    for i in range(2,nx,2):
        j = i//2-1
        res[:,i:,i-2] +=   eta[:,j:j+1]
        res[:,i:,i-1] +=  beta[:,j:j+1]
        res[:,i:,i  ] += alpha[:,j:j+1]
    for i in range(3,nx,2): # last part of the odd parts
        j = i//2-1
        res[:,i,i-2] +=  -eta_l[:,j]
        res[:,i,i-1] +=  beta_l[:,j]
        res[:,i,i  ] += alpha_l[:,j]

    # trapezoidal rule for the part with N=1 interval
    res[:,1,:2] = 0.5 * h[:,0]

    return res

if __name__ == "__main__":
    x = torch.logspace(-4, 1, 100)
    # x = torch.linspace(0, 9, 100)
    y = torch.exp(-x*x/2.0)
    side = "right"
    method = "simpson"
    # method = "trapz"

    if side == "left":
        ycumsum = np.sqrt(np.pi*0.5) * torch.erf(x/np.sqrt(2))
    else:
        ycumsum = np.sqrt(np.pi*0.5) * torch.erfc(x/np.sqrt(2))
    cumsum = CumSumQuad(x, side=side, method=method).cumsum(y)
    print((cumsum-ycumsum).abs().mean())

    import matplotlib.pyplot as plt
    plt.plot(x, ycumsum)
    plt.plot(x, cumsum)
    plt.show()
