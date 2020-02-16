import torch
from abc import abstractmethod

class BaseEKS(torch.nn.Module):
    def __init__(self):
        super(BaseEKS, self).__init__()

    @abstractmethod
    def forward(self, density):
        pass

    def __add__(self, other):
        other = _normalize(other)
        return AddEKS(self, other)

    def __sub__(self, other):
        other = _normalize(other)
        return AddEKS(self, NegEKS(other))

    def __rsub__(self, other):
        other = _normalize(other)
        return AddEKS(NegEKS(self), other)

    def __mul__(self, other):
        other = _normalize(other)
        return MultEKS(self, other)

    def __div__(self, other):
        other = _normalize(other)
        return DivEKS(self, other)

    def __rdiv__(self, other):
        other = _normalize(other)
        return DivEKS(other, self)

    def __neg__(self):
        return NegEKS(self)

class VKS(torch.nn.Module):
    def __init__(self, eks_model, grid):
        super(VKS, self).__init__()
        self.eks_model = eks_model
        self.grid = grid

    def forward(self, x):
        assert x.ndim == 2, "The input to VKS module must be 2-dimensional tensor (nbatch, nrgrid)"
        if x.requires_grad:
            xinp = x
        else:
            xinp = x.clone().requires_grad_()

        with torch.enable_grad():
            y = self.eks_model(xinp) # (nbatch,nr)
            yint = self.grid.integralbox(y, dim=-1)
            ysum = yint.sum()
        grad_enabled = torch.is_grad_enabled()
        dx = torch.autograd.grad(ysum, (xinp,),
            create_graph=grad_enabled)[0]
        return dx # (same shape as x)


class TensorEKS(BaseEKS):
    def __init__(self, tensor):
        super(TensorEKS, self).__init__()
        self.tensor = tensor

    def forward(self, density):
        return density * 0 + self.tensor

class ConstEKS(BaseEKS):
    def __init__(self, c):
        super(ConstEKS, self).__init__()
        self.c = c

    def forward(self, density):
        return density * 0 + self.c

######################## arithmetics ########################
class NegEKS(BaseEKS):
    def __init__(self, eks):
        super(NegEKS, self).__init__()
        self.eks = eks

    def forward(self, density):
        return -self.eks(density)

class AddEKS(BaseEKS):
    def __init__(self, a, b):
        super(AddEKS, self).__init__()
        self.a = a
        self.b = b

    def forward(self, density):
        return self.a(density) + self.b(density)

class MultEKS(BaseEKS):
    def __init__(self, a, b):
        super(MultEKS, self).__init__()
        self.a = a
        self.b = b

    def forward(self, density):
        return self.a(density) * self.b(density)

class DivEKS(BaseEKS):
    def __init__(self, a, b):
        super(DivEKS, self).__init__()
        self.a = a
        self.b = b

    def forward(self, density):
        return self.a(density) / self.b(density)


def _normalize(a):
    if isinstance(a, BaseEKS):
        return a
    elif isinstance(a, torch.Tensor):
        return TensorEKS()
    elif isinstance(a, int) or isinstance(a, float):
        return ConstEKS(a * 1.0)
    else:
        raise TypeError("Unknown type %s for operating with EKS object" % type(a))
