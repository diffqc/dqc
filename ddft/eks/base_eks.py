import torch
from abc import abstractmethod
import lintorch as lt

__all__ = ["BaseEKS"]

class BaseEKS(torch.nn.Module, lt.EditableModule):
    def __init__(self):
        super(BaseEKS, self).__init__()
        self._grid = None

    def set_grid(self, grid):
        self._grid = grid

    # should be called internally only
    @property
    def grid(self):
        if self._grid is None:
            raise RuntimeError("The grid must be set by set_grid first before calling grid")
        return self._grid

    @abstractmethod
    def forward(self, density):
        pass

    def potential(self, density):
        if density.requires_grad:
            xinp = density
        else:
            xinp = density.detach().requires_grad_()

        dv = self.grid.get_dvolume()
        with torch.enable_grad():
            y = self.forward(xinp) # (nbatch,nr)
            y = y * dv
            ysum = y.sum()
        grad_enabled = torch.is_grad_enabled()
        dx = torch.autograd.grad(ysum, (xinp,),
            create_graph=grad_enabled)[0]
        return dx / dv

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

    ############### editable module part ###############
    @abstractmethod
    def getfwdparams(self):
        pass

    @abstractmethod
    def setfwdparams(self, *params):
        pass

    def getparams(self, methodname):
        if methodname == "forward" or methodname == "__call__":
            return self.getfwdparams()
        elif methodname == "potential":
            return self.getfwdparams() + self.grid.getparams("get_dvolume")
        else:
            raise RuntimeError("The method %s has not been specified for getparams" % methodname)

    def setparams(self, methodname, *params):
        if methodname == "forward" or methodname == "__call__":
            self.setfwdparams(*params)
        elif method == "potential":
            nfwdparams = len(self.getfwdparams)
            self.setfwdparams(*params[:nfwdparams])
            self.grid.setparams("get_dvolume", *params[nfwdparams:])
        else:
            raise RuntimeError("The method %s has not been specified for setparams" % methodname)

class TensorEKS(BaseEKS):
    def __init__(self, tensor):
        super(TensorEKS, self).__init__()
        self.tensor = tensor

    def forward(self, density):
        return density * 0 + self.tensor

    def getfwdparams(self):
        return [self.tensor]

    def setfwdparams(self, *params):
        self.tensor, = params

class ConstEKS(BaseEKS):
    def __init__(self, c):
        super(ConstEKS, self).__init__()
        self.c = c

    def forward(self, density):
        return density * 0 + self.c

    def getfwdparams(self):
        return [self.c]

    def setfwdparams(self, *params):
        self.c, = params

######################## arithmetics ########################
class NegEKS(BaseEKS):
    def __init__(self, eks):
        super(NegEKS, self).__init__()
        self.eks = eks

    def set_grid(self, grid):
        self.eks.set_grid(grid)

    def forward(self, density):
        return -self.eks(density)

    def potential(self, density):
        return -self.eks.potential(density)

    def getfwdparams(self):
        return self.eks.getfwdparams()

    def setfwdparams(self, *params):
        self.eks.setfwdparams(*params)

class AddEKS(BaseEKS):
    def __init__(self, a, b):
        super(AddEKS, self).__init__()
        self.a = a
        self.b = b

    def set_grid(self, grid):
        self.a.set_grid(grid)
        self.b.set_grid(grid)

    def forward(self, density):
        return self.a(density) + self.b(density)

    def potential(self, density):
        return self.a.potential(density) + self.b.potential(density)

    def getfwdparams(self):
        return self.a.getfwdparams() + self.b.getfwdparams()

    def setfwdparams(self, *params):
        na = len(self.a.getfwdparams())
        self.a.setfwdparams(*params[:na])
        self.b.setfwdparams(*params[na:])

class MultEKS(BaseEKS):
    def __init__(self, a, b):
        super(MultEKS, self).__init__()
        self.a = a
        self.b = b

    def set_grid(self, grid):
        self.a.set_grid(grid)
        self.b.set_grid(grid)

    def forward(self, density):
        return self.a(density) * self.b(density)

    def getfwdparams(self):
        return self.a.getfwdparams() + self.b.getfwdparams()

    def setfwdparams(self, *params):
        na = len(self.a.getfwdparams())
        self.a.setfwdparams(*params[:na])
        self.b.setfwdparams(*params[na:])

class DivEKS(BaseEKS):
    def __init__(self, a, b):
        super(DivEKS, self).__init__()
        self.a = a
        self.b = b

    def set_grid(self, grid):
        self.a.set_grid(grid)
        self.b.set_grid(grid)

    def forward(self, density):
        return self.a(density) / self.b(density)

    def getfwdparams(self):
        return self.a.getfwdparams() + self.b.getfwdparams()

    def setfwdparams(self, *params):
        na = len(self.a.getfwdparams())
        self.a.setfwdparams(*params[:na])
        self.b.setfwdparams(*params[na:])

def _normalize(a):
    if isinstance(a, BaseEKS):
        return a
    elif isinstance(a, torch.Tensor):
        return TensorEKS()
    elif isinstance(a, int) or isinstance(a, float):
        return ConstEKS(a * 1.0)
    else:
        raise TypeError("Unknown type %s for operating with EKS object" % type(a))
