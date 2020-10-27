import torch
from abc import abstractmethod
import xitorch as xt

__all__ = ["BaseEKS"]

class BaseEKS(xt.EditableModule):
    def __init__(self):
        super(BaseEKS, self).__init__()
        self._grid = None

    def set_grid(self, grid):
        self._grid = grid

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # should be called internally only
    @property
    def grid(self):
        if self._grid is None:
            raise RuntimeError("The grid must be set by set_grid first before calling grid")
        return self._grid

    @abstractmethod
    def forward(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        """
        Returns the energy per unit volume at each point in the grid.
        """
        pass

    def potential(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        """
        Returns the potential at each point in the grid.
        """
        assert (gradn_up is None) == (gradn_dn is None)
        if gradn_up is not None:
            raise RuntimeError("Automatic potential finder with gradn is not "
                               "available yet. Please implement it manually in "
                               "class %s" % self.__class__.__name__)
        if density_up.requires_grad:
            xinp_u = density_up
        else:
            xinp_u = density_up.detach().requires_grad_()

        if density_dn.requires_grad:
            xinp_d = density_dn
        else:
            xinp_d = density_dn.detach().requires_grad_()

        dv = self.grid.get_dvolume()
        # the factor will be cancelled out, so removing it from graph will
        # increase the numerical stability
        factor = dv.min().detach()
        # making the minimum to be 1 seem to make it more robust
        # this is probably because it will be multiplied with the potential
        # which could be very small due to small density and could cause
        # underflow
        dv = dv / factor
        dv = dv.expand(density_up.shape[0], -1)

        with torch.enable_grad():
            y = self.forward(xinp_u, xinp_d) # (nbatch,nr)

        dx_u, dx_d = torch.autograd.grad(y, (xinp_u, xinp_d), grad_outputs=dv,
            create_graph=torch.is_grad_enabled())
        return dx_u / dv, dx_d / dv

    # properties
    @property
    def need_gradn(self):
        return False

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
    def getfwdparamnames(self, prefix=""):
        return []

    def getparamnames(self, methodname, prefix=""):
        if methodname == "forward" or methodname == "__call__":
            return self.getfwdparamnames(prefix=prefix)
        elif methodname == "potential":
            return self.getfwdparamnames(prefix=prefix) + self.grid.getparamnames("get_dvolume", prefix=prefix+"grid.")
        else:
            raise KeyError("Getparamnames has no %s method" % methodname)

class TensorEKS(BaseEKS):
    def __init__(self, tensor):
        super(TensorEKS, self).__init__()
        self.tensor = tensor

    def forward(self, density):
        return density * 0 + self.tensor

    def getfwdparamnames(self, prefix=""):
        return [prefix+"tensor"]


class ConstEKS(BaseEKS):
    def __init__(self, c):
        super(ConstEKS, self).__init__()
        self.c = c

    def forward(self, density):
        return density * 0 + self.c

    def getfwdparamnames(self, prefix=""):
        return [prefix+"c"]


######################## arithmetics ########################
class NegEKS(BaseEKS):
    def __init__(self, eks):
        super(NegEKS, self).__init__()
        self.eks = eks

    def set_grid(self, grid):
        self.eks.set_grid(grid)

    def forward(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        return -self.eks(density_up, density_dn, gradn_up, gradn_dn)

    def potential(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        return -self.eks.potential(density_up, density_dn, gradn_up, gradn_dn)

    @property
    def need_gradn(self):
        return self.eks.need_gradn

    def getfwdparamnames(self, prefix=""):
        return self.eks.getfwdparamnames(prefix=prefix+"eks.")

class AddEKS(BaseEKS):
    def __init__(self, a, b):
        super(AddEKS, self).__init__()
        self.a = a
        self.b = b
        self._need_gradn = self.a.need_gradn or self.b.need_gradn

    def set_grid(self, grid):
        self.a.set_grid(grid)
        self.b.set_grid(grid)
        self._grid = grid

    def forward(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        return self.a(density_up, density_dn, gradn_up, gradn_dn) + \
               self.b(density_up, density_dn, gradn_up, gradn_dn)

    def potential(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        apot = self.a.potential(density_up, density_dn, gradn_up, gradn_dn)
        bpot = self.b.potential(density_up, density_dn, gradn_up, gradn_dn)
        return (apot[0] + bpot[0]), (apot[1] + bpot[1])

    @property
    def need_gradn(self):
        return self._need_gradn

    def getfwdparamnames(self, prefix=""):
        return self.a.getfwdparamnames(prefix=prefix+"a.") + \
               self.b.getfwdparamnames(prefix=prefix+"b.")

class MultEKS(BaseEKS):
    def __init__(self, a, b):
        super(MultEKS, self).__init__()
        self.a = a
        self.b = b
        self._need_gradn = self.a.need_gradn or self.b.need_gradn

    def set_grid(self, grid):
        self.a.set_grid(grid)
        self.b.set_grid(grid)

    def forward(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        return self.a(density_up, density_dn, gradn_up, gradn_dn) * \
               self.b(density_up, density_dn, gradn_up, gradn_dn)

    @property
    def need_gradn(self):
        return self._need_gradn

    def getfwdparamnames(self, prefix=""):
        return self.a.getfwdparamnames(prefix=prefix+"a.") + \
               self.b.getfwdparamnames(prefix=prefix+"b.")

class DivEKS(BaseEKS):
    def __init__(self, a, b):
        super(DivEKS, self).__init__()
        self.a = a
        self.b = b
        self._need_gradn = self.a.need_gradn or self.b.need_gradn

    def set_grid(self, grid):
        self.a.set_grid(grid)
        self.b.set_grid(grid)

    def forward(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        return self.a(density_up, density_dn, gradn_up, gradn_dn) / \
               self.b(density_up, density_dn, gradn_up, gradn_dn)

    @property
    def need_gradn(self):
        return self._need_gradn

    def getfwdparamnames(self, prefix=""):
        return self.a.getfwdparamnames(prefix=prefix+"a.") + \
               self.b.getfwdparamnames(prefix=prefix+"b.")

def _normalize(a):
    if isinstance(a, BaseEKS):
        return a
    elif isinstance(a, torch.Tensor):
        return TensorEKS()
    elif isinstance(a, int) or isinstance(a, float):
        return ConstEKS(a * 1.0)
    else:
        raise TypeError("Unknown type %s for operating with EKS object" % type(a))
