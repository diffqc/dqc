import torch
from abc import abstractmethod
import xitorch as xt

__all__ = ["BaseEKS"]

class BaseEKS(xt.EditableModule):
    def __init__(self):
        super(BaseEKS, self).__init__()
        self._grid = None
        self._hmodel = None

    def set_grid(self, grid):
        self._grid = grid

    def set_hmodel(self, hmodel):
        self._hmodel = hmodel
        self._grid = hmodel.grid

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # should be called internally only
    @property
    def grid(self):
        if self._grid is None:
            raise RuntimeError("The grid must be set by set_grid or set_hmodel first before calling grid")
        return self._grid

    @property
    def hmodel(self):
        if self._hmodel is None:
            raise RuntimeError("The hmodel must be set by set_hmodel first before calling grid")
        return self._hmodel

    @abstractmethod
    def forward(self, densinfo_u, densinfo_d):
        """
        Returns the energy per unit volume at each point in the grid.
        """
        pass

    # to be deprecated
    def potential(self, densinfo_u, densinfo_d):
        """
        Returns the potential at each point in the grid.
        """
        gradn_up = densinfo_u.gradn
        gradn_dn = densinfo_d.gradn

        assert (gradn_up is None) == (gradn_dn is None)
        if gradn_up is not None:
            raise RuntimeError("Automatic potential finder with gradn is not "
                               "available yet. Please implement it manually in "
                               "class %s" % self.__class__.__name__)

        if densinfo_u.density.requires_grad:
            densinfo_u0 = densinfo_u
        else:
            newdens = densinfo_u.density.detach().requires_grad_()
            densinfo_u0 = densinfo_u._replace(density=newdens)

        if densinfo_d.density.requires_grad:
            densinfo_d0 = densinfo_d
        else:
            newdens = densinfo_d.density.detach().requires_grad_()
            densinfo_d0 = densinfo_u._replace(density=newdens)

        with torch.enable_grad():
            y = self.forward(densinfo_u0, densinfo_d0)

        dx_u, dx_d = torch.autograd.grad(
            outputs=y,
            inputs=(densinfo_u0.density, densinfo_d0.density),
            grad_outputs=torch.ones_like(y),
            create_graph=torch.is_grad_enabled())

        return dx_u, dx_d

    # to be made abstract
    def potential_linop(self, densinfo_u, densinfo_d):
        if densinfo_u is densinfo_d:
            potu, potd = self.potential(densinfo_u, densinfo_d)
            u = self.hmodel.get_vext(potu)
            return u, u
        else:
            potu, potd = self.potential(densinfo_u, densinfo_d)
            u = self.hmodel.get_vext(potu)
            d = self.hmodel.get_vext(potd)
        return u, d

    # properties
    @property
    def need_gradn(self):
        return False

    def __add__(self, other):
        other = _normalize(other)
        return AddEKS(self, other)

    ############### editable module part ###############
    @abstractmethod
    def getfwdparamnames(self, prefix=""):
        return []

    def getparamnames(self, methodname, prefix=""):
        if methodname == "forward" or methodname == "__call__":
            return self.getfwdparamnames(prefix=prefix)
        elif methodname == "potential":
            return self.getfwdparamnames(prefix=prefix)
        elif methodname == "potential_linop":
            return self.getfwdparamnames(prefix=prefix) + \
                   self.hmodel.getparamnames("get_vext", prefix=prefix+"hmodel.")
        else:
            raise KeyError("Getparamnames has no %s method" % methodname)


######################## arithmetics ########################
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

    def set_hmodel(self, hmodel):
        self.a.set_hmodel(hmodel)
        self.b.set_hmodel(hmodel)
        self._grid = hmodel.grid
        self._hmodel = hmodel

    def forward(self, densinfo_u, densinfo_d):
        return self.a(densinfo_u, densinfo_d) + \
               self.b(densinfo_u, densinfo_d)

    def potential(self, densinfo_u, densinfo_d):
        apot = self.a.potential(densinfo_u, densinfo_d)
        bpot = self.b.potential(densinfo_u, densinfo_d)
        return (apot[0] + bpot[0]), (apot[1] + bpot[1])

    def potential_linop(self, densinfo_u, densinfo_d):
        vau, vad = self.a.potential_linop(densinfo_u, densinfo_d)
        vbu, vbd = self.b.potential_linop(densinfo_u, densinfo_d)
        return vau + vbu, vad + vbd

    @property
    def need_gradn(self):
        return self._need_gradn

    def getfwdparamnames(self, prefix=""):
        return self.a.getfwdparamnames(prefix=prefix+"a.") + \
               self.b.getfwdparamnames(prefix=prefix+"b.")

class BaseLDA(BaseEKS):
    @abstractmethod
    def energy_unpol(self, rho):
        """
        Returns energy density given the density for unpolarized case.

        Arguments
        ---------
        rho: torch.Tensor
            The total density value.

        Returns
        -------
        ene: torch.Tensor
            The energy density per unit volume with the same shape as ``rho``.
        """
        pass

    @abstractmethod
    def energy_pol(self, rho_u, rho_d):
        """
        Returns energy density given the density for polarized case.

        Arguments
        ---------
        rho_u: torch.Tensor
            The spin-up density value.
        rho_d: torch.Tensor
            The spin-down density value.

        Returns
        -------
        ene: torch.Tensor
            The energy density per unit volume with the same shape as ``rho``.
        """
        pass

    @abstractmethod
    def potential_unpol(self, rho):
        """
        Returns vxc given the density for unpolarized case.

        Arguments
        ---------
        rho: torch.Tensor
            The total density value.

        Returns
        -------
        vrho: torch.Tensor
            The derivative of energy density per unit volume w.r.t. density.
            Has the same shape as ``rho``.
        """
        pass

    @abstractmethod
    def potential_pol(self, rho_u, rho_d):
        """
        Returns vxc given the density for polarized case.

        Arguments
        ---------
        rho_u: torch.Tensor
            The spin-up density value.
        rho_d: torch.Tensor
            The spin-down density value.

        Returns
        -------
        vrho: torch.Tensor
            The derivative of energy density per unit volume w.r.t. density.
            Has shape of ``(2, *rho.shape)``.
        """
        pass

    @abstractmethod
    def getfwdparamnames(self, prefix=""):
        pass

    def forward(self, densinfo_u, densinfo_d):
        if id(densinfo_u) == id(densinfo_d):
            rho = densinfo_u.density + densinfo_d.density
            ev = self.energy_unpol(rho)
        else:
            ev = self.energy_pol(densinfo_u.density, densinfo_d.density)
        return ev

    def potential_linop(self, densinfo_u, densinfo_d):
        # obtain the potential as a function of space
        if id(densinfo_u) == id(densinfo_d):
            rho = densinfo_u.density + densinfo_d.density
            vxc_u = self.potential_unpol(rho)

            vxc_ulinop = self.hmodel.get_vext(vxc_u)
            return vxc_ulinop, vxc_ulinop
        else:
            vxc_ud = self.potential_pol(densinfo_u.density, densinfo_d.density)
            vxc_u = vxc_ud[0]
            vxc_d = vxc_ud[1]

            # get the linear operator
            vxc_ulinop = self.hmodel.get_vext(vxc_u)
            vxc_dlinop = self.hmodel.get_vext(vxc_d)
            return vxc_ulinop, vxc_dlinop

class BaseGGA(BaseEKS):
    @abstractmethod
    def energy_unpol(self, rho, sigma):
        """
        Returns energy density given the density for unpolarized case.

        Arguments
        ---------
        rho: torch.Tensor
            The total density value.
        sigma: torch.Tensor
            Contracted density gradient, i.e.
            ``gradx(n)**2 + grady(n)**2 + gradz(n)**2``.
            Must have the same shape as ``rho``.

        Returns
        -------
        ene: torch.Tensor
            The energy density per unit volume with the same shape as ``rho``.
        """
        pass

    @abstractmethod
    def energy_pol(self, rho_u, rho_d,
                   sigma_uu, sigma_ud, sigma_dd):
        """
        Returns energy density given the density for polarized case.
        All arguments must have the same shape

        Arguments
        ---------
        rho_u: torch.Tensor
            The spin-up density value.
        rho_d: torch.Tensor
            The spin-down density value.
        sigma_uu: torch.Tensor
            Contracted density gradient for spin-up and spin-up, i.e.
            ``gradx(nu)*gradx(nu) + grady(nu)*grady(nu) + gradz(nu)*gradz(nu)``.
        sigma_ud: torch.Tensor
            Contracted density gradient for spin-up and spin-down.
        sigma_dd: torch.Tensor
            Contracted density gradient for spin-down and spin-down.

        Returns
        -------
        ene: torch.Tensor
            The energy density per unit volume with the same shape as ``rho_u``.
        """
        pass

    @abstractmethod
    def potential_unpol(self, rho, sigma):
        """
        Returns vxc given the density for unpolarized case.

        Arguments
        ---------
        rho: torch.Tensor
            The total density value.
        sigma: torch.Tensor
            Contracted density gradient, i.e.
            ``gradx(n)**2 + grady(n)**2 + gradz(n)**2``.

        Returns
        -------
        vrho: torch.Tensor
            The derivative of energy density per unit volume w.r.t. density.
            Has the same shape as ``rho``.
        vsigma: torch.Tensor
            The derivative of energy density per unit volume w.r.t. contracted
            gradient density (or sigma).
            Has the same shape as ``sigma``.
        """
        pass

    @abstractmethod
    def potential_pol(self, rho_u, rho_d,
                        sigma_uu, sigma_ud, sigma_dd):
        """
        Returns vxc given the density for polarized case.

        Arguments
        ---------
        rho_u: torch.Tensor
            The spin-up density value.
        rho_d: torch.Tensor
            The spin-down density value.
        sigma_uu: torch.Tensor
            Contracted density gradient for spin-up and spin-up, i.e.
            ``gradx(nu)*gradx(nu) + grady(nu)*grady(nu) + gradz(nu)*gradz(nu)``.
        sigma_ud: torch.Tensor
            Contracted density gradient for spin-up and spin-down.
        sigma_dd: torch.Tensor
            Contracted density gradient for spin-down and spin-down.

        Returns
        -------
        vrho: torch.Tensor
            The derivative of energy density per unit volume w.r.t. density.
            Has shape of ``(2, *rho.shape)``.
        vsigma: torch.Tensor
            The derivative of energy density per unit volume w.r.t. contracted
            density gradient.
            Has shape of ``(3, *sigma.shape)``
        """
        pass

    @abstractmethod
    def getfwdparamnames(self, prefix=""):
        pass

    def forward(self, densinfo_u, densinfo_d):
        if id(densinfo_u) == id(densinfo_d):  # unpolarized
            rho = densinfo_u.density + densinfo_d.density
            gradn = densinfo_u.gradn + densinfo_d.gradn  # (3, ...)
            sigma = torch.sum(gradn * gradn, dim=0)  # (...)
            ev = self.energy_unpol(rho, sigma)
        else:
            rho_u = densinfo_u.density
            rho_d = densinfo_d.density
            grad_u = densinfo_u.gradn
            grad_d = densinfo_d.gradn
            sigma_uu = torch.sum(grad_u * grad_u, dim=0)
            sigma_ud = torch.sum(grad_u * grad_d, dim=0)
            sigma_dd = torch.sum(grad_d * grad_d, dim=0)
            ev = self.energy_pol(rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd)
        return ev

    def potential_linop(self, densinfo_u, densinfo_d):
        # obtain the potential and grad potential as a function of space
        if id(densinfo_u) == id(densinfo_d):  # unpolarized
            rho = densinfo_u.density + densinfo_d.density
            gradn = densinfo_u.gradn + densinfo_d.gradn  # (3, ...)
            sigma = torch.sum(gradn * gradn, dim=0)  # (...)
            vrho, vsigma = self.potential_unpol(rho, sigma)
            vxc_u = vrho
            grad_vxc_u = 2 * vsigma * gradn

            vxc_ulinop = self.hmodel.get_vext(vxc_u) + \
                         self.hmodel.get_grad_vext(grad_vxc_u)
            return vxc_ulinop, vxc_ulinop
        else:
            rho_u = densinfo_u.density
            rho_d = densinfo_d.density
            grad_u = densinfo_u.gradn
            grad_d = densinfo_d.gradn
            sigma_uu = torch.sum(grad_u * grad_u, dim=0)
            sigma_ud = torch.sum(grad_u * grad_d, dim=0)
            sigma_dd = torch.sum(grad_d * grad_d, dim=0)

            # calculate the potential and grad potential
            vrho, vsigma = self.potential_pol(rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd)
            vxc_u, vxc_d = vrho
            grad_vxc_u = 2 * vsigma[0] * grad_u + vsigma[1] * grad_d
            grad_vxc_d = 2 * vsigma[2] * grad_d + vsigma[1] * grad_u

            vxc_ulinop = self.hmodel.get_vext(vxc_u) + \
                         self.hmodel.get_grad_vext(grad_vxc_u)
            vxc_dlinop = self.hmodel.get_vext(vxc_d) + \
                         self.hmodel.get_grad_vext(grad_vxc_d)
            return vxc_ulinop, vxc_dlinop

def _normalize(a):
    if isinstance(a, BaseEKS):
        return a
    else:
        raise TypeError("Unknown type %s for operating with EKS object" % type(a))
