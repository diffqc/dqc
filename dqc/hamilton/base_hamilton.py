from __future__ import annotations
import torch
import xitorch as xt
from abc import abstractmethod, abstractproperty
from typing import List, Optional, Union, overload
from dqc.grid.base_grid import BaseGrid
from dqc.xc.base_xc import BaseXC
from dqc.utils.datastruct import SpinParam

class BaseHamilton(xt.EditableModule):
    """
    Hamilton is a class that provides the LinearOperator of the Hamiltonian
    components.
    """
    @abstractproperty
    def nao(self) -> int:
        """
        Returns the number of atomic orbital basis
        """
        pass

    @abstractmethod
    def build(self) -> BaseHamilton:
        """
        Construct the elements needed for the Hamiltonian.
        Heavy-lifting operations should be put here.
        """
        pass

    @abstractmethod
    def get_kinnucl(self) -> xt.LinearOperator:
        """
        Returns the LinearOperator of the one-electron operator (i.e. kinetic
        and nuclear attraction).
        """
        # return: (*BH, nao, nao)
        pass

    @abstractmethod
    def get_overlap(self) -> xt.LinearOperator:
        """
        Returns the LinearOperator representing the overlap of the basis.
        """
        # return: (*BH, nao, nao)
        pass

    @abstractmethod
    def get_elrep(self, dm: torch.Tensor) -> xt.LinearOperator:
        """
        Obtains the LinearOperator of the Coulomb electron repulsion operator.
        """
        # dm: (*BD, nao, nao)
        # return: (*BDH, nao, nao)
        pass

    @abstractmethod
    def ao_orb2dm(self, orb: torch.Tensor, orb_weight: torch.Tensor) -> torch.Tensor:
        """
        Convert the atomic orbital to the density matrix.
        """
        # orb: (*BO, nao, norb)
        # orb_weight: (*BW, norb)
        # return: (*BOWH, nao, nao)
        pass

    @abstractmethod
    def aodm2dens(self, dm: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        Get the density value in the Cartesian coordinate.
        """
        # dm: (*BD, nao, nao)
        # xyz: (*BR, ndim)
        # return: (*BRD)
        pass

    ############### grid-related ###############
    @abstractmethod
    def setup_grid(self, grid: BaseGrid, xc: Optional[BaseXC] = None) -> None:
        """
        Setup the basis (with its grad) in the spatial grid and prepare the
        gradient of atomic orbital according to the ones required by the xc.
        If xc is not given, then only setup the grid with ao (without any gradients
        of ao)
        """
        pass

    @abstractmethod
    def get_vext(self, vext: torch.Tensor) -> xt.LinearOperator:
        r"""
        Returns a LinearOperator of the external potential in the grid.

        .. math::
            \mathbf{V}_{ij} = \int b_i(\mathbf{r}) V(\mathbf{r}) b_j(\mathbf{r})\ d\mathbf{r}
        """
        # vext: (*BR, ngrid)
        # returns: (*BRH, nao, nao)
        pass

    @abstractmethod
    def get_grad_vext(self, grad_vext: torch.Tensor) -> xt.LinearOperator:
        r"""
        Returns a LinearOperator of the external gradient potential in the grid.

        .. math::
            \mathbf{G}_{ij} = \int b_i(\mathbf{r}) \mathbf{G}(\mathbf{r}) \cdot \nabla b_j(\mathbf{r})\ d\mathbf{r}
        """
        # grad_vext: (*BR, ngrid, ndim)
        # return: (*BRH, nao, nao)
        pass

    @abstractmethod
    def get_lapl_vext(self, lapl_vext: torch.Tensor) -> xt.LinearOperator:
        r"""
        Returns a LinearOperator of the external laplace potential in the grid.

        .. math::
            \mathbf{L}_{ij} = \int b_i(\mathbf{r}) L(\mathbf{r}) \cdot \nabla^2 b_j(\mathbf{r})\ d\mathbf{r}
        """
        # lapl_vext: (*BR, ngrid)
        # return: (*BRH, nao, nao)
        pass

    @overload
    def get_vxc(self, dm: SpinParam[torch.Tensor]) -> SpinParam[xt.LinearOperator]:
        ...

    @overload
    def get_vxc(self, dm: torch.Tensor) -> xt.LinearOperator:
        ...

    @abstractmethod
    def get_vxc(self, dm):
        """
        Returns a LinearOperator for the exchange-correlation potential.
        """
        # dm: (*BD, nao, nao)
        # return: (*BDH, nao, nao)
        # TODO: check if what we need for Meta-GGA involving kinetics and for
        # exact-exchange
        pass

    @abstractmethod
    def get_exc(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Returns the exchange-correlation energy using the xc object given in
        ``.setup_grid()``
        """
        # dm: (*BD, nao, nao)
        # return: (*BDH)
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        Return the paramnames
        """
        pass
