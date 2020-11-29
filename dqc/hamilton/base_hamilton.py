import torch
import xitorch as xt
from abc import abstractmethod
from typing import List
from dqc.grid.base_grid import BaseGrid
from dqc.xc.base_xc import BaseXC

class BaseHamilton(xt.EditableModule):
    """
    Hamilton is a class that provides the LinearOperator of the Hamiltonian
    components.
    """
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
        # orb: (*BO, norb, nao)
        # orb_weight: (*BW, norb)
        # return: (*BOWH, nao, nao)
        pass

    ############### grid-related ###############
    @abstractmethod
    def setup_grid(self, grid: BaseGrid, xcfamily: int = 0) -> None:
        """
        Setup the basis (with its grad) in the spatial grid.
        """
        pass

    @abstractmethod
    def get_vext(self, vext: torch.Tensor) -> xt.LinearOperator:
        r"""
        Returns a LinearOperatorof the external potential in the grid.

        .. math::
            \mathbf{V}_{ij} = \int b_i(\mathbf{r}) V(\mathbf{r}) b_j(\mathbf{r})\ d\mathbf{r}
        """
        # vext: (*BR, ngrid)
        # returns: (*BRH, nao, nao)
        pass

    @abstractmethod
    def get_grad_vext(self, grad_vext: torch.Tensor) -> xt.LinearOperator:
        r"""
        Returns a LinearOperatorof the external gradient potential in the grid.

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

    @abstractmethod
    def get_vxc(self, xc: BaseXC, dm: torch.Tensor) -> xt.LinearOperator:
        """
        Returns a LinearOperator for the exchange-correlation potential.
        """
        # dm: (*BD, nao, nao)
        # return: (*BDH, nao, nao)
        # TODO: check if what we need for Meta-GGA involving kinetics and for
        # exact-exchange
        pass

    @abstractmethod
    def get_exc(self, xc: BaseXC, dm: torch.Tensor) -> torch.Tensor:
        """
        Returns the exchange-correlation energy
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
