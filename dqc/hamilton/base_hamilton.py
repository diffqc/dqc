from __future__ import annotations
import torch
import xitorch as xt
from abc import abstractmethod, abstractproperty
from typing import List, Optional, Union, overload
from dqc.grid.base_grid import BaseGrid
from dqc.xc.base_xc import BaseXC
from dqc.df.base_df import BaseDF
from dqc.utils.datastruct import SpinParam

class BaseHamilton(xt.EditableModule):
    """
    Hamilton is a class that provides the LinearOperator of the Hamiltonian
    components.
    """
    ############ properties ############
    @abstractproperty
    def nao(self) -> int:
        """
        Returns the number of atomic orbital basis
        """
        pass

    @abstractproperty
    def kpts(self) -> torch.Tensor:
        """
        Returns the list of k-points in the Hamiltonian, raise TypeError if
        the Hamiltonian does not have k-points.
        Shape: (nkpts, ndim)
        """
        pass

    @abstractproperty
    def df(self) -> Optional[BaseDF]:
        """
        Returns the density fitting object (if any) attached to this Hamiltonian
        object. If None, returns None
        """
        pass

    ############# setups #############
    @abstractmethod
    def build(self) -> BaseHamilton:
        """
        Construct the elements needed for the Hamiltonian.
        Heavy-lifting operations should be put here.
        """
        pass

    @abstractmethod
    def setup_grid(self, grid: BaseGrid, xc: Optional[BaseXC] = None) -> None:
        """
        Setup the basis (with its grad) in the spatial grid and prepare the
        gradient of atomic orbital according to the ones required by the xc.
        If xc is not given, then only setup the grid with ao (without any gradients
        of ao)
        """
        pass

    ############ fock matrix components ############
    @abstractmethod
    def get_nuclattr(self) -> xt.LinearOperator:
        """
        Returns the LinearOperator of the nuclear Coulomb attraction.
        """
        # return: (*BH, nao, nao)
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
        Known as the J-matrix.
        """
        # dm: (*BD, nao, nao)
        # return: (*BDH, nao, nao)
        pass

    @overload
    def get_exchange(self, dm: torch.Tensor) -> xt.LinearOperator:
        ...

    @overload
    def get_exchange(self, dm: SpinParam[torch.Tensor]) -> SpinParam[xt.LinearOperator]:
        ...

    @abstractmethod
    def get_exchange(self, dm):
        """
        Obtains the LinearOperator of the exchange operator.
        It is -0.5 * K where K is the K matrix obtained from 2-electron integral.
        """
        # dm: (*BD, nao, nao)
        # return: (*BDH, nao, nao)
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

    ############### interface to dm ###############
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

    ############### energy of the Hamiltonian ###############
    @abstractmethod
    def get_e_hcore(self, dm: torch.Tensor) -> torch.Tensor:
        """
        Get the energy from the one-electron Hamiltonian. The input is total
        density matrix.
        """
        pass

    @abstractmethod
    def get_e_elrep(self, dm: torch.Tensor) -> torch.Tensor:
        """
        Get the energy from the electron repulsion. The input is total density
        matrix.
        """
        pass

    @abstractmethod
    def get_e_exchange(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Get the energy from the exact exchange.
        """
        pass

    @abstractmethod
    def get_e_xc(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Returns the exchange-correlation energy using the xc object given in
        ``.setup_grid()``
        """
        # dm: (*BD, nao, nao)
        # return: (*BDH)
        pass

    ############### free parameters for variational method ###############
    @overload
    def ao_orb_params2dm(self, ao_orb_params: torch.Tensor, orb_weight: torch.Tensor,
                         with_penalty: None) -> torch.Tensor:
        ...

    @overload
    def ao_orb_params2dm(self, ao_orb_params: torch.Tensor, orb_weight: torch.Tensor,
                         with_penalty: float) -> Union[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def ao_orb_params2dm(self, ao_orb_params, orb_weight, with_penalty=None):
        """
        Convert the atomic orbital free parameters (parametrized in such a way so
        it is not bounded) to the density matrix.

        Arguments
        ---------
        ao_orb_params: torch.Tensor
            The tensors that parametrized atomic orbital in an unbounded space.
        orb_weight: torch.Tensor
            The orbital weights.
        with_penalty: float or None
            If a float, it returns a tuple of tensors where the first element is
            ``dm``, and the second element is the penalty multiplied by the penalty weights.
            The penalty is to compensate the overparameterization of ``ao_orb_params``,
            stabilizing the Hessian for gradient calculation.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            The density matrix from the orbital parameters and (if ``with_penalty``)
            the penalty of the overparameterization of ``ao_orb_params``.

        Notes
        -----
        * The penalty should be 0 if ``ao_orb_params`` is from ``dm2ao_orb_params``.
        * The density matrix should be recoverable when put through ``dm2ao_orb_params``
          and ``ao_orb_params2dm``.
        """
        pass

    @abstractmethod
    def dm2ao_orb_params(self, dm: torch.Tensor, norb: int) -> torch.Tensor:
        """
        Convert from the density matrix to the orbital parameters.
        The map is not one-to-one, but instead one-to-many where there might
        be more than one orbital parameters to describe the same density matrix.
        For restricted systems, only one of the ``dm`` (``dm.u`` or ``dm.d``) is
        sufficient.

        Arguments
        ---------
        dm: torch.Tensor
            The density matrix.
        norb: int
            The number of orbitals for the system.

        Returns
        -------
        torch.Tensor
            The atomic orbital parameters.
        """
        pass

    ############### xitorch's editable module ###############
    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        Return the paramnames
        """
        pass
