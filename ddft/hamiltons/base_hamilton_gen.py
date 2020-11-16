from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
import xitorch as xt
from typing import List
from collections import namedtuple
from ddft.utils.datastruct import DensityInfo

class BaseHamiltonGenerator(xt.EditableModule):
    """
    This is a base class for Hamiltonian generator.
    """
    def __init__(self, grid, shape, dtype=None, device=None):
        self._grid = grid
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def set_basis(self, gradlevel=0):
        """
        Set the basis for external potential conversion to linear operator.

        Arguments
        ---------
        gradlevel: int
            Gradient level (0, 1, or 2) for the basis to be set up.
            If 0, only the basis as a function of grid points in the space are
            calculated (can calculate `get_vext`)
            If 1, basis and the grad_(x,y,z) are calculated. It allows for the
            calculation of ``get_vext`` and ``get_grad_vext``.
            If 2, basis, grad_(x,y,z), and grad^2 are calculated, enabling the
            computation of ``get_vext``, ``get_grad_vext``, and ``get_lapl_vext``.
        """
        pass

    @property
    def grid(self):
        return self._grid

    @abstractmethod
    def get_kincoul(self) -> xt.LinearOperator:
        """
        Return the LinearOperator of the kinetic energy and electron-ion
        Coulomb potential.

        Returns
        -------
        xt.LinearOperator
            LinearOperator with shape ``(..., nbasis, nbasis)`` of the kinetics
            and the electron-ion Coulomb potential.
        """
        pass

    @abstractmethod
    def get_overlap(self, *sparams) -> xt.LinearOperator:
        """
        Return the overlap matrix as LinearOperator given the parameters.

        Arguments
        ---------
        *sparams
            Additional parameters to specify the overlap matrix.

        Returns
        -------
        xt.LinearOperator
            The overlap matrix as LinearOperator with shape
            ``(..., nbasis, nbasis)`` (the batch dimension depends on
            ``sparams``)
        """
        pass

    ##################### grid-dependant linear operators #####################
    @abstractmethod
    def get_vext(self, vext:torch.Tensor) -> xt.LinearOperator:
        """
        Return the Hamiltonian LinearOperator of the external potential.

        Arguments
        ---------
        vext: torch.Tensor
            The external potential with the shape of ``(...,nr)`` where ``nr``
            is the number of points in the grid.

        Returns
        -------
        xt.LinearOperator
            External potential operator as Hermitian LinearOperator with shape
            ``(..., nbasis, nbasis)``
        """
        pass

    @abstractmethod
    def get_grad_vext(self, grad_vext:torch.Tensor) -> xt.LinearOperator:
        """
        Return the Hamiltonian LinearOperator of the external potential gradient.

        Arguments
        ---------
        grad_vext: torch.Tensor
            The external potential gradient with the shape of ``(..., nr, 3)``
            where ``nr`` is the number of points in the grid.
            The last dimension corresponds to grad_x, grad_y, and grad_y,
            respectively.

        Returns
        -------
        xt.LinearOperator
            External potential gradient operator as Hermitian LinearOperator
            with shape ``(..., nbasis, nbasis)``
        """
        pass

    @abstractmethod
    def dm2dens(self, dm:torch.Tensor, calc_gradn=False) -> DensityInfo:
        """
        Convert the density matrix to the density profile on the grid points.

        Arguments
        ---------
        dm: torch.Tensor
            The density matrix tensor with shape ``(..., nbasis, nbasis)``
        calc_gradn: bool
            If True, it will return "gradn" field in the DensityInfo. Otherwise,
            the "gradn" field will be None.

        Returns
        -------
        DensityInfo
            The namedtuple containing ("density", "gradn").
            "density" will be a tensor with shape ``(..., nr)``,
            "gradn" will be a None or a tuple of 3 tensors with shape
            ``(..., nr)`` corresponding to gradx, grady, and gradz.
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname:str, prefix:str="") -> List[str]:
        pass
