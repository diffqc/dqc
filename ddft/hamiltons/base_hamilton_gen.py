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
