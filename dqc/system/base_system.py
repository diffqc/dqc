from abc import abstractmethod, abstractproperty
import torch
import xitorch as xt
from typing import List, Union
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.grid.base_grid import BaseGrid
from dqc.utils.datastruct import SpinParam

class BaseSystem(xt.EditableModule):
    """
    System is a class describing the environment before doing the quantum
    chemistry calculation.
    """

    @abstractmethod
    def get_hamiltonian(self) -> BaseHamilton:
        """
        Returns the Hamiltonian object for the system
        """
        pass

    @abstractmethod
    def get_orbweight(self, polarized: bool = False) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """
        Returns the atomic orbital weights. If polarized == False, then it
        returns the total orbital weights. Otherwise, it returns a tuple of
        orbital weights for spin-up and spin-down.
        """
        # returns: (*BS, norb)
        pass

    @abstractmethod
    def get_nuclei_energy(self) -> torch.Tensor:
        """
        Returns the nuclei-nuclei repulsion energy.
        """
        pass

    @abstractmethod
    def setup_grid(self) -> None:
        """
        Construct the integration grid for the system
        """
        pass

    @abstractmethod
    def get_grid(self) -> BaseGrid:
        """
        Returns the grid of the system
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass

    ####################### system properties #######################
    @abstractproperty
    def spin(self) -> int:
        """
        Returns the total spin of the system.
        """
        pass

    @abstractproperty
    def charge(self) -> int:
        """
        Returns the charge of the system.
        """
        pass

    @abstractproperty
    def numel(self) -> int:
        """
        Returns the total number of the electrons in the system.
        """
        pass
