from abc import abstractmethod
import torch
import xitorch as xt
from typing import List
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.grid.base_grid import BaseGrid

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
    def get_orbweight(self) -> torch.Tensor:
        """
        Returns the atomic orbital weights.
        """
        # returns: (*BS, norb)
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
