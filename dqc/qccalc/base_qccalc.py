from abc import abstractmethod
from typing import List
import torch
import xitorch as xt

class BaseQCCalc(xt.EditableModule):
    @abstractmethod
    def run(self, **kwargs):
        """
        Run the calculation.
        Note that this method can be invoked several times for one object to
        try for various self-consistent options to reach convergence.
        """
        pass

    @abstractmethod
    def energy(self) -> torch.Tensor:
        """
        Obtain the energy of the system.
        """
        pass

    @abstractmethod
    def aodm(self) -> torch.Tensor:
        """
        Returns the density matrix in atomic orbital.
        """
        # return: (nao, nao)
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass
