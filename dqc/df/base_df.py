from __future__ import annotations
from abc import abstractmethod, abstractproperty
from typing import List
import torch
import xitorch as xt

class BaseDF(xt.EditableModule):
    """
    BaseDF represents the density fitting object used in calculating the
    electron repulsion (and xc energy?) in Hamiltonian.
    """
    @abstractmethod
    def build(self) -> BaseDF:
        """
        Construct the matrices required to perform the calculation and return
        self.
        """
        pass

    @abstractmethod
    def get_elrep(self, dm: torch.Tensor) -> xt.LinearOperator:
        """
        Construct the electron repulsion linear operator from the given density
        matrix using the density fitting method.
        """
        pass

    ################ properties ################
    @abstractproperty
    def j2c(self) -> torch.Tensor:
        """
        Returns the 2-centre 2-electron integrals of the auxiliary basis.
        """
        pass

    @abstractproperty
    def j3c(self) -> torch.Tensor:
        """
        Return the 3-centre 2-electron integrals of the auxiliary basis and the
        basis.
        """
        pass

    ################ properties ################
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass
