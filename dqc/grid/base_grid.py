from abc import abstractmethod
from typing import List
import torch
import xitorch as xt

class BaseGrid(xt.EditableModule):
    """
    Grid is a class that regulates the integration points over the spatial
    dimensions.
    """
    @abstractmethod
    def get_dvolume(self) -> torch.Tensor:
        """
        Obtain the torch.tensor containing the dV elements for the integration.

        Returns
        -------
        torch.tensor (*BG, ngrid)
            The dV elements for the integration
        """
        pass

    @abstractmethod
    def get_rgrid(self) -> torch.Tensor:
        """
        Returns the grid points position in the Cartesian coordinate.

        Returns
        -------
        torch.tensor (*BG, ngrid, ndim)
            The grid points position.
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass
