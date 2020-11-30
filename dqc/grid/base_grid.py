from abc import abstractmethod, abstractproperty
from typing import List
import torch
import xitorch as xt

class BaseGrid(xt.EditableModule):
    """
    Grid is a class that regulates the integration points over the spatial
    dimensions.
    """
    @abstractproperty
    def dtype(self) -> torch.dtype:
        pass

    @abstractproperty
    def device(self) -> torch.device:
        pass

    @abstractproperty
    def coord_type(self) -> str:
        """
        Returns the type of the coordinate returned in get_rgrid
        """
        pass

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
        Returns the grid points position in the specified coordinate in
        self.coord_type.

        Returns
        -------
        torch.tensor (*BG, ngrid, ndim)
            The grid points position.
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass
