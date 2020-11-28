from abc import abstractmethod, abstractproperty
from typing import List
import xitorch as xt

class BaseGrid(xt.EditableModule):
    """
    Grid is a class that regulates the integration points over the spatial
    dimensions.
    """
    @abstractmethod
    def get_dvolume(self):
        """
        Obtain the torch.tensor containing the dV elements for the integration.

        Returns
        -------
        torch.tensor (*BG, ngrid)
            The dV elements for the integration
        """
        pass

    @abstractproperty
    def rgrid(self):
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
