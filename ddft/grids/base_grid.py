from abc import abstractmethod, abstractproperty

class BaseGrid(object):
    @abstractmethod
    def integralbox(self, p, dim=-1):
        """
        Performing the integral over the spatial grid of the signal `p` where
        the signal in spatial grid is located at the dimension `dim`.

        Arguments
        ---------
        * p: torch.tensor (..., nr, ...)
            The tensor to be integrated over the spatial grid.
        * dim: int
            The dimension where it should be integrated.
        """
        pass

    @abstractproperty
    def rgrid(self):
        """
        Returns (nr, ndim) torch.tensor which represents the spatial position
        of the grid.
        """
        pass

    @abstractproperty
    def boxshape(self):
        """
        Returns the shape of the signal in the real spatial dimension.
        prod(boxshape) == rgrid.shape[0]
        """
        pass
