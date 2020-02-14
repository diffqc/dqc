from abc import abstractmethod, abstractproperty
import torch

class BaseGrid(object):

    @abstractmethod
    def get_integrand_box(self, p):
        """
        Get the integrand of p when performing integral(p * dVolume) over the
        last dimension.
        The output of this function will be summed when the integral over the
        volume is performed.

        Arguments
        ---------
        * p: torch.tensor (..., ..., nr)
            The tensor to be integrated over the spatial grid.

        Returns
        -------
        * pintegrand: torch.tensor (..., ..., nr)
            The integrand of the volume integral so that pintegrand.sum() should
            be performed when the volume integration is performed.
        """
        pass

    @abstractmethod
    def solve_poisson(self, f):
        """
        Solve Poisson's equation del^2 v = f, where f is a torch.tensor with
        shape (nbatch, nr) and v is also similar.

        Arguments
        ---------
        * f: torch.tensor (nbatch, nr)
            Input tensor

        Results
        -------
        * v: torch.tensor (nbatch, nr)
            The tensor that fulfills del^2 v = f. Please note that v can be
            non-unique due to ill-conditioned operator del^2.
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
        integrand = self.get_integrand_box(p.transpose(dim,-1))
        res = torch.sum(integrand, dim=-1)
        return res.transpose(dim,-1)

    def mmintegralbox(self, p1, p2):
        """
        Perform the equivalent of matrix multiplication but replacing the sum
        with integral sum.

        Arguments
        ---------
        * p1: torch.tensor (..., n1, nr)
        * p2: torch.tensor (..., nr, n2)
            The integrands of the matrix multiplication integration.

        Returns
        -------
        * mm: torch.tensor (..., n1, n2)
            The result of matrix multiplication integration.
        """
        pleft = self.get_integrand_box(p1)
        return torch.matmul(pleft, p2)
