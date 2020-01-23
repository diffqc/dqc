from abc import abstractmethod, abstractproperty
import torch

class BaseLinearModule(torch.nn.Module):
    """
    Base module of linear modules.
    Linear module is a module that can be expressed as a matrix of size
        (nrows, ncols).
    """
    @abstractmethod
    def forward(self, x, *params):
        """
        Calculate the operation of the transformation with `x` where
        the detail of the transformation is set by *params.
        `x` and each of `params` should be differentiable.

        Arguments
        ---------
        * x: torch.tensor (nbatch, ncols) or (nbatch, ncols, nelmt)
            The input vector of the linear transformation.
        * *params: list of torch.tensor (nbatch, ...)
            The differentiable parameters that sets the linear transformation.

        Returns
        -------
        * y: torch.tensor (nbatch, nrows) or (nbatch, nrows, nelmt)
            The output of the linear transformation, i.e. y = Ax
        """
        pass

    @abstractproperty
    def shape(self):
        """
        Returns (nrows, ncols)
        """
        pass

    def diag(self, *params):
        """
        Returns the diagonal elements of the transformation.
        The transformation must be a square matrix for this to be valid.
        The return type should have shape of (nbatch,nrows)
        """
        msg = "The .diag() method is unimplemented for class %s" % \
              (self.__class__.__name__)
        raise RuntimeError(msg)

    @property
    def iscomplex(self):
        return False

    @property
    def issymmetric(self):
        return True
