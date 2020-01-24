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

    @abstractmethod
    def transpose(self, y, *params):
        pass

    @abstractproperty
    def shape(self):
        """
        Returns (nrows, ncols)
        """
        pass

    @property
    def T(self):
        if not hasattr(self, "_T"):
            self._T = TransposeModule(self)
        return self._T

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
    def issymmetric(self):
        return True

class TransposeModule(BaseLinearModule):
    def __init__(self, model):
        super(TransposeModule, self).__init__()
        self.model = model
        self._shape = [model.shape[1], model.shape[0]]

    def forward(self, x, *args):
        return self.model.transpose(x, *args)

    def transpose(self, x, *args):
        return self.model(x, *args)

    @property
    def T(self):
        return self.model

    @property
    def shape(self):
        return self._shape

    def diag(self, *params):
        return self.model.diag(*params)

    @property
    def issymmetric(self):
        return self.model.issymmetric
