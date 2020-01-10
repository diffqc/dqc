from abc import abstractmethod, abstractproperty
import torch
from ddft.utils.rootfinder import lbfgs

class BaseTransform(object):
    """
    Transform objects are linear transformation that can be written as a matrix,
    A. However, sometimes the matrix is so big, so it is just accessed by
    the forward and transpose transformation.
    """
    @abstractmethod
    def _forward(self, x):
        """
        Applying the transformation to `x` with shape (nbatch, ninputs)
        The output is (nbatch, noutputs).
        """
        pass

    @abstractmethod
    def _transpose(self, y):
        """
        Applying the transpose transformation to `y` with shape
        (nbatch, noutputs).
        The output is (nbatch, ninputs).
        """
        pass

    def diag(self):
        """
        Returns tensor which represents the diagonal of the transformation.
        The shape of the tensor should be (nbatch, ninputs/noutputs).
        If the ninputs != noutputs, then it should be invalid.
        """
        raise RuntimeError(
            "The .diag() method for class %s is not implemented." \
            % (self.__class__.__name__))

    @abstractproperty
    def shape(self):
        """
        The shape of the matrix. It should be (nbatch, nrows, ncols).
        """
        pass

    def __call__(self, x):
        """
        Applying the transformation to `x`.
        If x has the shape (nbatch, ninputs), the output is (nbatch, noutputs).
        If x is (nbatch, ninputs, nx), the output is (nbatch, noutputs, nx)
        """
        if len(x.shape) == 2:
            return self._forward(x)
        else:
            nx = x.shape[-1]
            res = [self._forward(x[:,:,i]).unsqueeze(-1) for i in range(nx)]
            return torch.cat(res, dim=-1)

    @property
    def T(self):
        """
        Return the transpose transformation object.
        """
        if not hasattr(self, "_transpose_transform_"):
            self._transpose_transform_ = TransposeTransform(self)
        return self._transpose_transform_

    def inv(self, **options):
        """
        Return the inverse transformation object.
        """
        if not hasattr(self, "_inverse_"):
            self._inverse_ = InverseTransform(self, **options)
        return self._inverse_

    def __add__(self, other):
        other = self._normalize_type(other)
        return AddTransform(self, other)

    def __sub__(self, other):
        other = self._normalize_type(other)
        return SubTransform(self, other)

    def __mul__(self, other):
        other = self._normalize_type(other)
        return ConcatTransform(self, other)

    def __neg__(self):
        return NegTransform(self)

    def _normalize_type(self, a):
        from ddft.transforms.tensor_transform import IdentityTransform

        if type(a) in [int, float]:
            return IdentityTransform(self.shape, a)
        elif type(a) == torch.Tensor:
            if a.numel() == 1:
                return IdentityTransform(self.shape, a)
            elif len(a.shape) == 1 and a.shape[0] == self.shape[0]:
                return IdentityTransform(self.shape, a)
            else:
                raise TypeError("Don't know how to handle tensor input with shape: %s" % str(a.shape))
        elif isinstance(a, BaseTransform):
            return a
        else:
            raise TypeError("Unknown operand %s with Transform object" % (a.__class__.__name__))

class SymmetricTransform(BaseTransform):
    def _transpose(self, y):
        return self._forward(y)

    @property
    def T(self):
        return self

class TransposeTransform(BaseTransform):
    def __init__(self, a):
        self.a = a

    def _forward(self, x):
        return self.a._transpose(x)

    def _transpose(self, y):
        return self.a(y)

    def diag(self):
        return self.a.diag()

    @property
    def T(self):
        return self.a

    @property
    def shape(self):
        shape = self.a.shape
        return (shape[0], shape[2], shape[1])

class InverseTransform(BaseTransform):
    def __init__(self, a, **options):
        self.a = a
        self.options = options

    def _forward(self, x):
        def loss(y):
            return self.a(y) - x
        jinv0 = 1.0
        y0 = torch.zeros_like(x).to(x.device)
        y = lbfgs(loss, y0, jinv0, **self.options)
        return y

    def _transpose(self, y):
        def loss(x):
            return self.a._transpose(x) - y
        jinv0 = 1.0
        x0 = torch.zeros_like(y).to(y.device)
        x = lbfgs(loss, x0, jinv0, **self.options)
        return x

    def diag(self):
        return 1. / self.a.diag()

    @property
    def shape(self):
        shape = self.a.shape
        return (shape[0], shape[2], shape[1])

######################## composite transforms ################################

class AddTransform(BaseTransform):
    def __init__(self, a, b):
        self.a = a
        self.b = b

        assert self.a.shape == self.b.shape, "Mismatch size of add transforms"

    def _forward(self, x):
        return self.a(x) + self.b(x)

    def _transpose(self, y):
        return self.a.T(y) + self.b.T(y)

    def diag(self):
        return self.a.diag() + self.b.diag()

    @property
    def shape(self):
        shape = self.a.shape
        return (shape[0], shape[1], shape[2])

class SubTransform(BaseTransform):
    def __init__(self, a, b):
        self.a = a
        self.b = b

        assert self.a.shape == self.b.shape, "Mismatch size of subtraction transforms"

    def _forward(self, x):
        return self.a(x) - self.b(x)

    def _transpose(self, y):
        return self.a.T(y) - self.b.T(y)

    def diag(self):
        return self.a.diag() - self.b.diag()

    @property
    def shape(self):
        shape = self.a.shape
        return (shape[0], shape[1], shape[2])

class ConcatTransform(BaseTransform):
    def __init__(self, a, b):
        self.a = a
        self.b = b

        # check the shape
        shapea = self.a.shape
        shapeb = self.b.shape
        assert shapea[-1] == shapeb[-2] and shapea[0] == shapeb[0], "Mismatch size of concatenated transforms"

    def _forward(self, x):
        return self.a(self.b(x))

    def _transpose(self, y):
        return self.b.T(self.a.T(y))

    @property
    def shape(self):
        shapea = self.a.shape
        shapeb = self.b.shape
        return (shapea[0], shapea[1], shapeb[2])

class NegTransform(BaseTransform):
    def __init__(self, a):
        self.a = a

    def _forward(self, x):
        return -self.a(x)

    def _transpose(self, y):
        return -self.a.T(y)

    def diag(self):
        return -self.a.diag()

    @property
    def shape(self):
        shape = self.a.shape
        return (shape[0], shape[1], shape[2])
