from abc import abstractmethod

class BaseTransform(object):
    """
    Transform objects are linear transformation that can be written as a matrix,
    A. However, sometimes the matrix is so big, so it is just accessed by
    the forward and transpose transformation.
    """
    @abstractmethod
    def __call__(self, x):
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
        return AddTransform(self, other)

    def __sub__(self, other):
        return SubTransform(self, other)

    def __mul__(self, other):
        return ConcatTransform(self, other)

    def __neg__(self):
        return NegTransform(self)

class SymmetricTransform(BaseTransform):
    @abstractmethod
    def _transpose(self, y):
        return self.__call__(y)

    @property
    def T(self):
        return self

class TransposeTransform(BaseTransform):
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return self.a._transpose(x)

    def _transpose(self, y):
        return self.a(y)

    @property
    def T(self):
        return self.a

class InverseTransform(BaseTransform):
    def __init__(self, a, **options):
        self.a = a
        self.options = options

    def __call__(self, x):
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

####################### arithmetics operation #######################

class AddTransform(BaseTransform):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a(x) + self.b(x)

    def _transpose(self, y):
        return self.a.T(y) + self.b.T(y)

class SubTransform(BaseTransform):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a(x) - self.b(x)

    def _transpose(self, y):
        return self.a.T(y) - self.b.T(y)

class ConcatTransform(BaseTransform):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a(self.b(x))

    def _transpose(self, y):
        return self.b.T(self.a.T(y))

class NegTransform(BaseTransform):
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return -self.a(x)

    def _transpose(self, y):
        return -self.a.T(y)
