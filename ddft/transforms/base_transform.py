from abc import abstractmethod

class BaseTransform(object):
    """
    Transform objects are linear transformation that can be written as a matrix,
    A. However, sometimes the matrix is so big, so it is just accessed by
    the forward and transpose transformation.
    """
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def _transpose(self, y):
        pass

    @property
    def T(self):
        if not hasattr(self, "_transpose_transform_"):
            self._transpose_transform_ = TransposeTransform(self)
        return self._transpose_transform_

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
