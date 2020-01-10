import torch
from ddft.transforms.base_transform import BaseTransform, SymmetricTransform

class IdentityTransform(SymmetricTransform):
    def __init__(self, shape, val=1):
        self._shape = shape
        self.val = val

    @property
    def shape(self):
        return self._shape

    def _forward(self, x):
        return x * self.val

class MatrixTransform(BaseTransform):
    def __init__(self, A):
        self.A = A

    @property
    def shape(self):
        return self.A.shape

    def _forward(self, x):
        return torch.bmm(self.A, x.unsqueeze(-1)).squeeze(-1)

    def _transpose(self, y):
        return torch.bmm(self.A.T, y.unsqueeze(-1)).squeeze(-1)

    def diag(self):
        return torch.diagonal(self.A, dim1=-2, dim2=-1)
