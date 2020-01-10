import torch
from ddft.transforms.base_transform import BaseTransform, SymmetricTransform

class IdentityTransform(SymmetricTransform):
    def __init__(self, shape, val=1):
        self._shape = shape
        assert self._shape[1] == self._shape[2], "The identity transform must be a square matrix"
        self.val = val

        # check the shape of val
        if type(val) == torch.Tensor:
            if self.val.ndim == 1:
                assert self.val.shape[0] == self._shape[0], "The batch size must match"
                self.val = self.val.unsqueeze(-1)
            elif self.val.ndim == 0:
                self.val = self.val.unsqueeze(-1).unsqueeze(-1)
            else:
                raise RuntimeError("The tensor val must be 1-dimension or 0-dimension")
        elif type(val) in [int, float]:
            self.val = torch.ones(self._shape[0], self._shape[1]) * 1.0 * self.val

    @property
    def shape(self):
        return self._shape

    def _forward(self, x):
        return x * self.val

    def diag(self):
        return self.val

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
