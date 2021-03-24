from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional, Union, List, TypeVar, Generic

__all__ = ["CGTOBasis", "AtomCGTOBasis", "ValGrad"]

T = TypeVar('T')

# type of the atom Z
ZType = Union[int, float, torch.Tensor]

def is_z_float(a: ZType):
    # returns if the given z-type is a floating point
    if isinstance(a, torch.Tensor):
        return a.is_floating_point()
    else:
        return isinstance(a, float)

@dataclass
class CGTOBasis:
    angmom: int
    alphas: torch.Tensor  # (nbasis,)
    coeffs: torch.Tensor  # (nbasis,)
    normalized: bool = False

@dataclass
class AtomCGTOBasis:
    atomz: ZType
    bases: List[CGTOBasis]
    pos: torch.Tensor  # (ndim,)

# input basis type
BasisInpType = Union[str, List[CGTOBasis], List[str], List[List[CGTOBasis]]]

@dataclass
class SpinParam(Generic[T]):
    u: T
    d: T

@dataclass
class ValGrad:
    value: torch.Tensor  # torch.Tensor of the value in the grid
    grad: Optional[torch.Tensor] = None  # torch.Tensor representing (gradx, grady, gradz) with shape
    # ``(..., 3)``
    lapl: Optional[torch.Tensor] = None  # torch.Tensor of the laplace of the value

    def __add__(self, b: ValGrad) -> ValGrad:
        return ValGrad(
            value=self.value + b.value,
            grad=self.grad + b.grad if self.grad is not None else None,
            lapl=self.lapl + b.lapl if self.lapl is not None else None,
        )

    def __mul__(self, f: Union[float, int, torch.Tensor]) -> ValGrad:
        if isinstance(f, torch.Tensor):
            assert f.numel() == 1, "ValGrad multiplication with tensor can only be done with 1-element tensor"

        return ValGrad(
            value=self.value * f,
            grad=self.grad * f if self.grad is not None else None,
            lapl=self.lapl * f if self.lapl is not None else None,
        )
