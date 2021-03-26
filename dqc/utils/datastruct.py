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

    def wfnormalize_(self) -> CGTOBasis:
        # wavefunction normalization
        # the normalization is obtained from CINTgto_norm from
        # libcint/src/misc.c, or
        # https://github.com/sunqm/libcint/blob/b8594f1d27c3dad9034984a2a5befb9d607d4932/src/misc.c#L80

        # Please note that the square of normalized wavefunctions do not integrate
        # to 1, but e.g. for s: 4*pi, p: (4*pi/3)

        # if the basis has been normalized before, then do nothing
        if self.normalized:
            return self

        # precomputed factor:
        # 2 ** (2 * angmom + 3) * factorial(angmom + 1) * / (factorial(angmom * 2 + 2) * np.sqrt(np.pi)))
        factor = [
            2.256758334191025,  # 0
            1.5045055561273502,  # 1
            0.6018022224509401,  # 2
            0.17194349212884005,  # 3
            0.03820966491752001,  # 4
            0.006947211803185456,  # 5
            0.0010688018158746854,  # 6
        ][self.angmom]
        self.coeffs = self.coeffs * torch.sqrt(factor * (2 * self.alphas) ** (self.angmom + 1.5))
        self.normalized = True
        return self

@dataclass
class AtomCGTOBasis:
    atomz: ZType
    bases: List[CGTOBasis]
    pos: torch.Tensor  # (ndim,)

# input basis type
BasisInpType = Union[str, List[CGTOBasis], List[str], List[List[CGTOBasis]]]

@dataclass
class DensityFitInfo:
    method: str
    auxbases: List[AtomCGTOBasis]

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
