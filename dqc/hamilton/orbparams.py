from typing import overload, Tuple
import torch

class BaseOrbParams(object):
    """
    Class that provides free-parameterization of orthogonal orbitals.
    """
    @overload
    @staticmethod
    def params2orb(params: torch.Tensor, coeffs: torch.Tensor, with_penalty: None) -> torch.Tensor:
        ...

    @overload
    @staticmethod
    def params2orb(params: torch.Tensor, coeffs: torch.Tensor, with_penalty: float) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @staticmethod
    def params2orb(params, coeffs, with_penalty):
        """
        Convert the parameters & coefficients to the orthogonal orbitals.
        ``params`` is the tensor to be optimized in variational method, while
        ``coeffs`` is a tensor that is needed to get the orbital, but it is not
        optimized in the variational method.
        """
        pass

    @staticmethod
    def orb2params(orb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the free parameters from the orthogonal orbitals. Returns ``params``
        and ``coeffs`` described in ``params2orb``.
        """
        pass

class QROrbParams(BaseOrbParams):
    @overload
    @staticmethod
    def params2orb(params: torch.Tensor, coeffs: torch.Tensor, with_penalty: None) -> torch.Tensor:
        ...

    @overload
    @staticmethod
    def params2orb(params: torch.Tensor, coeffs: torch.Tensor, with_penalty: float) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @staticmethod
    def params2orb(params, coeffs, with_penalty):
        orb, _ = torch.linalg.qr(params)
        if with_penalty is None:
            return orb
        else:
            # QR decomposition's solution is not unique in a way that every column
            # can be multiplied by -1 and it still a solution
            # So, to remove the non-uniqueness, we will make the sign of the sum
            # positive.
            s1 = torch.sign(orb.sum(dim=-2, keepdim=True))  # (*BD, 1, norb)
            s2 = torch.sign(params.sum(dim=-2, keepdim=True))
            penalty = torch.mean((orb * s1 - params * s2) ** 2) * with_penalty
            return orb, penalty

    @staticmethod
    def orb2params(orb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coeffs = torch.tensor([0], dtype=orb.dtype, device=orb.device)
        return orb, coeffs
