from typing import overload, Tuple
import torch

class BaseOrbParams(object):
    """
    Class that provides free-parameterization of orthogonal orbitals.
    """
    @overload
    @staticmethod
    def params2orb(params: torch.Tensor, with_penalty: None) -> torch.Tensor:
        ...

    @overload
    @staticmethod
    def params2orb(params: torch.Tensor, with_penalty: float) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @staticmethod
    def params2orb(params, with_penalty):
        """
        Convert the parameters to the orthogonal orbitals.
        """
        pass

    @staticmethod
    def orb2params(orb: torch.Tensor) -> torch.Tensor:
        """
        Get the free parameters from the orthogonal orbitals.
        """
        pass

class QROrbParams(BaseOrbParams):
    @overload
    @staticmethod
    def params2orb(params: torch.Tensor, with_penalty: None) -> torch.Tensor:
        ...

    @overload
    @staticmethod
    def params2orb(params: torch.Tensor, with_penalty: float) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @staticmethod
    def params2orb(params, with_penalty):
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
    def orb2params(orb: torch.Tensor) -> torch.Tensor:
        return orb
