from typing import overload, Tuple
import torch

__all__ = ["BaseOrbParams", "QROrbParams", "MatExpOrbParams"]

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

class MatExpOrbParams(BaseOrbParams):
    """
    Orthogonal orbital parameterization using matrix exponential.
    The orthogonal orbital is represented by:

        P = matrix_exp(Q) @ C

    where C is an orthogonal coefficient tensor, and Q is the parameters defining
    the rotation of the orthogonal tensor.
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
        # params: (*, nparams)
        # coeffs: (*, nao, norb)
        nao = coeffs.shape[-2]
        norb = coeffs.shape[-1]
        nparams = params.shape[-1]
        bshape = params.shape[:-1]

        # construct the rotation parameters
        triu_idxs = torch.triu_indices(nao, nao, offset=1)[..., :nparams]
        rotmat = torch.zeros((*bshape, nao, nao), dtype=params.dtype, device=params.device)
        rotmat[..., triu_idxs[0], triu_idxs[1]] = params
        rotmat = rotmat - rotmat.transpose(-2, -1).conj()

        # calculate the orthogonal orbital
        ortho_orb = torch.matrix_exp(rotmat) @ coeffs

        if with_penalty:
            penalty = torch.zeros((1,), dtype=params.dtype, device=params.device)
            return ortho_orb, penalty
        else:
            return ortho_orb

    @staticmethod
    def orb2params(orb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # orb: (*, nao, norb)
        nao = orb.shape[-2]
        norb = orb.shape[-1]
        nparams = norb * (nao - norb) + norb * (norb - 1) // 2

        # the orbital becomes the coefficients while params is all zeros (no rotation)
        coeffs = orb
        params = torch.zeros((*orb.shape[:-2], nparams), dtype=orb.dtype, device=orb.device)
        return params, coeffs
