from typing import Tuple, Optional, Any
import torch
import xitorch as xt
import xitorch.linalg
from dqc.qccalc.base_qccalc import BaseQCCalc
from dqc.utils.misc import memoize_method

__all__ = ["hessian_pos", "vibration", "edipole", "equadrupole"]

# This file contains functions to calculate the perturbation properties of systems.

@memoize_method
def hessian_pos(qc: BaseQCCalc) -> torch.Tensor:
    """
    Returns the Hessian of energy with respect to atomic positions.

    Arguments
    ---------
    qc: BaseQCCalc
        Quantum Chemistry calculation that has run.

    Returns
    -------
    torch.Tensor
        Tensor with shape (natoms * ndim, natoms * ndim) represents the Hessian
        of the energy with respect to the atomic position
    """
    ene = qc.energy()
    system = qc.get_system()
    atompos = system.atompos

    # check if the atompos requires grad
    _check_differentiability(atompos, "atom positions", "hessian")

    # calculate the jacobian
    jac_e_pos = _jac(ene, atompos, create_graph=True)  # (natoms * ndim)
    hess_e_pos = _jac(jac_e_pos, atompos)  # (natoms * ndim, natoms * ndim)
    return hess_e_pos

@memoize_method
def vibration(qc: BaseQCCalc) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the vibration mode of the system based on the Hessian of energy
    with respect to atomic position.

    Arguments
    ---------
    qc: BaseQCCalc
        The qc calc object that has been executed.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of tensors where the first tensor is the frequency in atomic unit
        with shape `(natoms * ndim)` sorted from the largest to smallest,
        and the second tensor is the normal coordinate axes in atomic unit
        with shape `(natoms * ndim, natoms * ndim)` where each column corresponds
        to each axis sorted from the largest frequency to smallest frequency.
    """
    hess = hessian_pos(qc)  # (natoms * ndim, natoms * ndim)
    mass = qc.get_system().atommasses  # (natoms)
    ndim = hess.shape[0] // mass.shape[0]
    mass_mat = torch.diag_embed(mass.repeat_interleave(ndim))  # (natoms * ndim)^2

    # eival: (natoms * ndim)
    # eivec: (natoms * ndim, natoms * ndim)
    # eival and eivec are automatically sorted from smallest eival to the largest
    hess = hess + hess.transpose(-2, -1).conj()
    Alinop = xt.LinearOperator.m(hess, is_hermitian=True)
    Mlinop = xt.LinearOperator.m(mass_mat, is_hermitian=True)
    eival, eivec = xt.linalg.symeig(A=Alinop, M=Mlinop)
    freq = eival ** 0.5

    # reverse the sorting to make it sorted from largest to smallest
    freq = torch.flip(freq, dims=(-1,))
    eivec = torch.flip(eivec, dims=(-1,))
    return freq, eivec

@memoize_method
def edipole(qc: BaseQCCalc) -> torch.Tensor:
    """
    Returns the electric dipole moment of the system, i.e. derivative of energy
    w.r.t. electric field.

    Arguments
    ---------
    qc: BaseQCCalc
        The qc calc object that has been executed.

    Returns
    -------
    torch.Tensor
        Tensor representing the dipole moment in atomic unit with shape (ndim,)
    """
    ene = qc.energy()
    system = qc.get_system()
    efield = system.efield

    # check if the electric field requires grad
    _check_differentiability(efield, "electric field", "dipole")
    assert isinstance(efield, torch.Tensor)

    dipole = _jac(ene, efield)
    return dipole

@memoize_method
def equadrupole(qc: BaseQCCalc) -> torch.Tensor:
    """
    Returns the electric quadrupole moment of the system, i.e. derivative of energy
    w.r.t. electric field.

    Arguments
    ---------
    qc: BaseQCCalc
        The qc calc object that has been executed.

    Returns
    -------
    torch.Tensor
        Tensor representing the quadrupole moment in atomic unit in (ndim, ndim)
    """
    ene = qc.energy()
    system = qc.get_system()
    efield = system.efield

    # check if the electric field requires grad
    _check_differentiability(efield, "electric field", "quadpole")
    assert isinstance(efield, torch.Tensor)

    dipole = _jac(ene, efield, create_graph=True)  # (ndim,)
    quadrupole = _jac(dipole, efield)  # (ndim, ndim)
    return quadrupole

########### helper functions ###########

def _jac(a: torch.Tensor, b: torch.Tensor, create_graph: Optional[bool] = None,
         retain_graph: bool = True):
    # calculate the jacobian of a w.r.t. b
    # a: (*BA)
    # b: (*BB)
    # returns (*BA, prod(*BB))
    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    assert create_graph is not None

    aflat = a.reshape(-1)
    anumel = a.numel()
    bnumel = b.numel()
    res = torch.empty((anumel, bnumel), dtype=a.dtype, device=a.device)
    for i in range(anumel):
        res[i] = torch.autograd.grad(aflat[i], b, create_graph=create_graph,
                                     retain_graph=retain_graph)[0].reshape(-1)
    res = res.reshape((*a.shape, bnumel))
    return res

def _check_differentiability(a: Any, aname: str, propname: str):
    # check if a is a differentiable tensor and raise an error if it is not
    if not (isinstance(a, torch.Tensor) and a.requires_grad):
        msg = "Differentiable tensor %s is required to calculate the %s" % (aname, propname)
        raise RuntimeError(msg)
