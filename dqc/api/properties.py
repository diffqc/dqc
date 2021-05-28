from typing import Tuple, Optional, Any
import torch
import numpy as np
import xitorch as xt
import xitorch.linalg
import xitorch.grad
import xitorch.optimize
from dqc.qccalc.base_qccalc import BaseQCCalc
from dqc.utils.misc import memoize_method
from dqc.utils.datastruct import SpinParam
from dqc.utils.units import length_to, freq_to, edipole_to, equadrupole_to, ir_ints_to, \
                            raman_ints_to

__all__ = ["hessian_pos", "vibration", "edipole", "equadrupole", "is_orb_min",
           "lowest_eival_orb_hessian", "ir_spectrum", "raman_spectrum"]

# This file contains functions to calculate the perturbation properties of systems.

def hessian_pos(qc: BaseQCCalc, unit: Optional[str] = None) -> torch.Tensor:
    """
    Returns the Hessian of energy with respect to atomic positions.

    Arguments
    ---------
    qc: BaseQCCalc
        Quantum Chemistry calculation that has run.

    unit: str or None
        The returned unit. If ``None``, returns in atomic unit.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(natoms * ndim, natoms * ndim)`` represents the Hessian
        of the energy with respect to the atomic position
    """
    hess = _hessian_pos(qc)
    hess = length_to(hess, unit)
    return hess

def vibration(qc: BaseQCCalc, freq_unit: Optional[str] = "cm^-1",
              length_unit: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the vibration mode of the system based on the Hessian of energy
    with respect to atomic position.

    Arguments
    ---------
    qc: BaseQCCalc
        The qc calc object that has been executed.
    freq_unit: str or None
        The returned unit for the frequency. If ``None``, returns in atomic unit.
    length_unit: str or None
        The returned unit for the normal mode coordinate. If ``None``, returns in
        atomic unit

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of tensors where the first tensor is the frequency in atomic unit
        with shape ``(natoms * ndim)`` sorted from the largest to smallest,
        and the second tensor is the normal coordinate axes in atomic unit
        with shape ``(natoms * ndim, natoms * ndim)`` where each column corresponds
        to each axis sorted from the largest frequency to smallest frequency.
    """
    freq, mode = _vibration(qc)
    freq = freq_to(freq, freq_unit)
    mode = length_to(mode, length_unit)
    return freq, mode

def ir_spectrum(qc: BaseQCCalc, freq_unit: Optional[str] = "cm^-1",
                ints_unit: Optional[str] = "(debye/angst)^2/amu") \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the frequency and intensity of the IR vibrational spectra.
    Unlike ``vibration``, this method only returns parts where the frequency is
    positive.

    Arguments
    ---------
    qc: BaseQCCalc
        The qc calc object that has been executed.
    freq_unit: str or None
        The returned unit for the frequency. If ``None``, returns in atomic unit.
    ints_unit: str or None
        The returned unit for the intensity. If ``None``, returns in atomic unit.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of tensors where the first tensor is the frequency in the given unit
        with shape ``(nfreqs,)`` sorted from the largest to smallest, and the second
        tensor is the IR intensity with the same order as the frequency.
    """
    freq, ir_ints = _ir_spectrum(qc)
    freq = freq_to(freq, freq_unit)
    ir_ints = ir_ints_to(ir_ints, ints_unit)
    return freq, ir_ints

def raman_spectrum(qc: BaseQCCalc, freq_unit: Optional[str] = "cm^-1",
                   ints_unit: Optional[str] = "angst^4/amu") \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the frequency, static Raman intensity spectra, and depolarization ratio.
    Like IR spectrum, this method only returns parts where the frequency is positive.

    Arguments
    ---------
    qc: BaseQCCalc
        The qc calc object that has been executed.
    freq_unit: str or None
        The returned unit for the frequency. If ``None``, returns in atomic unit.
    ints_unit: str or None
        The returned unit for the Raman intensity. If ``None``, returns in atomic unit.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of tensors where the first tensor is the frequency in the given unit
        with shape ``(nfreqs,)`` sorted from the largest to smallest, and the second
        tensor is the IR intensity with the same order as the frequency.
    """
    freq, raman_ints = _raman_spectrum(qc)
    freq = freq_to(freq, freq_unit)
    raman_ints = raman_ints_to(raman_ints, ints_unit)
    return freq, raman_ints

def edipole(qc: BaseQCCalc, unit: Optional[str] = "Debye") -> torch.Tensor:
    """
    Returns the electric dipole moment of the system, i.e. negative derivative
    of energy w.r.t. electric field.
    The dipole is pointing from negative to positive charge.

    Arguments
    ---------
    qc: BaseQCCalc
        The qc calc object that has been executed.
    unit: str or None
        The returned dipole unit. If ``None``, returns in atomic unit.

    Returns
    -------
    torch.Tensor
        Tensor representing the dipole moment in atomic unit with shape ``(ndim,)``
    """
    edip = _edipole(qc)
    edip = edipole_to(edip, unit)
    return edip

def equadrupole(qc: BaseQCCalc, unit: Optional[str] = "Debye*Angst") -> torch.Tensor:
    """
    Returns the electric quadrupole moment of the system, i.e. derivative of energy
    w.r.t. electric field.

    Arguments
    ---------
    qc: BaseQCCalc
        The qc calc object that has been executed.
    unit: str or None
        The returned quadrupole unit. If ``None``, returns in atomic unit.

    Returns
    -------
    torch.Tensor
        Tensor representing the quadrupole moment in atomic unit in ``(ndim, ndim)``
    """
    equad = _equadrupole(qc)
    equad = equadrupole_to(equad, unit)
    return equad

@memoize_method
def lowest_eival_orb_hessian(qc: BaseQCCalc) -> torch.Tensor:
    """
    Get the lowest eigenvalue of the orbital Hessian

    Arguments
    ---------
    qc: BaseQCCalc
        The qc calc object that has been executed.

    Returns
    -------
    torch.Tensor
        A single-element tensor representing the lowest eigenvalue of the
        Hessian of energy with respect to orbital parameters.
        It is useful to check the convergence stability whether it ends up
        in a ground state or an excited state.
    """
    # check if the orbital is in the ground state
    dm = qc.aodm()
    polarized = isinstance(dm, SpinParam)
    system = qc.get_system()
    h = system.get_hamiltonian()

    # (nao, norb)
    orb_weights = system.get_orbweight(polarized=polarized)
    norb = SpinParam.apply_fcn(lambda orb_weights: len(orb_weights), orb_weights)
    norb_max = SpinParam.reduce(norb, max)
    orb_p = SpinParam.apply_fcn(
        lambda dm, norb: h.dm2ao_orb_params(dm, norb=norb), dm, norb)  # (*, nao, norb1), (*, nao, norb2)

    # concatenate the parameters in -1 dim if it is polarized
    if isinstance(orb_p, SpinParam):
        orb_params = torch.cat((orb_p.u, orb_p.d), dim=-1).detach().requires_grad_()
    else:
        orb_params = orb_p.detach().requires_grad_()

    # now reconstruct the orbital from the orbital parameters (just to construct
    # the graph)
    def get_ene(orb_params):
        if polarized:
            orb_p = SpinParam(u=orb_params[..., :norb.u], d=orb_params[..., norb.u:])
        else:
            orb_p = orb_params
        dm2 = SpinParam.apply_fcn(
            lambda orb_p, orb_weights: h.ao_orb_params2dm(orb_p, orb_weights),
            orb_p, orb_weights)
        ene = qc.dm2energy(dm2)
        return ene

    # construct the hessian of the energy w.r.t. orb_params
    hess = xt.grad.hess(get_ene, (orb_params,), idxs=0)
    assert isinstance(hess, xt.LinearOperator)

    # get the lowest eigenvalue
    eival, eivec = xt.linalg.symeig(hess, neig=1, mode="lowest")
    return eival

def is_orb_min(qc: BaseQCCalc, threshold: float = -1e-3) -> bool:
    """
    Check the stability of the SCF if it is the minimum, not a saddle point.

    Arguments
    ---------
    qc: BaseQCCalc
        The qc calc object that has been executed.
    threshold: float
        The threshold value of the lowest eigenvalue of the Hessian matrix to be
        qualified as a positive definite matrix.

    Returns
    -------
    bool
        ``True`` if the state is minimum, otherwise returns ``False``.
    """
    eival = lowest_eival_orb_hessian(qc)
    return bool(torch.all(eival > threshold))

@memoize_method
def _hessian_pos(qc: BaseQCCalc) -> torch.Tensor:
    # calculate the hessian in atomic unit
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
def _vibration(qc: BaseQCCalc) -> Tuple[torch.Tensor, torch.Tensor]:
    # calculate the vibrational mode and returns the frequencies and normal modes
    # freqs are sorted from largest to smallest

    hess = hessian_pos(qc)  # (natoms * ndim, natoms * ndim)
    mass = qc.get_system().atommasses  # (natoms)
    ndim = hess.shape[0] // mass.shape[0]
    mass_mat = torch.diag_embed(mass.repeat_interleave(ndim))  # (natoms * ndim)^2

    # eival: (natoms * ndim)
    # eivec: (natoms * ndim, natoms * ndim)
    # eival and eivec are automatically sorted from smallest eival to the largest
    hess = (hess + hess.transpose(-2, -1).conj()) * 0.5
    Alinop = xt.LinearOperator.m(hess, is_hermitian=True)
    Mlinop = xt.LinearOperator.m(mass_mat, is_hermitian=True)
    eival, eivec = xt.linalg.symeig(A=Alinop, M=Mlinop)
    freq = (eival.abs() ** 0.5) * torch.sign(eival) / (2 * np.pi)

    # reverse the sorting to make it sorted from largest to smallest
    freq = torch.flip(freq, dims=(-1,))
    eivec = torch.flip(eivec, dims=(-1,))
    return freq, eivec

@memoize_method
def _ir_spectrum(qc: BaseQCCalc) -> Tuple[torch.Tensor, torch.Tensor]:
    # Calculates the IR vibrational spectrum and returns the frequencies and
    # intensities.
    # Unlike vibration, this function only returns the positive frequencies,
    # sorted from the largest frequency to the lowest frequency.
    system = qc.get_system()
    atompos = system.atompos  # (natoms, ndim)

    # only retain the positive frequencies
    freqs, normal_modes = _vibration(qc)  # (natoms * ndim), (natoms * ndim, natoms * ndim)
    freqs, normal_modes = _only_positive_freqs(freqs, normal_modes)

    # get the derivative of dipole moment w.r.t. positions
    with torch.enable_grad():
        mu = _edipole(qc)  # (ndim)
    dmu_dr = _jac(mu, atompos)  # (ndim, natoms * ndim)
    dmu_dq = torch.matmul(dmu_dr, normal_modes)  # (ndim, nfreqs)
    ir_ints = torch.einsum("df,df->f", dmu_dq, dmu_dq)  # (nfreqs,)

    return freqs, ir_ints

@memoize_method
def _raman_spectrum(qc: BaseQCCalc) -> Tuple[torch.Tensor, torch.Tensor]:
    # calculate the frequency and intensity of Raman spectra in atomic unit
    # ref: https://doi.org/10.1080/00268970701516412
    system = qc.get_system()
    atompos = system.atompos  # (natoms, ndim)
    efields = system.efield
    assert isinstance(efields, tuple) and len(efields) >= 1

    # get the vibrational frequencies and normal modes
    # freqs: (nmodes,)
    # normal_modes: (natoms * ndim, nmodes)
    freqs, normal_modes = _only_positive_freqs(*_vibration(qc))

    # get the derivative of dipole moment w.r.t. efield and positions
    with torch.enable_grad():
        mu = _edipole(qc)  # (ndim)
        alpha = _jac(mu, efields[0])  # (ndim, ndim)
    dalpha_dr = _jac(alpha, atompos)  # (ndim, ndim, natoms * ndim)
    dalpha_dq = torch.matmul(dalpha_dr, normal_modes)  # (ndim, ndim, nmodes)

    # eq (3) & (4) in the ref
    alpha_p2 = (torch.einsum("iim->m", dalpha_dq) / 3.0) ** 2
    gamma_p2 = 0.5 * ((dalpha_dq[0, 0] - dalpha_dq[1, 1]) ** 2 +
                      (dalpha_dq[0, 0] - dalpha_dq[2, 2]) ** 2 +
                      (dalpha_dq[1, 1] - dalpha_dq[2, 2]) ** 2 +
                      3 * (dalpha_dq[0, 1] ** 2 + dalpha_dq[0, 2] ** 2 +
                           dalpha_dq[1, 0] ** 2 + dalpha_dq[1, 2] ** 2 +
                           dalpha_dq[2, 0] ** 2 + dalpha_dq[2, 1] ** 2))

    # eq (2) in the ref
    raman_ints = 45 * alpha_p2 + 7 * gamma_p2
    return freqs, raman_ints

@memoize_method
def _edipole(qc: BaseQCCalc) -> torch.Tensor:
    # calculate the electric dipole and returns in atomic unit
    # the electric dipole is pointing from - to + charge
    ene = qc.energy()
    system = qc.get_system()
    efield = system.efield
    assert isinstance(efield, tuple)
    assert len(efield) > 0, "To calculate dipole, the constant electric field must be provided"

    # check if the electric field requires grad
    _check_differentiability(efield[0], "electric field", "dipole")
    assert isinstance(efield[0], torch.Tensor)

    # get the contribution from electron
    dipole = -_jac(ene, efield[0])

    # get the contribution from ions
    atompos = system.atompos  # (natoms, ndim)
    atomzs = system.atomzs.to(atompos.dtype)  # (natoms)
    ion_dipole = torch.einsum("ad,a->d", atompos, atomzs)

    return dipole + ion_dipole

@memoize_method
def _equadrupole(qc: BaseQCCalc) -> torch.Tensor:
    # calculate the electric quadrupole and returns in atomic unit
    ene = qc.energy()
    system = qc.get_system()
    efield = system.efield
    assert isinstance(efield, tuple)
    assert len(efield) > 1, "To calculate quadrupole, the gradient electric field must be provided"

    # check if the electric field requires grad
    _check_differentiability(efield[1], "gradient electric field", "quadpole")
    assert isinstance(efield[1], torch.Tensor)

    ndim = 3
    quadrupole = -2 * _jac(ene, efield[1], create_graph=True)  # (ndim, ndim)
    quadrupole = quadrupole.reshape(ndim, ndim)

    # get the contribution from ions
    atompos = system.atompos  # (natoms, ndim)
    atomzs = system.atomzs.to(atompos.dtype)  # (natoms)
    ion_quadrupole = torch.einsum("ad,ae,a->de", atompos, atompos, atomzs)

    return quadrupole + ion_quadrupole

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

def _only_positive_freqs(freqs: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # only selects the components corresponding to the positive frequencies
    pos_freqs = freqs > 0
    freqs = freqs[pos_freqs]  # (nfreqs,)
    x = x[:, pos_freqs]  # (natoms * ndim, nfreqs)
    return freqs, x
