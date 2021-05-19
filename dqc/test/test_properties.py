from typing import Union, List
import torch
import numpy as np
import pytest
import psutil
from dqc.api.properties import hessian_pos, vibration, edipole, equadrupole, \
                               ir_spectrum, raman_spectrum, is_orb_min
from dqc.system.mol import Mol
from dqc.qccalc.hf import HF
from dqc.xc.base_xc import BaseXC
from dqc.utils.safeops import safepow
from dqc.utils.datastruct import ValGrad, SpinParam

dtype = torch.float64

@pytest.fixture
def h2o_qc():
    # run the self-consistent HF iteration for h2o
    atomzs = torch.tensor([8, 1, 1], dtype=torch.int64)
    # from CCCBDB (calculated geometry for H2O)
    atomposs = torch.tensor([
        [0.0, 0.0, 0.2156],
        [0.0, 1.4749, -0.8625],
        [0.0, -1.4749, -0.8625],
    ], dtype=dtype).requires_grad_()
    efield = torch.zeros(3, dtype=dtype).requires_grad_()
    grad_efield = torch.zeros((3, 3), dtype=dtype).requires_grad_()

    efields = (efield, grad_efield)
    mol = Mol(moldesc=(atomzs, atomposs), basis="3-21G", dtype=dtype, efield=efields)
    qc = HF(mol).run()
    return qc

def test_hess(h2o_qc):
    # test if the hessian is Hermitian
    hess = hessian_pos(h2o_qc)
    assert torch.allclose(hess, hess.transpose(-2, -1).conj(), atol=2e-6)

def test_vibration(h2o_qc):
    # test if the vibration of h2o is similar to what pyscf computes

    freq_cm1, normcoord = vibration(h2o_qc, freq_unit="cm^-1")

    # from CCCBDB (calculated frequencies for H2O)
    calc_freq_cm1 = torch.tensor([3944., 3811., 1800.], dtype=dtype)

    assert torch.allclose(freq_cm1[:3], calc_freq_cm1, rtol=1e-3)

def test_edipole(h2o_qc):
    # test if the electric dipole of h2o similar to pyscf

    h2o_dip = edipole(h2o_qc, unit="debye")

    # from cccbdb (calculated electric dipole moment for H2O)
    pyscf_h2o_dip = torch.tensor([0, 0, -2.388e+00], dtype=dtype)

    assert torch.allclose(h2o_dip, pyscf_h2o_dip, rtol=2e-4)

def test_equadrupole(h2o_qc):
    # test if the quadrupole properties close to cccbdb precomputed values

    h2o_quad = equadrupole(h2o_qc, unit="debye*angst")

    cccbdb_h2o_quad = torch.tensor([
        [-6.838, 0.0, 0.0],
        [0.0, -3.972, 0.0],
        [0.0, 0.0, -5.882],
    ], dtype=dtype)

    assert torch.allclose(h2o_quad, cccbdb_h2o_quad, rtol=2e-4)

def test_ir_spectrum(h2o_qc):
    freq, ir_ints = ir_spectrum(h2o_qc, freq_unit="cm^-1", ints_unit="km/mol")

    # from CCCBDB (calculated frequencies for H2O)
    calc_freq_cm1 = torch.tensor([3944., 3811., 1800.], dtype=dtype)
    # from CCCBDB (calculated vibrational properties for H2O)
    ir_ints1 = torch.tensor([9.123e+00, 4.7e-02, 7.989e+01], dtype=dtype)

    assert torch.allclose(freq[:3], calc_freq_cm1, rtol=1e-3)
    assert torch.allclose(ir_ints[:3], ir_ints1, rtol=1e-2)

def test_raman_spectrum(h2o_qc):
    freq, raman_ints = raman_spectrum(h2o_qc, freq_unit="cm^-1", ints_unit="angst^4/amu")

    # from CCCBDB (calculated vibrational properties for H2O)
    calc_raman_ints1 = torch.tensor([44.12, 95.71, 11.5], dtype=dtype)
    assert torch.allclose(raman_ints[:3], calc_raman_ints1, rtol=1e-3)

def test_stability_check(h2o_qc):
    assert is_orb_min(h2o_qc)

def test_instability_check():
    # check if is_orb_min returns False for a known excited state

    atomzs = [8, 8]  # O2
    atomposs = torch.tensor([[-1.0, 0., 0.], [1.0, 0., 0.]], dtype=dtype)
    mol = Mol(moldesc=(atomzs, atomposs), basis="3-21G", dtype=dtype)

    # known excited state of O2
    dm0 = torch.tensor([[ 2.0565e+00,  5.7991e-02, -1.4252e-02, -1.4887e-10,  3.4960e-12,
         -4.9103e-01, -4.8578e-02, -1.8022e-10,  4.2321e-12, -6.0466e-03,
         -9.3459e-03,  9.2610e-02,  1.5246e-10, -3.5804e-12,  1.8488e-01,
          1.2152e-03,  1.9884e-10, -4.6695e-12],
        [ 5.7991e-02,  1.5438e-01,  2.7579e-02,  5.6299e-11, -1.3222e-12,
          4.0907e-01,  2.2136e-02,  7.2174e-11, -1.6949e-12, -9.3459e-03,
          1.2575e-02, -9.5869e-02, -5.1626e-11,  1.2124e-12, -1.1735e-01,
         -2.8390e-02, -7.8057e-11,  1.8331e-12],
        [-1.4252e-02,  2.7579e-02,  7.0115e-02, -1.0749e-09,  2.5243e-11,
         -3.0443e-02,  2.1238e-02, -1.3078e-09,  3.0711e-11, -9.2610e-02,
          9.5869e-02, -3.1184e-02,  1.0017e-09, -2.3524e-11,  2.7358e-01,
         -2.0648e-02,  1.2479e-09, -2.9305e-11],
        [-1.4887e-10,  5.6299e-11, -1.0749e-09,  6.7372e-01, -2.6033e-17,
          8.7507e-10, -9.8746e-10,  7.3367e-01,  1.7549e-17, -1.5246e-10,
          5.1626e-11,  1.0017e-09, -8.3089e-02, -3.6066e-17,  1.0181e-09,
          8.6090e-10, -2.0251e-01,  3.4694e-17],
        [ 3.4960e-12, -1.3222e-12,  2.5243e-11, -2.6033e-17,  6.7372e-01,
         -2.0550e-11,  2.3189e-11, -6.9415e-18,  7.3367e-01,  3.5804e-12,
         -1.2123e-12, -2.3524e-11, -5.8293e-17, -8.3089e-02, -2.3909e-11,
         -2.0217e-11, -7.6328e-17, -2.0251e-01],
        [-4.9103e-01,  4.0907e-01, -3.0443e-02,  8.7507e-10, -2.0550e-11,
          1.4806e+00,  4.8853e-02,  1.0990e-09, -2.5808e-11,  1.8488e-01,
         -1.1735e-01, -2.7358e-01, -1.0181e-09,  2.3909e-11, -8.9577e-01,
         -5.1827e-02, -1.2743e-09,  2.9924e-11],
        [-4.8578e-02,  2.2136e-02,  2.1238e-02, -9.8746e-10,  2.3189e-11,
          4.8853e-02,  9.2908e-03, -1.1974e-09,  2.8120e-11, -1.2152e-03,
          2.8390e-02, -2.0648e-02,  8.6090e-10, -2.0217e-11,  5.1827e-02,
         -8.0444e-03,  1.0836e-09, -2.5447e-11],
        [-1.8022e-10,  7.2174e-11, -1.3078e-09,  7.3367e-01, -6.9415e-18,
          1.0990e-09, -1.1974e-09,  8.1787e-01,  2.7034e-17, -1.9884e-10,
          7.8056e-11,  1.2479e-09, -2.0251e-01, -9.3297e-17,  1.2743e-09,
          1.0836e-09, -3.4018e-01,  3.4694e-18],
        [ 4.2321e-12, -1.6949e-12,  3.0711e-11,  1.7549e-17,  7.3367e-01,
         -2.5808e-11,  2.8120e-11,  2.7034e-17,  8.1787e-01,  4.6696e-12,
         -1.8331e-12, -2.9305e-11,  8.5966e-17, -2.0251e-01, -2.9925e-11,
         -2.5447e-11,  4.5103e-17, -3.4018e-01],
        [-6.0466e-03, -9.3459e-03, -9.2610e-02, -1.5246e-10,  3.5804e-12,
          1.8488e-01, -1.2152e-03, -1.9884e-10,  4.6696e-12,  2.0565e+00,
          5.7991e-02,  1.4252e-02,  1.4887e-10, -3.4960e-12, -4.9103e-01,
          4.8578e-02,  1.8022e-10, -4.2321e-12],
        [-9.3459e-03,  1.2575e-02,  9.5869e-02,  5.1626e-11, -1.2123e-12,
         -1.1735e-01,  2.8390e-02,  7.8056e-11, -1.8331e-12,  5.7991e-02,
          1.5438e-01, -2.7579e-02, -5.6299e-11,  1.3221e-12,  4.0907e-01,
         -2.2136e-02, -7.2173e-11,  1.6949e-12],
        [ 9.2610e-02, -9.5869e-02, -3.1184e-02,  1.0017e-09, -2.3524e-11,
         -2.7358e-01, -2.0648e-02,  1.2479e-09, -2.9305e-11,  1.4252e-02,
         -2.7579e-02,  7.0115e-02, -1.0749e-09,  2.5243e-11,  3.0443e-02,
          2.1238e-02, -1.3078e-09,  3.0711e-11],
        [ 1.5246e-10, -5.1626e-11,  1.0017e-09, -8.3089e-02, -5.8293e-17,
         -1.0181e-09,  8.6090e-10, -2.0251e-01,  8.5966e-17,  1.4887e-10,
         -5.6299e-11, -1.0749e-09,  6.7372e-01,  1.2840e-17, -8.7507e-10,
         -9.8746e-10,  7.3367e-01, -1.3878e-16],
        [-3.5804e-12,  1.2124e-12, -2.3524e-11, -3.6066e-17, -8.3089e-02,
          2.3909e-11, -2.0217e-11, -9.3297e-17, -2.0251e-01, -3.4960e-12,
          1.3221e-12,  2.5243e-11,  1.2840e-17,  6.7372e-01,  2.0550e-11,
          2.3189e-11,  1.7000e-16,  7.3367e-01],
        [ 1.8488e-01, -1.1735e-01,  2.7358e-01,  1.0181e-09, -2.3909e-11,
         -8.9577e-01,  5.1827e-02,  1.2743e-09, -2.9925e-11, -4.9103e-01,
          4.0907e-01,  3.0443e-02, -8.7507e-10,  2.0550e-11,  1.4806e+00,
         -4.8853e-02, -1.0990e-09,  2.5808e-11],
        [ 1.2152e-03, -2.8390e-02, -2.0648e-02,  8.6090e-10, -2.0217e-11,
         -5.1827e-02, -8.0444e-03,  1.0836e-09, -2.5447e-11,  4.8578e-02,
         -2.2136e-02,  2.1238e-02, -9.8746e-10,  2.3189e-11, -4.8853e-02,
          9.2908e-03, -1.1974e-09,  2.8119e-11],
        [ 1.9884e-10, -7.8057e-11,  1.2479e-09, -2.0251e-01, -8.0051e-17,
         -1.2743e-09,  1.0836e-09, -3.4018e-01,  4.1598e-17,  1.8022e-10,
         -7.2173e-11, -1.3078e-09,  7.3367e-01,  1.6988e-16, -1.0990e-09,
         -1.1974e-09,  8.1787e-01,  1.0408e-17],
        [-4.6695e-12,  1.8331e-12, -2.9305e-11,  3.6171e-17, -2.0251e-01,
          2.9924e-11, -2.5447e-11,  2.8404e-18, -3.4018e-01, -4.2321e-12,
          1.6949e-12,  3.0711e-11, -1.3969e-16,  7.3367e-01,  2.5808e-11,
          2.8119e-11,  1.0408e-17,  8.1787e-01]], dtype=dtype)
    qc = HF(mol).run(dm0=dm0)
    ene = qc.energy()

    assert not is_orb_min(qc)

@pytest.mark.parametrize(
    "check_type",
    ["ene", "jac_ene"]
)
def test_properties_gradcheck(check_type):
    # check if the analytical formula required to calculate the properties
    # agrees with numerical difference
    # NOTE: very slow

    atomzs = torch.tensor([8, 1, 1], dtype=torch.int64)
    atomposs = torch.tensor([
        [0.0, 0.0, 0.2217],
        [0.0, 1.4309, -0.8867],
        [0.0, -1.4309, -0.8867],
    ], dtype=dtype).requires_grad_()

    # test gradient on electric field
    efield = torch.zeros(3, dtype=dtype).requires_grad_()
    grad_efield = torch.zeros((3, 3), dtype=dtype).requires_grad_()

    def get_energy(atomposs, efield, grad_efield):
        efields = (efield, grad_efield)
        mol = Mol(moldesc=(atomzs, atomposs), basis="3-21G", dtype=dtype, efield=efields)
        qc = HF(mol).run()
        ene = qc.energy()
        return ene

    if check_type == "ene":
        # dipole and quadrupole
        torch.autograd.gradcheck(get_energy, (atomposs, efield, grad_efield))
        # 2nd grad for hessian, ir intensity, and part of raman intensity
        torch.autograd.gradgradcheck(get_energy, (atomposs, efield, grad_efield.detach()))

    def get_jac_ene(atomposs, efield, grad_efield):
        # get the jacobian of energy w.r.t. atompos
        atomposs = atomposs.requires_grad_()
        ene = get_energy(atomposs, efield, grad_efield)
        jac_ene = torch.autograd.grad(ene, atomposs, create_graph=True)[0]
        return jac_ene

    if check_type == "jac_ene":
        torch.autograd.gradcheck(get_jac_ene, (atomposs.detach(), efield, grad_efield.detach()))
        # raman spectra intensities
        torch.autograd.gradgradcheck(get_jac_ene, (atomposs.detach(), efield, grad_efield.detach()),
                                     atol=3e-4)
