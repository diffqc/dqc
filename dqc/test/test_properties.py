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
    mol = Mol(moldesc=(atomzs, atomposs), basis="3-21G", dtype=dtype, spin=2)

    # known excited state of O2
    dm0_u = torch.tensor([[ 1.0294e+00,  2.6963e-02,  2.5265e-02, -1.6489e-10, -2.4207e-11,
             -2.4604e-01,  2.0213e-03, -8.2600e-11, -1.2127e-11,  2.8280e-03,
             -8.1538e-03,  3.0887e-02, -1.7132e-10, -2.5153e-11,  5.4352e-02,
             -1.0137e-02,  1.9502e-10,  2.8631e-11],
            [ 2.6963e-02,  8.3028e-02, -1.0376e-02,  7.0493e-10,  1.0349e-10,
              2.0755e-01, -1.0429e-02,  3.7677e-10,  5.5315e-11, -8.1538e-03,
              7.9825e-03, -4.3185e-02, -3.4201e-10, -5.0211e-11, -3.2869e-02,
             -1.0274e-02, -7.8769e-10, -1.1564e-10],
            [ 2.5265e-02, -1.0376e-02,  2.2042e-01, -9.1151e-10, -1.3382e-10,
             -1.7641e-01,  1.4353e-01, -4.7051e-10, -6.9077e-11, -3.0887e-02,
              4.3185e-02, -1.7566e-01, -2.3912e-10, -3.5106e-11,  2.8375e-02,
             -1.2863e-01,  3.5324e-11,  5.1860e-12],
            [-1.6489e-10,  7.0493e-10, -9.1151e-10,  3.4294e-01, -1.4140e-03,
              1.7234e-09,  1.5858e-09,  3.6662e-01,  7.1637e-05,  1.7132e-10,
              3.4201e-10, -2.3912e-10, -4.2419e-02,  1.5676e-04, -2.5084e-09,
              1.0616e-09, -1.0085e-01, -1.0083e-04],
            [-2.4207e-11,  1.0349e-10, -1.3382e-10, -1.4140e-03,  3.5236e-01,
              2.5301e-10,  2.3282e-10,  7.1637e-05,  3.6614e-01,  2.5152e-11,
              5.0211e-11, -3.5105e-11,  1.5676e-04, -4.3463e-02, -3.6826e-10,
              1.5586e-10, -1.0083e-04, -1.0018e-01],
            [-2.4604e-01,  2.0755e-01, -1.7641e-01,  1.7234e-09,  2.5301e-10,
              7.2183e-01, -1.1077e-01, -9.9618e-11, -1.4625e-11,  5.4352e-02,
             -3.2869e-02, -2.8375e-02,  2.5084e-09,  3.6826e-10, -2.1715e-01,
              4.3801e-02,  1.1914e-09,  1.7491e-10],
            [ 2.0213e-03, -1.0429e-02,  1.4353e-01,  1.5858e-09,  2.3282e-10,
             -1.1077e-01,  9.9498e-02,  1.7121e-09,  2.5136e-10,  1.0137e-02,
              1.0274e-02, -1.2863e-01,  1.0616e-09,  1.5586e-10, -4.3801e-02,
             -9.3783e-02,  8.3368e-10,  1.2239e-10],
            [-8.2600e-11,  3.7677e-10, -4.7051e-10,  3.6662e-01,  7.1637e-05,
             -9.9618e-11,  1.7121e-09,  4.0106e-01,  1.7817e-03, -1.9502e-10,
              7.8769e-10,  3.5324e-11, -1.0085e-01, -1.0083e-04, -1.1914e-09,
              8.3368e-10, -1.6602e-01, -8.7540e-04],
            [-1.2127e-11,  5.5315e-11, -6.9077e-11,  7.1637e-05,  3.6614e-01,
             -1.4625e-11,  2.5136e-10,  1.7817e-03,  3.8918e-01, -2.8631e-11,
              1.1564e-10,  5.1860e-12, -1.0083e-04, -1.0018e-01, -1.7491e-10,
              1.2240e-10, -8.7540e-04, -1.6019e-01],
            [ 2.8280e-03, -8.1538e-03, -3.0887e-02,  1.7132e-10,  2.5152e-11,
              5.4352e-02,  1.0137e-02, -1.9502e-10, -2.8631e-11,  1.0294e+00,
              2.6963e-02, -2.5265e-02,  1.6489e-10,  2.4207e-11, -2.4604e-01,
             -2.0213e-03,  8.2600e-11,  1.2127e-11],
            [-8.1538e-03,  7.9825e-03,  4.3185e-02,  3.4201e-10,  5.0211e-11,
             -3.2869e-02,  1.0274e-02,  7.8769e-10,  1.1564e-10,  2.6963e-02,
              8.3028e-02,  1.0376e-02, -7.0493e-10, -1.0349e-10,  2.0755e-01,
              1.0429e-02, -3.7677e-10, -5.5314e-11],
            [ 3.0887e-02, -4.3185e-02, -1.7566e-01, -2.3912e-10, -3.5105e-11,
             -2.8375e-02, -1.2863e-01,  3.5324e-11,  5.1860e-12, -2.5265e-02,
              1.0376e-02,  2.2042e-01, -9.1151e-10, -1.3382e-10,  1.7641e-01,
              1.4353e-01, -4.7051e-10, -6.9077e-11],
            [-1.7132e-10, -3.4201e-10, -2.3912e-10, -4.2419e-02,  1.5676e-04,
              2.5084e-09,  1.0616e-09, -1.0085e-01, -1.0083e-04,  1.6489e-10,
             -7.0493e-10, -9.1151e-10,  3.4294e-01, -1.4140e-03, -1.7234e-09,
              1.5858e-09,  3.6662e-01,  7.1637e-05],
            [-2.5153e-11, -5.0211e-11, -3.5106e-11,  1.5676e-04, -4.3463e-02,
              3.6826e-10,  1.5586e-10, -1.0083e-04, -1.0018e-01,  2.4207e-11,
             -1.0349e-10, -1.3382e-10, -1.4140e-03,  3.5236e-01, -2.5301e-10,
              2.3282e-10,  7.1637e-05,  3.6614e-01],
            [ 5.4352e-02, -3.2869e-02,  2.8375e-02, -2.5084e-09, -3.6826e-10,
             -2.1715e-01, -4.3801e-02, -1.1914e-09, -1.7491e-10, -2.4604e-01,
              2.0755e-01,  1.7641e-01, -1.7234e-09, -2.5301e-10,  7.2183e-01,
              1.1077e-01,  9.9618e-11,  1.4625e-11],
            [-1.0137e-02, -1.0274e-02, -1.2863e-01,  1.0616e-09,  1.5586e-10,
              4.3801e-02, -9.3783e-02,  8.3368e-10,  1.2240e-10, -2.0213e-03,
              1.0429e-02,  1.4353e-01,  1.5858e-09,  2.3282e-10,  1.1077e-01,
              9.9498e-02,  1.7121e-09,  2.5136e-10],
            [ 1.9502e-10, -7.8769e-10,  3.5324e-11, -1.0085e-01, -1.0083e-04,
              1.1914e-09,  8.3368e-10, -1.6602e-01, -8.7540e-04,  8.2600e-11,
             -3.7677e-10, -4.7051e-10,  3.6662e-01,  7.1637e-05,  9.9618e-11,
              1.7121e-09,  4.0106e-01,  1.7817e-03],
            [ 2.8631e-11, -1.1564e-10,  5.1860e-12, -1.0083e-04, -1.0018e-01,
              1.7491e-10,  1.2239e-10, -8.7540e-04, -1.6019e-01,  1.2127e-11,
             -5.5314e-11, -6.9077e-11,  7.1637e-05,  3.6614e-01,  1.4625e-11,
              2.5136e-10,  1.7817e-03,  3.8918e-01]], dtype=dtype)
    dm0_d = torch.tensor([[ 1.0280e+00,  3.1794e-02, -1.0819e-02,  1.2262e-08,  1.8002e-09,
        -2.5047e-01, -3.0175e-02,  1.4768e-08,  2.1682e-09, -4.3972e-03,
        -3.7278e-03,  4.4949e-02, -1.2216e-08, -1.7934e-09,  1.0229e-01,
        -5.3830e-04, -1.6867e-08, -2.4764e-09],
        [ 3.1794e-02,  6.9842e-02,  1.8088e-02, -3.9174e-09, -5.7513e-10,
        1.9844e-01,  1.5956e-02, -4.7078e-09, -6.9116e-10, -3.7278e-03,
        5.2292e-03, -4.4950e-02,  3.1146e-09,  4.5727e-10, -6.3267e-02,
        -1.3251e-02,  5.7055e-09,  8.3765e-10],
        [-1.0819e-02,  1.8088e-02,  3.6338e-02,  8.0914e-08,  1.1879e-08,
        -1.8540e-03,  1.2511e-02,  9.8821e-08,  1.4508e-08, -4.4949e-02,
        4.4950e-02, -2.2809e-02, -7.2340e-08, -1.0621e-08,  1.3093e-01,
        -1.5190e-02, -9.1728e-08, -1.3467e-08],
        [ 1.2262e-08, -3.9174e-09,  8.0914e-08,  3.2507e-01,  3.0592e-02,
        -7.0288e-08,  7.4593e-08,  3.6175e-01,  3.4084e-02,  1.2216e-08,
        -3.1146e-09, -7.2340e-08, -4.0906e-02, -2.3139e-02, -8.0499e-08,
        -6.3907e-08, -9.6796e-02, -3.3237e-02],
        [ 1.8002e-09, -5.7513e-10,  1.1879e-08,  3.0592e-02,  1.2119e-01,
        -1.0319e-08,  1.0951e-08,  3.4084e-02,  1.3460e-01,  1.7934e-09,
        -4.5727e-10, -1.0621e-08, -2.3139e-02,  1.1330e-01, -1.1818e-08,
        -9.3824e-09, -3.3237e-02,  1.2471e-01],
        [-2.5047e-01,  1.9844e-01, -1.8540e-03, -7.0288e-08, -1.0319e-08,
        7.9298e-01,  4.3644e-02, -8.7023e-08, -1.2776e-08,  1.0229e-01,
        -6.3267e-02, -1.3093e-01,  8.0499e-08,  1.1818e-08, -5.1059e-01,
        -1.6902e-02,  1.0510e-07,  1.5429e-08],
        [-3.0175e-02,  1.5956e-02,  1.2511e-02,  7.4593e-08,  1.0951e-08,
        4.3644e-02,  7.0618e-03,  9.1473e-08,  1.3429e-08,  5.3830e-04,
        1.3251e-02, -1.5190e-02, -6.3907e-08, -9.3824e-09,  1.6902e-02,
        -5.9249e-03, -8.1371e-08, -1.1946e-08],
        [ 1.4768e-08, -4.7078e-09,  9.8821e-08,  3.6175e-01,  3.4084e-02,
        -8.7023e-08,  9.1473e-08,  4.1087e-01,  3.9192e-02,  1.6867e-08,
        -5.7055e-09, -9.1728e-08, -9.6796e-02, -3.3237e-02, -1.0510e-07,
        -8.1371e-08, -1.6366e-01, -4.5156e-02],
        [ 2.1682e-09, -6.9116e-10,  1.4508e-08,  3.4084e-02,  1.3460e-01,
        -1.2776e-08,  1.3429e-08,  3.9192e-02,  1.4967e-01,  2.4764e-09,
        -8.3765e-10, -1.3467e-08, -3.3237e-02,  1.2471e-01, -1.5429e-08,
        -1.1946e-08, -4.5156e-02,  1.3728e-01],
        [-4.3972e-03, -3.7278e-03, -4.4949e-02,  1.2216e-08,  1.7934e-09,
        1.0229e-01,  5.3830e-04,  1.6867e-08,  2.4764e-09,  1.0280e+00,
        3.1794e-02,  1.0819e-02, -1.2262e-08, -1.8002e-09, -2.5047e-01,
        3.0175e-02, -1.4768e-08, -2.1682e-09],
        [-3.7278e-03,  5.2292e-03,  4.4950e-02, -3.1146e-09, -4.5727e-10,
        -6.3267e-02,  1.3251e-02, -5.7055e-09, -8.3765e-10,  3.1794e-02,
        6.9842e-02, -1.8088e-02,  3.9174e-09,  5.7513e-10,  1.9844e-01,
        -1.5956e-02,  4.7078e-09,  6.9116e-10],
        [ 4.4949e-02, -4.4950e-02, -2.2809e-02, -7.2340e-08, -1.0621e-08,
        -1.3093e-01, -1.5190e-02, -9.1728e-08, -1.3467e-08,  1.0819e-02,
        -1.8088e-02,  3.6338e-02,  8.0914e-08,  1.1879e-08,  1.8540e-03,
        1.2511e-02,  9.8821e-08,  1.4508e-08],
        [-1.2216e-08,  3.1146e-09, -7.2340e-08, -4.0906e-02, -2.3139e-02,
        8.0499e-08, -6.3907e-08, -9.6796e-02, -3.3237e-02, -1.2262e-08,
        3.9174e-09,  8.0914e-08,  3.2507e-01,  3.0592e-02,  7.0288e-08,
        7.4593e-08,  3.6175e-01,  3.4084e-02],
        [-1.7934e-09,  4.5727e-10, -1.0621e-08, -2.3139e-02,  1.1330e-01,
        1.1818e-08, -9.3824e-09, -3.3237e-02,  1.2471e-01, -1.8002e-09,
        5.7513e-10,  1.1879e-08,  3.0592e-02,  1.2119e-01,  1.0319e-08,
        1.0951e-08,  3.4084e-02,  1.3460e-01],
        [ 1.0229e-01, -6.3267e-02,  1.3093e-01, -8.0499e-08, -1.1818e-08,
        -5.1059e-01,  1.6902e-02, -1.0510e-07, -1.5429e-08, -2.5047e-01,
        1.9844e-01,  1.8540e-03,  7.0288e-08,  1.0319e-08,  7.9298e-01,
        -4.3644e-02,  8.7023e-08,  1.2776e-08],
        [-5.3830e-04, -1.3251e-02, -1.5190e-02, -6.3907e-08, -9.3824e-09,
        -1.6902e-02, -5.9249e-03, -8.1371e-08, -1.1946e-08,  3.0175e-02,
        -1.5956e-02,  1.2511e-02,  7.4593e-08,  1.0951e-08, -4.3644e-02,
        7.0618e-03,  9.1473e-08,  1.3429e-08],
        [-1.6867e-08,  5.7055e-09, -9.1728e-08, -9.6796e-02, -3.3237e-02,
        1.0510e-07, -8.1371e-08, -1.6366e-01, -4.5156e-02, -1.4768e-08,
        4.7078e-09,  9.8821e-08,  3.6175e-01,  3.4084e-02,  8.7023e-08,
        9.1473e-08,  4.1087e-01,  3.9192e-02],
        [-2.4764e-09,  8.3765e-10, -1.3467e-08, -3.3237e-02,  1.2471e-01,
        1.5429e-08, -1.1946e-08, -4.5156e-02,  1.3728e-01, -2.1682e-09,
        6.9116e-10,  1.4508e-08,  3.4084e-02,  1.3460e-01,  1.2776e-08,
        1.3429e-08,  3.9192e-02,  1.4967e-01]], dtype=dtype)
    dm0 = SpinParam(u=dm0_u, d=dm0_d)
    qc = HF(mol).run(dm0=dm0)
    ene = qc.energy()
    assert not is_orb_min(qc)

    # known ground state of O2
    dm1 = SpinParam(u=torch.tensor([[ 1.0314e+00,  2.6402e-02,  2.2305e-02, -5.0988e-18,  7.6293e-17,
             -2.6505e-01, -1.0055e-02, -9.9891e-18, -1.4917e-17,  8.1977e-04,
             -6.9304e-03,  2.7420e-02,  1.6957e-19,  4.0982e-18,  7.3541e-02,
             -2.0496e-02,  2.3776e-18,  4.5883e-18],
            [ 2.6402e-02,  8.2333e-02, -2.7136e-03,  5.3653e-17,  2.8785e-18,
              2.1983e-01,  3.8132e-03,  6.2604e-17,  1.7648e-18, -6.9304e-03,
              7.9674e-03, -4.1288e-02, -1.2766e-17,  1.3455e-19, -4.9807e-02,
             -1.5893e-03, -2.0812e-17,  1.2729e-18],
            [ 2.2305e-02, -2.7136e-03,  2.1167e-01, -3.4947e-17,  6.8472e-17,
             -1.7250e-01,  1.3472e-01, -1.2506e-17, -2.6087e-17, -2.7420e-02,
              4.1288e-02, -1.8088e-01, -2.2394e-17,  1.7731e-17,  1.8459e-02,
             -1.3819e-01,  1.9477e-18,  1.8886e-17],
            [-5.0988e-18,  5.3653e-17, -3.4947e-17,  3.5763e-01, -7.9483e-09,
             -9.0265e-17,  3.4610e-17,  3.6552e-01,  6.4945e-10, -3.6333e-17,
              6.7020e-17, -2.9883e-17, -4.7618e-02,  1.6956e-09, -2.4199e-17,
             -8.3485e-17, -9.9544e-02, -7.3520e-10],
            [ 7.6293e-17,  2.8785e-18,  6.8472e-17, -7.9483e-09,  3.5763e-01,
              1.7140e-16,  2.4282e-16,  6.4945e-10,  3.6552e-01,  1.0004e-17,
              3.8007e-17, -7.1669e-17,  1.6956e-09, -4.7618e-02, -2.8480e-16,
              1.7399e-16, -7.3520e-10, -9.9544e-02],
            [-2.6505e-01,  2.1983e-01, -1.7250e-01, -9.0265e-17,  1.7140e-16,
              8.4827e-01, -6.7474e-02, -7.0544e-17,  2.5447e-16,  7.3541e-02,
             -4.9807e-02, -1.8459e-02,  8.0806e-17, -4.1889e-17, -3.3721e-01,
              8.9809e-02,  8.2853e-17, -7.0800e-17],
            [-1.0055e-02,  3.8132e-03,  1.3472e-01,  3.4610e-17,  2.4282e-16,
             -6.7474e-02,  9.8213e-02,  5.9180e-17,  1.9000e-16,  2.0496e-02,
              1.5893e-03, -1.3819e-01,  2.3614e-17, -9.1877e-17, -8.9809e-02,
             -9.7167e-02,  3.3540e-17, -1.1837e-16],
            [-9.9891e-18,  6.2604e-17, -1.2506e-17,  3.6552e-01,  6.4945e-10,
             -7.0544e-17,  5.9180e-17,  3.8096e-01,  1.0167e-08,  1.1258e-17,
              3.5967e-17, -7.1515e-17, -9.9544e-02, -7.3520e-10, -1.4324e-16,
             -1.2175e-16, -1.5276e-01, -5.7119e-09],
            [-1.4917e-17,  1.7648e-18, -2.6087e-17,  6.4945e-10,  3.6552e-01,
              2.5447e-16,  1.9000e-16,  1.0167e-08,  3.8096e-01, -7.5979e-17,
              4.2324e-17,  2.5391e-17, -7.3520e-10, -9.9544e-02, -2.3320e-16,
              2.4117e-16, -5.7119e-09, -1.5276e-01],
            [ 8.1977e-04, -6.9304e-03, -2.7420e-02, -3.6333e-17,  1.0004e-17,
              7.3541e-02,  2.0496e-02,  1.1258e-17, -7.5979e-17,  1.0314e+00,
              2.6402e-02, -2.2305e-02,  1.1060e-17,  4.2626e-17, -2.6505e-01,
              1.0055e-02,  1.6638e-17,  5.6828e-17],
            [-6.9304e-03,  7.9674e-03,  4.1288e-02,  6.7020e-17,  3.8007e-17,
             -4.9807e-02,  1.5893e-03,  3.5967e-17,  4.2324e-17,  2.6402e-02,
              8.2333e-02,  2.7136e-03,  1.9327e-17, -4.4877e-17,  2.1983e-01,
             -3.8132e-03,  6.9816e-18, -5.1567e-17],
            [ 2.7420e-02, -4.1288e-02, -1.8088e-01, -2.9883e-17, -7.1669e-17,
             -1.8459e-02, -1.3819e-01, -7.1515e-17,  2.5391e-17, -2.2305e-02,
              2.7136e-03,  2.1167e-01, -2.3998e-17, -4.5347e-17,  1.7250e-01,
              1.3472e-01, -4.0526e-17, -4.8017e-17],
            [ 1.6957e-19, -1.2766e-17, -2.2394e-17, -4.7618e-02,  1.6956e-09,
              8.0806e-17,  2.3614e-17, -9.9544e-02, -7.3520e-10,  1.1060e-17,
              1.9327e-17, -2.3998e-17,  3.5763e-01, -7.9483e-09, -6.4576e-17,
              9.2303e-17,  3.6552e-01,  6.4945e-10],
            [ 4.0982e-18,  1.3455e-19,  1.7731e-17,  1.6956e-09, -4.7618e-02,
             -4.1889e-17, -9.1877e-17, -7.3520e-10, -9.9544e-02,  4.2626e-17,
             -4.4877e-17, -4.5347e-17, -7.9483e-09,  3.5763e-01,  1.0087e-16,
              3.4536e-18,  6.4945e-10,  3.6552e-01],
            [ 7.3541e-02, -4.9807e-02,  1.8459e-02, -2.4199e-17, -2.8480e-16,
             -3.3721e-01, -8.9809e-02, -1.4324e-16, -2.3320e-16, -2.6505e-01,
              2.1983e-01,  1.7250e-01, -6.4576e-17,  1.0087e-16,  8.4827e-01,
              6.7474e-02, -8.1359e-17,  1.2660e-16],
            [-2.0496e-02, -1.5893e-03, -1.3819e-01, -8.3485e-17,  1.7399e-16,
              8.9809e-02, -9.7167e-02, -1.2175e-16,  2.4117e-16,  1.0055e-02,
             -3.8132e-03,  1.3472e-01,  9.2303e-17,  3.4536e-18,  6.7474e-02,
              9.8213e-02,  8.9849e-17, -3.0424e-17],
            [ 2.3776e-18, -2.0812e-17,  1.9477e-18, -9.9544e-02, -7.3520e-10,
              8.2853e-17,  3.3540e-17, -1.5276e-01, -5.7119e-09,  1.6638e-17,
              6.9816e-18, -4.0526e-17,  3.6552e-01,  6.4945e-10, -8.1359e-17,
              8.9849e-17,  3.8096e-01,  1.0167e-08],
            [ 4.5883e-18,  1.2729e-18,  1.8886e-17, -7.3520e-10, -9.9544e-02,
             -7.0800e-17, -1.1837e-16, -5.7119e-09, -1.5276e-01,  5.6828e-17,
             -5.1567e-17, -4.8017e-17,  6.4945e-10,  3.6552e-01,  1.2660e-16,
             -3.0424e-17,  1.0167e-08,  3.8096e-01]], dtype=dtype),
             d=torch.tensor([[ 1.0281e+00,  3.2549e-02,  2.3654e-02, -5.7526e-19,  9.1280e-17,
             -2.4147e-01,  6.2820e-03, -1.0733e-17,  2.0113e-17,  3.1737e-03,
             -6.8464e-03,  2.9980e-02,  1.2937e-17, -2.6240e-17,  4.7523e-02,
             -4.4086e-03,  1.7836e-17, -3.5684e-17],
            [ 3.2549e-02,  6.9425e-02, -3.1931e-03, -9.1140e-18, -3.3428e-17,
              1.8435e-01, -1.1237e-02, -5.1470e-18, -9.7425e-18, -6.8464e-03,
              4.9859e-03, -4.3996e-02, -2.5507e-17,  3.4006e-17, -2.9451e-02,
             -1.9704e-02, -3.0373e-17,  4.3601e-17],
            [ 2.3654e-02, -3.1931e-03,  2.1154e-01, -2.8076e-17,  5.9914e-17,
             -1.7139e-01,  1.4970e-01, -7.6060e-18, -6.7898e-17, -2.9980e-02,
              4.3996e-02, -1.7019e-01,  2.7046e-17, -1.3006e-16,  1.9768e-02,
             -1.2533e-01,  4.9028e-17, -1.6282e-16],
            [-5.7526e-19, -9.1140e-18, -2.8076e-17,  1.1592e-01, -3.9689e-09,
              9.2444e-17, -5.8931e-18,  1.2946e-01, -6.9761e-10,  6.7673e-17,
             -2.1706e-17, -7.3792e-18,  1.1592e-01, -3.9689e-09, -8.5672e-17,
              5.6604e-17,  1.2946e-01, -6.9761e-10],
            [ 9.1280e-17, -3.3428e-17,  5.9914e-17, -3.9689e-09,  1.1592e-01,
             -3.1427e-16,  1.3565e-16, -6.9761e-10,  1.2946e-01, -3.9180e-17,
              1.7268e-16,  3.7563e-17, -3.9689e-09,  1.1592e-01,  3.6545e-16,
             -5.4123e-17, -6.9761e-10,  1.2946e-01],
            [-2.4147e-01,  1.8435e-01, -1.7139e-01,  9.2444e-17, -3.1427e-16,
              7.1511e-01, -1.3883e-01,  1.0021e-16, -1.2954e-16,  4.7523e-02,
             -2.9451e-02, -1.9768e-02, -5.4095e-18,  9.0430e-17, -1.7027e-01,
              2.7448e-02, -2.5620e-17,  1.3500e-16],
            [ 6.2820e-03, -1.1237e-02,  1.4970e-01, -5.8931e-18,  1.3565e-16,
             -1.3883e-01,  1.1011e-01,  8.2257e-18,  7.0123e-17,  4.4086e-03,
              1.9704e-02, -1.2533e-01,  2.9558e-17,  2.6845e-17, -2.7448e-02,
             -9.4819e-02,  4.6912e-17,  1.9703e-17],
            [-1.0733e-17, -5.1470e-18, -7.6060e-18,  1.2946e-01, -6.9761e-10,
              1.0021e-16,  8.2257e-18,  1.4458e-01,  3.3921e-09,  1.1672e-17,
             -2.1207e-17, -2.9119e-17,  1.2946e-01, -6.9761e-10, -8.3287e-17,
              4.8543e-17,  1.4458e-01,  3.3921e-09],
            [ 2.0113e-17, -9.7425e-18, -6.7898e-17, -6.9761e-10,  1.2946e-01,
             -1.2954e-16,  7.0123e-17,  3.3921e-09,  1.4458e-01, -3.0696e-17,
              8.4297e-17,  6.2386e-17, -6.9761e-10,  1.2946e-01,  1.0318e-16,
             -4.8138e-17,  3.3921e-09,  1.4458e-01],
            [ 3.1737e-03, -6.8464e-03, -2.9980e-02,  6.7673e-17, -3.9180e-17,
              4.7523e-02,  4.4086e-03,  1.1672e-17, -3.0696e-17,  1.0281e+00,
              3.2549e-02, -2.3654e-02, -1.2196e-17,  5.6924e-17, -2.4147e-01,
             -6.2820e-03, -1.5404e-17,  7.1001e-17],
            [-6.8464e-03,  4.9859e-03,  4.3996e-02, -2.1706e-17,  1.7268e-16,
             -2.9451e-02,  1.9704e-02, -2.1207e-17,  8.4297e-17,  3.2549e-02,
              6.9425e-02,  3.1931e-03,  3.6940e-18, -1.1743e-17,  1.8435e-01,
              1.1237e-02,  6.0848e-18, -2.9222e-17],
            [ 2.9980e-02, -4.3996e-02, -1.7019e-01, -7.3792e-18,  3.7563e-17,
             -1.9768e-02, -1.2533e-01, -2.9119e-17,  6.2386e-17, -2.3654e-02,
              3.1931e-03,  2.1154e-01, -2.2094e-17,  1.2803e-17,  1.7139e-01,
              1.4970e-01, -4.0062e-17,  1.3720e-17],
            [ 1.2937e-17, -2.5507e-17,  2.7046e-17,  1.1592e-01, -3.9689e-09,
             -5.4095e-18,  2.9558e-17,  1.2946e-01, -6.9761e-10, -1.2196e-17,
              3.6940e-18, -2.2094e-17,  1.1592e-01, -3.9689e-09,  5.0151e-18,
              4.3691e-17,  1.2946e-01, -6.9761e-10],
            [-2.6240e-17,  3.4006e-17, -1.3006e-16, -3.9689e-09,  1.1592e-01,
              9.0430e-17,  2.6845e-17, -6.9761e-10,  1.2946e-01,  5.6924e-17,
             -1.1743e-17,  1.2803e-17, -3.9689e-09,  1.1592e-01, -2.1963e-16,
             -7.3881e-17, -6.9761e-10,  1.2946e-01],
            [ 4.7523e-02, -2.9451e-02,  1.9768e-02, -8.5672e-17,  3.6545e-16,
             -1.7027e-01, -2.7448e-02, -8.3287e-17,  1.0318e-16, -2.4147e-01,
              1.8435e-01,  1.7139e-01,  5.0151e-18, -2.1963e-16,  7.1511e-01,
              1.3883e-01,  1.9370e-18, -2.9474e-16],
            [-4.4086e-03, -1.9704e-02, -1.2533e-01,  5.6604e-17, -5.4123e-17,
              2.7448e-02, -9.4819e-02,  4.8543e-17, -4.8138e-17, -6.2820e-03,
              1.1237e-02,  1.4970e-01,  4.3691e-17, -7.3881e-17,  1.3883e-01,
              1.1011e-01,  3.6897e-17, -8.3231e-17],
            [ 1.7836e-17, -3.0373e-17,  4.9028e-17,  1.2946e-01, -6.9761e-10,
             -2.5620e-17,  4.6912e-17,  1.4458e-01,  3.3921e-09, -1.5404e-17,
              6.0848e-18, -4.0062e-17,  1.2946e-01, -6.9761e-10,  1.9370e-18,
              3.6897e-17,  1.4458e-01,  3.3921e-09],
            [-3.5684e-17,  4.3601e-17, -1.6282e-16, -6.9761e-10,  1.2946e-01,
              1.3500e-16,  1.9703e-17,  3.3921e-09,  1.4458e-01,  7.1001e-17,
             -2.9222e-17,  1.3720e-17, -6.9761e-10,  1.2946e-01, -2.9474e-16,
             -8.3231e-17,  3.3921e-09,  1.4458e-01]], dtype=dtype))
    qc = HF(mol).run(dm0=dm1)
    ene = qc.energy()
    assert is_orb_min(qc)

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
