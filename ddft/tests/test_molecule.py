import time
import pytest
import numpy as np
import matplotlib.pyplot as plt
import torch
import xitorch as xt
from torch.autograd import gradcheck, gradgradcheck
from ddft.eks.base_eks import BaseLDA, BaseEKS
from ddft.utils.safeops import safepow
from ddft.systems import mol
from ddft.qccalcs import dft

"""
Test various configurations using the molecule API.
"""

dtype = torch.float64
device = torch.device("cpu")

class PseudoLDA(BaseLDA):
# class PseudoLDA(BaseEKS):
    def __init__(self, a, p):
        super(PseudoLDA, self).__init__()
        self.a = a
        self.p = p

    def energy_unpol(self, rho):
        return self.a * safepow(rho.abs(), self.p)

    def potential_unpol(self, rho):
        return self.a * self.p * safepow(rho.abs(), self.p - 1)

    def getfwdparamnames(self, prefix=""):
        return [prefix+"a", prefix+"p"]

@pytest.mark.parametrize(
    "eks_model,system_str,energy_true",
    [
        ("lda,", "Be 0 0 0", -14.2219 ), # pyscf: -14.2207
        ("lda,", "Ne 0 0 0", -127.4718), # pyscf: -127.4690
        ("pbe,", "Be 0 0 0", -14.5432 ), # pyscf: -14.5422
        ("pbe,", "Ne 0 0 0", -128.5023), # pyscf: -128.4996
    ]
)
def test_atom(eks_model, system_str, energy_true):
    basis = "6-311++G**"
    runtest_molsystem_energy(system_str, basis, eks_model, energy_true)

@pytest.mark.parametrize(
    "eks_model,system_str,energy_true",
    [
        ("lda,", "H -0.5 0 0; H 0.5 0 0", -0.9791401),  # pyscf: -0.979143262
        ("lda,", "Li -2.5 0 0; Li 2.5 0 0", -14.393459),  # pyscf: -14.3927863482007
        ("lda,", "N -1 0 0; N 1 0 0", -107.7327),  # pyscf: -107.726124017789
        ("lda,", "F -1.25 0 0; F 1.25 0 0", -197.0101),  # pyscf: -197.005308558326
        ("lda,", "C -1 0 0; O 1 0 0", -111.49737),  # pyscf: -111.490687028797
    ]
)
def test_mol(eks_model, system_str, energy_true):
    basis = "6-311++G**"
    runtest_molsystem_energy(system_str, basis, eks_model, energy_true)

def test_atom_grad():
    isystem = 0
    systems = [ # atomzs
        [2], [4],
    ]
    basis = "6-31G"
    a = torch.tensor(-0.7385587663820223, dtype=dtype, device=device).requires_grad_()
    p = torch.tensor(4./3, dtype=dtype, device=device).requires_grad_()
    systemargs = ()
    def get_system():
        atomzs = systems[isystem]
        atomposs = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        system = (atomzs, atomposs)
        return mol(system, basis, requires_grad=True)

    runtest_molsystem_grad(a, p, get_system, systemargs)

@pytest.mark.parametrize(
    "atomzs,dist,gradx,grad2x",
    [((1, 1), 1.5, 6.967e-3, 2.872e-1),
     ((3, 3), 5.0, -6.292e-3, 1.900e-2),
     ((7, 7), 2.0, -2.497e-1, 2.261e0),
     ((9, 9), 3.0, 5.531e-2, 1.730e-1),
     ((6, 8), 2.0, -3.246e-1, 2.176e0)]
)
def test_mol_grad_all(atomzs, dist, gradx, grad2x):
    run_test_mol_grad(atomzs, dist, gradx, grad2x)

def run_test_mol_grad(atomzs, dist, gradx=None, grad2x=None):
    basis = "6-31G"

    a = torch.tensor(-0.7385587663820223, dtype=dtype, device=device).requires_grad_()
    p = torch.tensor(4./3, dtype=dtype, device=device).requires_grad_()
    fwd_options = {
        "method": "broyden1",
        "f_tol": 1e-9
    }
    bck_options = {
    }

    dist = torch.tensor(dist, dtype=dtype, device=device).requires_grad_()
    atomposs = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0]], dtype=dtype, device=device) * dist
    tinit = time.time()
    m = mol((atomzs, atomposs), basis, requires_grad=True)
    eks_model = PseudoLDA(a, p)
    scf = dft(m, eks_model=eks_model, fwd_options=fwd_options, bck_options=bck_options)
    energy = scf.energy()

    # only take grad in x to save time
    t0 = time.time()
    if gradx is None:
        print("Forward   : %.3fs" % (t0 - tinit))

    dedx = torch.autograd.grad(energy, dist, create_graph=True)[0]
    t1 = time.time()

    if gradx is not None:
        gradx = dedx * 0 + gradx
        assert torch.allclose(dedx, gradx, rtol=1e-2, atol=1e-5)
    else:
        print("Backward  : %.3fs" % (t1 - t0))

    t2 = time.time()
    d2edx2 = torch.autograd.grad(dedx, dist, create_graph=True)[0]
    t3 = time.time()

    if grad2x is not None:
        grad2x = d2edx2 * 0 + grad2x
        assert torch.allclose(d2edx2, grad2x, rtol=1e-2, atol=1e-5)
    else:
        print("2 backward: %.3fs" % (t3 - t2))

    print("Energy: ", energy.item())
    print("dEdx  : ", dedx.item())
    print("d2Edx2: ", d2edx2.item())

def runtest_molsystem_grad(a, p, get_system, systemargs,
        profiling=False,
        fwd_options=None,
        bck_options=None):
    def get_energy(a, p, *systemargs):
        m = get_system(*systemargs)
        eks_model = PseudoLDA(a, p)
        scf = dft(m, eks_model=eks_model, fwd_options=fwd_options, bck_options=bck_options)
        energy = scf.energy()
        return energy

    if not profiling:
        gradcheck(get_energy, (a, p, *systemargs))
        gradgradcheck(get_energy, (a, p, *systemargs), eps=1e-4, rtol=1e-2)
    else:
        import time
        t0 = time.time()
        energy = get_energy(a, p, *systemargs)
        ge = torch.ones_like(energy).requires_grad_()
        t1 = time.time()
        print("Forward   : %fs" % (t1-t0))
        x = systemargs[0]
        dedx, = torch.autograd.grad(energy, (x,), grad_outputs=ge, create_graph=True)
        t2 = time.time()
        print("Backward  : %fs" % (t2-t1))

        dedxx, = torch.autograd.grad(dedx, (x,), create_graph=True)
        t3 = time.time()
        print("2 backward: %fs" % (t3-t2))
        print(dedxx)

def runtest_molsystem_energy(system_str, basis, eks_model, energy_true=None):
    m = mol(system_str, basis)
    scf = dft(m, eks_model=eks_model)
    energy = scf.energy()
    print("Energy: %.7f" % energy)
    if energy_true is not None:
        assert torch.allclose(energy, torch.tensor(energy_true, dtype=energy.dtype))
    else:
        return energy

# functions used to get the true values of grad x of molecules
def calc_molsystem_energy_grad(atomzs, dist_central, eps=1e-2):
    basis = "6-31G"

    a = torch.tensor(-0.7385587663820223, dtype=dtype, device=device).requires_grad_()
    p = torch.tensor(4./3, dtype=dtype, device=device).requires_grad_()
    fwd_options = {
        "method": "broyden1",
        "f_tol": 1e-9
    }
    bck_options = {
    }

    dists = [dist_central - eps, dist_central, dist_central + eps]
    energies = []
    for d in dists:
        dist = torch.tensor(d, dtype=dtype, device=device).requires_grad_()
        atomposs = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0]], dtype=dtype, device=device) * dist
        m = mol((atomzs, atomposs), basis, requires_grad=True)
        eks_model = PseudoLDA(a, p)
        scf = dft(m, eks_model=eks_model, fwd_options=fwd_options, bck_options=bck_options)
        energies.append(scf.energy().item())

    dedx = (energies[-1] - energies[0]) / (2 * eps)
    d2edx2 = (energies[0] + energies[2] - 2 * energies[1]) / (eps * eps)
    print("dedx: ", dedx)
    print("d2edx2: ", d2edx2)

if __name__ == "__main__":
    # run_test_mol_grad((1., 1.), 1.5)
    run_test_mol_grad((6., 8.), 2.0)
    # runtest_molsystem_energy("Be 0 0 0", "6-311++G**", "lda,")
    # run_test_mol_grad((9., 9.), 3.0)
    # calc_molsystem_energy_grad([6., 8.], 2.0)

    # # uncomment to use pprofile
    # import pprofile
    # prof = pprofile.Profile()
    # with prof():
    #     runtest_molsystem_energy("Be 0 0 0", "6-311++G**", "lda,")
    # prof.print_stats()

    # (1, 1), 1.5: 6.967e-3, 2.872e-1
    # (3, 3), 5.0: -6.292e-3, 1.900e-2
    # [7, 7], 2.0: -2.497e-1, 2.261e0
    # [9, 9], 3.0: 5.531e-2, 1.730e-1
    # [6, 8], 2.0: -3.246e-1, 2.176e0
