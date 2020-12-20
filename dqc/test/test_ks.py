from itertools import product
import numpy as np
import torch
import pytest
from dqc.qccalc.ks import KS
from dqc.system.mol import Mol
from dqc.xc.base_xc import BaseXC
from dqc.utils.safeops import safepow, safenorm
from dqc.utils.datastruct import ValGrad

dtype = torch.float64

atomzs_poss = [
    ([1, 1], 1.0),
    ([3, 3], 5.0),
    ([7, 7], 2.0),
    ([9, 9], 2.5),
    ([6, 8], 2.0),
]
energies = [
    -0.979143260,  # pyscf: -0.979143262
    -14.3927863482007,  # pyscf: -14.3927863482007
    -107.726124017789,  # pyscf: -107.726124017789
    -197.005308558326,  # pyscf: -197.005308558326
    -111.490687028797,  # pyscf: -111.490687028797
]

@pytest.mark.parametrize(
    "xc,atomzs,dist,energy_true",
    [("lda,", *atomz_pos, energy) for (atomz_pos, energy) in zip(atomzs_poss, energies)]
)
def test_rks_energy(xc, atomzs, dist, energy_true):
    poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
    mol = Mol((atomzs, poss), basis="6-311++G**", dtype=dtype)
    qc = KS(mol, xc=xc, restricted=True).run()
    ene = qc.energy()
    assert torch.allclose(ene, ene * 0 + energy_true)

@pytest.mark.parametrize(
    "xc,atomzs,dist,grad2",
    [("lda,", *atomz_pos, grad2) for (atomz_pos, grad2) in product(atomzs_poss, [False, True])]
)
def test_rks_grad_pos(xc, atomzs, dist, grad2):
    # test grad of energy w.r.t. atom's position

    torch.manual_seed(123)
    # set stringent requirement for grad2
    bck_options = None if not grad2 else {
        "rtol": 1e-9,
        "atol": 1e-9,
    }

    def get_energy(dist_tensor):
        poss_tensor = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist_tensor
        mol = Mol((atomzs, poss_tensor), basis="3-21G", dtype=dtype, grid=3)
        qc = KS(mol, xc=xc, restricted=True).run(bck_options=bck_options)
        return qc.energy()
    dist_tensor = torch.tensor(dist, dtype=dtype, requires_grad=True)
    if grad2:
        torch.autograd.gradgradcheck(get_energy, (dist_tensor,),
                                     rtol=1e-2, atol=1e-5)
    else:
        torch.autograd.gradcheck(get_energy, (dist_tensor,))

@pytest.mark.parametrize(
    "xc,atomzs,dist,vext_p",
    [("lda,", *atomz_pos, 0.1) for atomz_pos in atomzs_poss]
)
def test_rks_grad_vext(xc, atomzs, dist, vext_p):
    # check if the gradient w.r.t. vext is obtained correctly (only check 1st
    # grad because we don't need 2nd grad at the moment)

    poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
    mol = Mol((atomzs, poss), basis="3-21G", dtype=dtype, grid=3)
    mol.setup_grid()
    rgrid = mol.get_grid().get_rgrid()  # (ngrid, ndim)
    rgrid_norm = torch.norm(rgrid, dim=-1)  # (ngrid,)

    def get_energy(vext_params):
        vext = rgrid_norm * rgrid_norm * vext_params  # (ngrid,)
        qc = KS(mol, xc=xc, vext=vext, restricted=True).run()
        ene = qc.energy()
        return ene

    vext_params = torch.tensor(vext_p, dtype=dtype).requires_grad_()
    torch.autograd.gradcheck(get_energy, (vext_params,))

class PseudoLDA(BaseXC):
    def __init__(self, a, p):
        self.a = a
        self.p = p

    @property
    def family(self):
        return 1

    def get_edensityxc(self, densinfo):
        if isinstance(densinfo, ValGrad):
            rho = densinfo.value.abs()  # safeguarding from nan
            return self.a * safepow(rho, self.p) ** self.p
        else:
            return 0.5 * (self.get_edensityxc(densinfo.u * 2) + self.get_edensityxc(densinfo.d * 2))

    def getparamnames(self, methodname, prefix=""):
        return [prefix + "a", prefix + "p"]

class PseudoPBE(BaseXC):
    def __init__(self, kappa, mu):
        self.kappa = kappa
        self.mu = mu

    @property
    def family(self):
        return 2  # GGA

    def get_edensityxc(self, densinfo):
        if isinstance(densinfo, ValGrad):
            rho = densinfo.value.abs()
            kf_rho = (3 * np.pi * np.pi) ** (1.0 / 3) * safepow(rho, 4.0 / 3)
            e_unif = -3.0 / (4 * np.pi) * kf_rho
            norm_grad = safenorm(densinfo.grad, dim=-1)
            s = norm_grad / (2 * kf_rho)
            fx = 1 + self.kappa - self.kappa / (1 + self.mu * s * s / self.kappa)
            return fx * e_unif
        else:
            return 0.5 * (self.get_edensityxc(densinfo.u * 2) + self.get_edensityxc(densinfo.d * 2))

    def getparamnames(self, methodname, prefix=""):
        return [prefix + "kappa", prefix + "mu"]

@pytest.mark.parametrize(
    "xccls,xcparams,atomzs,dist",
    [
        (PseudoLDA, (-0.7385587663820223, 4. / 3), *atomzs_poss[0]),
        (PseudoPBE, (0.804, 0.21951), *atomzs_poss[0]),
    ]
)
def test_rks_grad_vxc(xccls, xcparams, atomzs, dist):
    # check if the gradients w.r.t. vxc parameters are obtained correctly
    poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
    mol = Mol((atomzs, poss), basis="3-21G", dtype=dtype, grid=3)
    mol.setup_grid()

    def get_energy(*params):
        xc = xccls(*params)
        qc = KS(mol, xc=xc, restricted=True).run()
        ene = qc.energy()
        return ene

    params = tuple(torch.tensor(p, dtype=dtype).requires_grad_() for p in xcparams)
    torch.autograd.gradcheck(get_energy, params)

############### Unrestricted Kohn-Sham ###############
u_atomzs_spins = [
    # atomz, spin
    (1, 1),
    (3, 1),
    (5, 1),
    (8, 2),
]
u_atom_energies = [
    # numbers from pyscf with basis 6-311++G** with grid level 4 and LDA x
    -0.456918307830999,  # H
    -7.19137615551071,  # Li
    -24.0638478157822,  # B
    -73.987463670134,  # O
]
u_mols_dists_spins = [
    # atomzs,dist,spin
    ([8, 8], 2.0, 2),
]
u_mols_energies = [
    # numbers from pyscf with basis 6-311++G** with grid level 3 and LDA x
    -148.149998931489,  # O2
]

@pytest.mark.parametrize(
    "xc,atomzs,dist,energy_true",
    [("lda,", *atomz_pos, energy) for (atomz_pos, energy) in zip(atomzs_poss[:2], energies[:2])]
)
def test_uks_energy_same_as_rks(xc, atomzs, dist, energy_true):
    # test to see if uks energy gets the same energy as rks for non-polarized systems
    poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
    mol = Mol((atomzs, poss), basis="6-311++G**", dtype=dtype)
    qc = KS(mol, xc=xc, restricted=False).run()
    ene = qc.energy()
    assert torch.allclose(ene, ene * 0 + energy_true)

@pytest.mark.parametrize(
    "xc,atomz,spin,energy_true",
    [("lda,", atomz, spin, energy) for ((atomz, spin), energy) in zip(u_atomzs_spins, u_atom_energies)]
)
def test_uks_energy_atoms(xc, atomz, spin, energy_true):
    # check the energy of atoms with non-0 spins
    poss = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
    mol = Mol(([atomz], poss), basis="6-311++G**", grid=4, dtype=dtype, spin=spin)
    qc = KS(mol, xc=xc, restricted=False).run()
    ene = qc.energy()
    assert torch.allclose(ene, ene * 0 + energy_true, rtol=0.0, atol=1e-6)

@pytest.mark.parametrize(
    "xc,atomzs,dist,spin,energy_true",
    [("lda,", atomzs, dist, spin, energy) for ((atomzs, dist, spin), energy)
        in zip(u_mols_dists_spins, u_mols_energies)]
)
def test_uks_energy_mols(xc, atomzs, dist, spin, energy_true):
    # check the energy of molecules with non-0 spins
    poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
    mol = Mol((atomzs, poss), basis="6-311++G**", grid=3, dtype=dtype, spin=spin)
    qc = KS(mol, xc=xc, restricted=False).run()
    ene = qc.energy()
    assert torch.allclose(ene, ene * 0 + energy_true)

@pytest.mark.parametrize(
    "xccls,xcparams,atomz",
    [
        (PseudoLDA, (-0.7385587663820223, 4. / 3), u_atomzs_spins[0][0]),
        (PseudoPBE, (0.804, 0.21951), u_atomzs_spins[0][0]),
    ]
)
def test_uks_grad_vxc(xccls, xcparams, atomz):
    # check if the gradients w.r.t. vxc parameters are obtained correctly
    poss = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
    mol = Mol(([atomz], poss), basis="3-21G", dtype=dtype, grid=3)
    mol.setup_grid()

    def get_energy(*params):
        xc = xccls(*params)
        qc = KS(mol, xc=xc, restricted=False).run()
        ene = qc.energy()
        return ene

    params = tuple(torch.tensor(p, dtype=dtype).requires_grad_() for p in xcparams)
    torch.autograd.gradcheck(get_energy, params)

if __name__ == "__main__":
    import time
    xc = "lda,"
    basis = "3-21G"  # "6-311++G**"
    poss = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]], dtype=dtype).requires_grad_()
    moldesc = ([6, 8], poss)
    mol = Mol(moldesc, basis=basis, dtype=dtype, grid=3)
    # mol = Mol("Li -2.5 0 0; Li 2.5 0 0", basis="6-311++G**", dtype=dtype)
    # mol = Mol("H -0.5 0 0; H 0.5 0 0", basis=basis, dtype=dtype)
    profiler = 0
    with torch.autograd.profiler.profile(enabled=profiler, with_stack=True, record_shapes=True) as prof, \
            torch.autograd.detect_anomaly():
        t0 = time.time()
        qc = KS(mol, xc=xc, restricted=True).run()
        ene = qc.energy()
        t1 = time.time()
        print(ene)
        print("Forward time : %fs" % (t1 - t0))

        dedposs, = torch.autograd.grad(ene, poss, create_graph=True)
        t2 = time.time()
        print(dedposs)
        print("Backward time: %fs" % (t2 - t1))
        z = dedposs[-1, -1]

        d2edposs2 = torch.autograd.grad(z, poss)
        print(d2edposs2)
        t3 = time.time()
        print("2nd backward time: %fs" % (t3 - t1))

    if profiler:
        # prof.export_chrome_trace("trace")
        print(prof
              .key_averages(group_by_input_shape=True)
              .table(sort_by="self_cpu_time_total", row_limit=200)
              # .table(sort_by="cpu_memory_usage", row_limit=200)
              )