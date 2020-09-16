import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import xitorch as xt
from torch.autograd import gradcheck, gradgradcheck
from ddft.eks import BaseEKS
from ddft.utils.safeops import safepow
from ddft.systems import mol
from ddft.qccalcs import dft

"""
Test various configurations using the molecule API.
"""

dtype = torch.float64
device = torch.device("cpu")

class PseudoLDA(BaseEKS):
    def __init__(self, a, p):
        super(PseudoLDA, self).__init__()
        self.a = a
        self.p = p

    def forward(self, density):
        return self.a * safepow(density.abs(), self.p)

    def getfwdparamnames(self, prefix=""):
        return [prefix+"a", prefix+"p"]

def test_atom2():
    systems = {
        "Be 0 0 0": -14.2219,
        "Ne 0 0 0": -127.4718
    }
    basis = "6-311++G**"
    runtest_molsystem_energy(systems, basis)

def test_mol2():
    systems = {
        "H -0.5 0 0; H 0.5 0 0"  : -0.9791401, # pyscf: -0.979143262 # LDA, 6-311++G**, grid level 4
        "Li -2.5 0 0; Li 2.5 0 0": -14.393459, # pyscf: -14.3927863482007 # LDA, 6-311++G**, grid level 4
        "N -1 0 0; N 1 0 0"      : -107.7327, # pyscf: -107.726124017789 # LDA, 6-311++G**, grid level 4
        "F -1.25 0 0; F 1.25 0 0": -197.0101, # pyscf: -197.005308558326 # LDA, 6-311++G**, grid level 4
        "C -1 0 0; O 1 0 0"      : -111.49737, # pyscf: -111.490687028797 # LDA, 6-311++G**, grid level 4
    }
    basis = "6-311++G**"
    runtest_molsystem_energy(systems, basis)

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

def test_mol_grad():
    basis = "6-31G"
    isystem = 0
    systems = [ # (atomzs, dist)
        ([1,1], 1.0),
        ([3,3], 5.0),
        ([7,7], 2.0),
        ([9,9], 2.5),
        ([6,8], 2.0),
    ]

    dist = torch.tensor(systems[isystem][1], dtype=dtype, device=device).requires_grad_()
    a = torch.tensor(-0.7385587663820223, dtype=dtype, device=device).requires_grad_()
    p = torch.tensor(4./3, dtype=dtype, device=device).requires_grad_()

    systemargs = (dist,)
    def get_system(dist):
        atomzs = systems[isystem][0]
        atomposs = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0]], dtype=dtype, device=device) * dist
        system = (atomzs, atomposs)
        m = mol(system, basis, requires_grad=True)
        return m

    runtest_molsystem_grad(a, p, get_system, systemargs)

def runtest_molsystem_grad(a, p, get_system, systemargs):
    def get_energy(a, p, *systemargs):
        m = get_system(*systemargs)
        eks_model = PseudoLDA(a, p)
        scf = dft(m, eks_model=eks_model)
        energy = scf.energy()
        return energy

    # gradcheck(get_energy, (a, p, *systemargs))
    # gradgradcheck(get_energy, (a, p, *systemargs), eps=1e-4)

    import time
    t0 = time.time()
    energy = get_energy(a, p, *systemargs)
    ge = torch.ones_like(energy).requires_grad_()
    t1 = time.time()
    print("Forward   : %fs" % (t1-t0))
    # x = systemargs[0]
    # dedx, = torch.autograd.grad(energy, (x,), grad_outputs=ge, create_graph=True)
    # t2 = time.time()
    # print("Backward  : %fs" % (t2-t1))
    #
    # dedxx, = torch.autograd.grad(dedx, (x,), create_graph=True)
    # t3 = time.time()
    # print("2 backward: %fs" % (t3-t2))
    # print(dedxx)

def runtest_molsystem_energy(systems, basis):
    for s in systems:
        energy_true = systems[s]
        m = mol(s, basis)
        scf = dft(m, eks_model="lda")
        energy = scf.energy()
        print("%.7f" % energy)
        assert torch.allclose(energy, torch.tensor(energy_true, dtype=energy.dtype))

if __name__ == "__main__":
    # test_atom_grad()
    # with torch.autograd.detect_anomaly():
        test_mol_grad()
