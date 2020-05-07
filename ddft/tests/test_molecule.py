import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import gradcheck, gradgradcheck
from ddft.api.molecule import molecule
from ddft.eks import BaseEKS
from ddft.utils.safeops import safepow

dtype = torch.float64
class PseudoLDA(BaseEKS):
    def __init__(self, a, p):
        super(PseudoLDA, self).__init__()
        self.a = a
        self.p = p

    def forward(self, density):
        return self.a * safepow(density.abs(), self.p)

    def getfwdparams(self):
        return [self.a, self.p]

    def setfwdparams(self, *params):
        self.a, self.p = params[:2]
        return 2

def get_molecule(molname, distance=None, with_energy=False):
    if molname == "H2":
        def_distance = 1.0
        atomz = 1.0
        energy = -0.976186 # only works for LDA and 6-311++G** basis
    elif molname == "Li2":
        def_distance = 5.0
        atomz = 3.0
        energy = -14.0419 # only works for LDA and 6-311++G** basis
    elif molname == "N2":
        def_distance = 2.0
        atomz = 7.0
        energy = -107.5768 # only works for LDA and cc-pvdz
    elif molname == "CO":
        def_distance = 2.0
        atomz = torch.tensor([6.0, 8.0], dtype=dtype)
        energy = -111.3264 # only works for LDA and cc-pvdz
    elif molname == "F2":
        def_distance = 2.5
        atomz = 9.0
        energy = -196.7553 # only works for LDA and cc-pvdz
    else:
        raise RuntimeError("Unknown molecule %s" % molname)

    # setup the tensors
    if distance is None:
        distance = def_distance
    energy = torch.tensor(energy, dtype=dtype)
    atomzs = torch.tensor([1.0, 1.0], dtype=dtype) * atomz
    atomposs = distance * torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
    atomposs = atomposs.requires_grad_()

    if not with_energy:
        return atomzs, atomposs
    else:
        return atomzs, atomposs, energy

def test_mol():
    molnames = {
        "H2" : "6-311++G**",
        "Li2": "6-311++G**",
        "N2": "cc-pvdz",
        "F2": "cc-pvdz",
        "CO": "cc-pvdz",
    }
    for molname, basis in molnames.items():
        atomzs, atomposs, energy_true = get_molecule(molname, with_energy=True)
        a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
        p = torch.tensor([4./3]).to(dtype).requires_grad_()
        eks_model = PseudoLDA(a, p)
        energy, _ = molecule(atomzs, atomposs, eks_model=eks_model, basis=basis)
        assert torch.allclose(energy, energy_true)

def test_mol_grad():
    # setup the molecule's atoms positions
    atomzs, atomposs = get_molecule("H2")
    basis = "6-311++G**"

    # pseudo-lda eks model
    a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
    p = torch.tensor([4./3]).to(dtype).requires_grad_()

    def get_energy(a, p, atomzs, atomposs, output="energy"):
        eks_model = PseudoLDA(a, p)
        energy, density = molecule(atomzs, atomposs, eks_model=eks_model,
            basis=basis)
        if output == "energy":
            return energy
        elif output == "density":
            return density.abs().sum()

    gradcheck(get_energy, (a, p, atomzs, atomposs, "energy"))
    gradcheck(get_energy, (a, p, atomzs, atomposs, "density"))
    # gradgradcheck(get_energy, (a, p, atomzs, atomposs, "energy"))
    # gradgradcheck(get_energy, (a, p, atomzs, atomposs, False))

def test_vibration():
    plot = False
    all_dists = {
        "H2": (torch.tensor(
            ([0.5, 0.75, 1.0, 1.25] if plot else []) +
            [1.475, 1.4775, 1.48, 1.481, 1.4825, 1.485] +
            ([1.5, 1.75, 2.0, 2.5] if plot else []),
            dtype=dtype), "6-311++G**"),
        "Li2": (torch.tensor(
            ([2.0, 3.0, 4.0, 4.5, 5.0] if plot else []) +
            [5.13, 5.15, 5.17, 5.2, 5.23, 5.25, 5.27, 5.3, 5.33] +
            ([5.5, 6.0, 7.0] if plot else []),
            dtype=dtype), "cc-pvdz"),
        "N2": (torch.tensor(
            ([1.0, 1.5] if plot else []) +
            [2.05625, 2.0625, 2.06875, 2.075, 2.08125, 2.0875, 2.09375] +
            ([3.0, 4.0] if plot else []),
            dtype=dtype), "cc-pvdz"),
        # "CO": (torch.tensor(
        #     ([1.0, 1.5, 2.0] if plot else []) +
        #     [2.125, 2.15625, 2.1875, 2.21875, 2.25, 2.28125, 2.3125] +
        #     ([2.5, 3.0, 4.0] if plot else []),
        #     dtype=dtype), "cc-pvdz"),
        # "F2": (torch.tensor(
        #     ([1.5, 2.0] if plot else []) +
        #     [2.45, 2.475, 2.5, 2.525, 2.55, 2.575, 2.6] +
        #     ([3.0] if plot else []),
        #     dtype=dtype), "cc-pvdz"),
    }
    for molname,(dists, basis) in all_dists.items():
        runtest_vibration(molname, dists, basis=basis, plot=plot)

def runtest_vibration(molname, dists, basis="cc-pvdz", plot=False):
    def get_energy(a, p, dist):
        bck_options = {
            "max_niter": 100,
        }
        atomzs, atomposs = get_molecule(molname, distance=dist)
        eks_model = PseudoLDA(a, p)
        energy, density = molecule(atomzs, atomposs, basis=basis,
            eks_model=eks_model,
            bck_options=bck_options)
        return energy

    # pseudo-lda eks model
    a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
    p = torch.tensor([4./3]).to(dtype).requires_grad_()

    energies = [float(get_energy(a, p, dist).view(-1)[0]) for dist in dists]
    if plot:
        plt.plot(dists, energies, 'o-')
        plt.show()
        return

    # get the minimum index
    min_idx = 0
    min_ene = energies[0]
    for i in range(1,len(energies)):
        if energies[i] < min_ene:
            min_ene = energies[i]
            min_idx = i

    # fit the square curve
    dists_np = np.asarray(dists)
    energies_np = np.asarray(energies)
    polycoeffs = np.polyfit(dists_np, energies_np, 2)
    num_k = polycoeffs[0] * 2
    dist_opt = -polycoeffs[1] / (2.*polycoeffs[0])

    # get the vibration frequency
    min_dist = dists[min_idx].requires_grad_()
    t0 = time.time()
    min_energy = get_energy(a, p, min_dist)
    t1 = time.time()
    deds = torch.autograd.grad(min_energy, min_dist, create_graph=True)
    t2 = time.time()
    ana_k = torch.autograd.grad(deds, min_dist)
    t3 = time.time()

    print(energies)
    print("Molecule %s" % molname)
    print("Numerical: %.8e" % num_k)
    print("Analytical: %.8e" % ana_k)
    print("Optimum distance: %.3e" % dist_opt)
    print("Running time:")
    print("* Forward: %.3e" % (t1-t0))
    print("* 1st backward: %.3e" % (t2-t1))
    print("* 2nd backward: %.3e" % (t3-t2))
    print("")

    assert np.abs(num_k-ana_k)/num_k < 5e-3

if __name__ == "__main__":
    test_vibration()
