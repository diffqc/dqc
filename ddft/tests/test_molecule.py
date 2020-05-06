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
        if distance is None:
            distance = 1.0
        atomzs = torch.tensor([1.0, 1.0], dtype=dtype)
        atomposs = distance * torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
        atomposs = atomposs.requires_grad_()
        energy = torch.tensor(-0.976186, dtype=dtype) # only works for LDA
    elif molname == "Li2":
        if distance is None:
            distance = 3.0
        atomzs = torch.tensor([3.0, 3.0], dtype=dtype)
        atomposs = distance * torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
        atomposs = atomposs.requires_grad_()
        energy = torch.tensor(-14.1116, dtype=dtype) # only works for LDA
    else:
        raise RuntimeError("Unknown molecule %s" % molname)
    if not with_energy:
        return atomzs, atomposs
    else:
        return atomzs, atomposs, energy

def test_mol():
    molnames = ["H2", "Li2"]
    for molname in molnames:
        atomzs, atomposs, energy_true = get_molecule(molname, with_energy=True)
        a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
        p = torch.tensor([4./3]).to(dtype).requires_grad_()
        eks_model = PseudoLDA(a, p)
        energy, _ = molecule(atomzs, atomposs, eks_model=eks_model)
        assert torch.allclose(energy, energy_true)

def test_mol_grad():
    # setup the molecule's atoms positions
    atomzs, atomposs = get_molecule("H2")

    # pseudo-lda eks model
    a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
    p = torch.tensor([4./3]).to(dtype).requires_grad_()

    def get_energy(a, p, atomzs, atomposs, output="energy"):
        eks_model = PseudoLDA(a, p)
        energy, density = molecule(atomzs, atomposs, eks_model=eks_model)
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
        "H2": torch.tensor(
            ([0.5, 0.75, 1.0, 1.25] if plot else []) +
            [1.475, 1.4775, 1.48, 1.481, 1.4825, 1.485] +
            ([1.5, 1.75, 2.0, 2.5] if plot else []),
            dtype=dtype),
        "Li2": torch.tensor(
            ([1.0, 1.5, 2.0, 2.2, 2.5, 2.7, 3.0] if plot else []) +
            [3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18, 3.19, 3.2, 3.21] +
            ([3.25, 3.5, 4.0] if plot else []),
            dtype=dtype)
    }
    for molname,dists in all_dists.items():
        runtest_vibration(molname, dists, plot=plot)

def runtest_vibration(molname, dists, plot=False):
    def get_energy(a, p, dist):
        bck_options = {
            "max_niter": 100,
        }
        atomzs, atomposs = get_molecule(molname, distance=dist)
        eks_model = PseudoLDA(a, p)
        energy, density = molecule(atomzs, atomposs, eks_model=eks_model,
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
