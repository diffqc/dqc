import numpy as np
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

def get_molecule(molname, distance=1.0, with_energy=False):
    if molname == "H2":
        atomzs = torch.tensor([1.0, 1.0], dtype=dtype)
        atomposs = distance * torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
        atomposs = atomposs.requires_grad_()
        energy = torch.tensor(-0.976186, dtype=dtype) # only works for LDA
    else:
        raise RuntimeError("Unknown molecule %s" % molname)
    if not with_energy:
        return atomzs, atomposs
    else:
        return atomzs, atomposs, energy

def test_mol():
    atomzs, atomposs, energy_true = get_molecule("H2", with_energy=True)
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

def test_h2_vibration():
    def get_energy(a, p, dist):
        bck_options = {
            "max_niter": 100,
        }
        atomzs, atomposs = get_molecule("H2", distance=dist)
        eks_model = PseudoLDA(a, p)
        energy, density = molecule(atomzs, atomposs, eks_model=eks_model,
            bck_options=bck_options)
        return energy

    # pseudo-lda eks model
    a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
    p = torch.tensor([4./3]).to(dtype).requires_grad_()

    dists = torch.tensor(
        [1.475, 1.4775, 1.48, 1.481, 1.4825, 1.485],
        dtype=dtype)

    energies = [float(get_energy(a, p, dist).view(-1)[0]) for dist in dists]

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
    num_k = np.polyfit(dists_np, energies_np, 2)[0] * 2

    # get the vibration frequency
    min_dist = dists[min_idx].requires_grad_()
    min_energy = get_energy(a, p, min_dist)
    deds = torch.autograd.grad(min_energy, min_dist, create_graph=True)
    ana_k = torch.autograd.grad(deds, min_dist)

    print("Numerical: %.8e" % num_k)
    print("Analytical: %.8e" % ana_k)

    assert np.abs(num_k-ana_k)/num_k < 1e-2

if __name__ == "__main__":
    test_h2_vibration()
