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

def get_molecule(molname, with_energy=False):
    if molname == "H2":
        atomzs = torch.tensor([1.0, 1.0], dtype=dtype)
        atomposs = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype).requires_grad_()
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

    gradcheck(get_energy, (a, p, atomzs, atomposs, "energy"), rtol=3e-3)
    gradcheck(get_energy, (a, p, atomzs, atomposs, "density"))
    # gradgradcheck(get_energy, (a, p, atomzs, atomposs, "energy"))
    # gradgradcheck(get_energy, (a, p, atomzs, atomposs, False))
