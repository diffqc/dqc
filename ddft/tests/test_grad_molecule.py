import torch
from torch.autograd import gradcheck, gradgradcheck
from ddft.api.molecule import molecule
from ddft.eks import BaseEKS
from ddft.utils.safeops import safepow

def test_mol_grad():
    dtype = torch.float64
    class PseudoLDA(BaseEKS):
        def __init__(self, a, p):
            super(PseudoLDA, self).__init__()
            self.a = a
            self.p = p

        def forward(self, density):
            return self.a * safepow(density.abs(), self.p)

    # setup the molecule's atoms positions
    atomzs = torch.tensor([1.0, 1.0], dtype=dtype)
    distance = torch.tensor([1.0], dtype=dtype).requires_grad_()

    # pseudo-lda eks model
    a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
    p = torch.tensor([4./3]).to(dtype).requires_grad_()
    eks_model = PseudoLDA(a, p)

    def get_energy(a, p, distance, output="energy"):
        atompos = distance * torch.tensor([[-0.5], [0.5]], dtype=dtype) # (2,1)
        atompos = torch.cat((atompos, torch.zeros((2,2), dtype=dtype)), dim=-1)
        eks_model = PseudoLDA(a, p)
        energy, density = molecule(atomzs, atompos, eks_model=eks_model)
        if output == "energy":
            return energy
        elif output == "density":
            return density.abs().sum()

    gradcheck(get_energy, (a, p, distance, "energy"))
    gradcheck(get_energy, (a, p, distance, "density"))
    # gradgradcheck(get_energy, (a, p, distance, "energy"))
    # gradgradcheck(get_energy, (a, p, distance, False))
