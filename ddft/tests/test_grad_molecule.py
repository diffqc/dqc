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

    def get_energy(a, p, distance, eks_model=None):
        atompos = distance * torch.tensor([[-0.5], [0.5]], dtype=dtype) # (2,1)
        atompos = torch.cat((atompos, torch.zeros((2,2), dtype=dtype)), dim=-1)
        if eks_model is None:
            eks_model = PseudoLDA(a, p)
        energy, _ = molecule(atomzs, atompos, eks_model=eks_model)
        return energy

    gradcheck(get_energy, (a, p, distance))
    # gradgradcheck(get_energy, (a, p, distance))
