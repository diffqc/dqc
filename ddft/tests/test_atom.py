import torch
from torch.autograd import gradcheck, gradgradcheck
from ddft.api.atom import atom
from ddft.eks import BaseEKS
from ddft.utils.safeops import safepow

dtype = torch.float64

def test_atom():
    energies = {
        1: -0.4063,
        2: -2.7221,
    }

    for atomz, ene in energies.items():
        energy, density = atom(atomz, eks_model="lda",
                 gwmin=1e-5, gwmax=1e3, ng=60,
                 rmin=1e-6, rmax=1e2, nr=200,
                 dtype=dtype)
        assert torch.allclose(energy, torch.tensor([ene], dtype=dtype), atol=1e-4)

def test_radial():
    # test if the radial atom get the same energy as if it is solved radially
    # and non-radially (it should be the same!)
    config = {
        "gwmin": 1e-5,
        "gwmax": 1e2,
        "ng": 20,
        "rmin": 1e-6,
        "rmax": 1e2,
        "nr": 60,
        "dtype": torch.float64
    }
    atomzs_radial = [1,2,3,4,7,10,11,12,15,18]
    for atomz in atomzs_radial:
        energy_radial, density = atom(atomz, is_radial=True, **config)
        energy_nonradial, density = atom(atomz, is_radial=False, **config)
        print(atomz)
        assert torch.allclose(energy_radial, energy_nonradial)

class PseudoLDA(BaseEKS):
    def __init__(self, a, p):
        super(PseudoLDA, self).__init__()
        self.a = a
        self.p = p

    def forward(self, density):
        return self.a * safepow(density.abs(), self.p)

    def potential(self, density):
        return self.a * self.p * safepow(density.abs(), self.p-1)

    def getfwdparams(self):
        return [self.a, self.p]

    def setfwdparams(self, *params):
        self.a, self.p = params[:2]
        return 2

def test_atom_grad():
    atomz = 1
    a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
    # NOTE: it does not work for 4/3 (the difference is very subtle)
    p = torch.tensor([5./3]).to(dtype).requires_grad_()

    def get_output(a, p, output="energy"):
        eks_model = PseudoLDA(a, p)
        energy, density = atom(atomz, eks_model=eks_model,
            gwmin=1e-5, gwmax=1e3, ng=60,
            rmin=1e-6, rmax=1e2, nr=200, dtype=dtype)
        if output == "energy":
            return energy
        else:
            return (density**4).sum()

    eps = 1e-4
    gradcheck(get_output, (a, p, "energy"), eps=eps)
    gradcheck(get_output, (a, p, "density"), eps=eps)
    gradgradcheck(get_output, (a, p, "energy"), eps=eps)
    gradgradcheck(get_output, (a, p, "density"), eps=eps)
