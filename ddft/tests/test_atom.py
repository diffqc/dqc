import torch
from ddft.api.atom import atom

def test_atom():
    dtype = torch.float64
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
