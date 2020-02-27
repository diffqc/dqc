import torch
from ddft.api.atom import atom

def test_atom():
    dtype = torch.float64
    energies = {
        1: -0.4066,
        2: -2.7238,
    }

    for atomz, ene in energies.items():
        energy, density = atom(atomz, eks_model="lda", dtype=dtype)
        assert torch.allclose(energy, torch.tensor([ene], dtype=dtype), atol=1e-4)
