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
