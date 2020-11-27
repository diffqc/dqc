import torch
import pytest
from ddft.basissets.cgtobasis import loadbasis
from ddft.hamiltons.libcintwrapper import LibcintWrapper
from ddft.basissets.cgtobasis import AtomCGTOBasis

@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic"]
)
def test_integral_grad(int_type):
    dtype = torch.double
    pos1 = torch.tensor([0.0, 0.0,  0.8], dtype=dtype, requires_grad=True)
    pos2 = torch.tensor([0.0, 0.0, -0.8], dtype=dtype, requires_grad=True)

    basis = "3-21G"

    def get_int1e(pos1, pos2, name):
        bases = loadbasis("1:%s" % basis, dtype=dtype, requires_grad=False)
        atombasis1 = AtomCGTOBasis(atomz=1, bases=bases, pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=1, bases=bases, pos=pos2)
        env = LibcintWrapper([atombasis1, atombasis2], spherical=False)
        if name == "overlap":
            return env.overlap()
        elif name == "kinetic":
            return env.kinetic()
        elif name == "nuclattr":
            return env.nuclattr()
        elif name == "elrep":
            return env.elrep()
        else:
            raise RuntimeError()

    # integrals gradcheck
    torch.autograd.gradcheck(get_int1e, (pos1, pos2, int_type))
    torch.autograd.gradgradcheck(get_int1e, (pos1, pos2, int_type))

@pytest.mark.parametrize(
    "eval_type",
    ["", "grad", "lapl"]
)
def test_eval_grad(eval_type):
    dtype = torch.double
    pos1 = torch.tensor([0.0, 0.0,  0.8], dtype=dtype, requires_grad=True)
    pos2 = torch.tensor([0.0, 0.0, -0.8], dtype=dtype, requires_grad=True)

    basis = "3-21G"

    # set the grid
    gradcheck = True
    n = 3 if gradcheck else 1000
    z = torch.linspace(-5, 5, n, dtype=dtype)
    zeros = torch.zeros(n, dtype=dtype)
    rgrid = torch.cat((zeros[None, :], zeros[None, :], z[None, :]), dim=0).T.contiguous().to(dtype)

    def evalgto(pos1, pos2, rgrid, name):
        bases = loadbasis("1:%s" % basis, dtype=dtype, requires_grad=False)
        atombasis1 = AtomCGTOBasis(atomz=1, bases=bases, pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=1, bases=bases, pos=pos2)
        env = LibcintWrapper([atombasis1, atombasis2], spherical=False)
        if name == "":
            return env.eval_gto(rgrid)
        elif name == "grad":
            return env.eval_gradgto(rgrid)
        elif name == "lapl":
            return env.eval_laplgto(rgrid)
        else:
            raise RuntimeError("Unknown name: %s" % name)

    # evals gradcheck
    torch.autograd.gradcheck(evalgto, (pos1, pos2, rgrid, eval_type))
    torch.autograd.gradgradcheck(evalgto, (pos1, pos2, rgrid, eval_type))
