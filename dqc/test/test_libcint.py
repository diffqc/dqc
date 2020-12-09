from collections import namedtuple
import torch
import pytest
import warnings
from dqc.api.loadbasis import loadbasis
from dqc.hamilton.lcintwrap import LibcintWrapper
from dqc.utils.datastruct import AtomCGTOBasis, CGTOBasis

# import pyscf
try:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pyscf
except ImportError:
    raise ImportError("pyscf is needed for this test")

AtomEnv = namedtuple("AtomEnv", ["poss", "basis", "rgrid", "atomzs"])

def get_atom_env(dtype, basis="3-21G", ngrid=0):
    d = 0.8
    pos1 = torch.tensor([0.0, 0.0, d], dtype=dtype, requires_grad=True)
    pos2 = torch.tensor([0.0, 0.0, -d], dtype=dtype, requires_grad=True)
    poss = [pos1, pos2]
    atomzs = [1, 1]

    rgrid = None
    if ngrid > 0:
        # set the grid
        n = ngrid
        z = torch.linspace(-5, 5, n, dtype=dtype)
        zeros = torch.zeros(n, dtype=dtype)
        rgrid = torch.cat((zeros[None, :], zeros[None, :], z[None, :]), dim=0).T.contiguous().to(dtype)

    atomenv = AtomEnv(
        poss=poss,
        basis=basis,
        rgrid=rgrid,
        atomzs=atomzs
    )
    return atomenv

def get_mol_pyscf(dtype, basis="3-21G"):
    d = 0.8
    mol = pyscf.gto.M(atom="H 0 0 {d}; H 0 0 -{d}".format(d=d), basis=basis, unit="Bohr")
    return mol

@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic", "nuclattr", "elrep"]
)
def test_integral_vs_pyscf(int_type):
    # check if the integrals from dqc agrees with pyscf

    dtype = torch.double
    atomenv = get_atom_env(dtype)
    allbases = [
        loadbasis("%d:%s" % (atomz, atomenv.basis), dtype=dtype, requires_grad=False)
        for atomz in atomenv.atomzs
    ]

    atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=allbases[0], pos=atomenv.poss[0])
    atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=allbases[1], pos=atomenv.poss[1])
    env = LibcintWrapper([atombasis1, atombasis2], spherical=True)
    if int_type == "overlap":
        mat = env.overlap()
    elif int_type == "kinetic":
        mat = env.kinetic()
    elif int_type == "nuclattr":
        mat = env.nuclattr()
    elif int_type == "elrep":
        mat = env.elrep()

    # get the matrix from pyscf
    mol = get_mol_pyscf(dtype)
    if int_type == "overlap":
        int_name = "int1e_ovlp_sph"
    elif int_type == "kinetic":
        int_name = "int1e_kin_sph"
    elif int_type == "nuclattr":
        int_name = "int1e_nuc_sph"
    elif int_type == "elrep":
        int_name = "int2e_sph"
    mat_scf = pyscf.gto.moleintor.getints(int_name, mol._atm, mol._bas, mol._env)

    assert torch.allclose(torch.tensor(mat_scf, dtype=dtype), mat)

@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic", "nuclattr", "elrep"]
)
def test_integral_grad_pos(int_type):
    dtype = torch.double

    atomenv = get_atom_env(dtype)
    pos1 = atomenv.poss[0]
    pos2 = atomenv.poss[1]
    allbases = [
        loadbasis("%d:%s" % (atomz, atomenv.basis), dtype=dtype, requires_grad=False)
        for atomz in atomenv.atomzs
    ]

    def get_int1e(pos1, pos2, name):
        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=allbases[0], pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=allbases[1], pos=pos2)
        env = LibcintWrapper([atombasis1, atombasis2], spherical=True)
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
    "int_type",
    ["overlap", "kinetic", "nuclattr"]
)
def test_integral_grad_basis(int_type):
    dtype = torch.double
    torch.manual_seed(123)

    atomenv = get_atom_env(dtype)
    pos1 = atomenv.poss[0]
    pos2 = atomenv.poss[1]
    def get_int1e(alphas1, alphas2, coeffs1, coeffs2, name):
        # alphas*: (2, ngauss)
        bases1 = [
            CGTOBasis(angmom=0, alphas=alphas1[0], coeffs=coeffs1[0]),
            CGTOBasis(angmom=1, alphas=alphas1[1], coeffs=coeffs1[1]),
        ]
        bases2 = [
            CGTOBasis(angmom=0, alphas=alphas2[0], coeffs=coeffs2[0]),
            CGTOBasis(angmom=1, alphas=alphas2[1], coeffs=coeffs2[1]),
        ]
        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=bases1, pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=bases2, pos=pos2)
        env = LibcintWrapper([atombasis1, atombasis2], spherical=True, basis_normalized=True)
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

    # change the numbers to 1 for debugging
    ncontr = 2
    alphas1 = torch.rand((2, ncontr), dtype=dtype, requires_grad=True)
    alphas2 = torch.rand((2, ncontr), dtype=dtype, requires_grad=True)
    coeffs1 = torch.rand((2, ncontr), dtype=dtype, requires_grad=True)
    coeffs2 = torch.rand((2, ncontr), dtype=dtype, requires_grad=True)

    torch.autograd.gradcheck(get_int1e, (alphas1, alphas2, coeffs1, coeffs2, int_type))
    # torch.autograd.gradcheck(get_int1e, (alphas1, alphas2, coeffs1, coeffs2, int_type))

@pytest.mark.parametrize(
    "eval_type",
    ["", "grad"]
)
def test_eval_gto_vs_pyscf(eval_type):
    basis = "6-311++G**"
    dtype = torch.double
    d = 0.8

    # setup the system for dqc
    atomenv = get_atom_env(dtype, ngrid=100)
    rgrid = atomenv.rgrid
    allbases = [
        loadbasis("%d:%s" % (atomz, atomenv.basis), dtype=dtype, requires_grad=False)
        for atomz in atomenv.atomzs
    ]
    atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=allbases[0], pos=atomenv.poss[0])
    atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=allbases[1], pos=atomenv.poss[1])
    wrapper = LibcintWrapper([atombasis1, atombasis2], spherical=True)
    if eval_type == "":
        ao_value = wrapper.eval_gto(rgrid)
    elif eval_type == "grad":
        ao_value = wrapper.eval_gradgto(rgrid)

    # system in pyscf
    mol = get_mol_pyscf(dtype)

    coords_np = rgrid.detach().numpy()
    if eval_type == "":
        ao_value_scf = mol.eval_gto("GTOval_sph", coords_np)
    elif eval_type == "grad":
        ao_value_scf = mol.eval_gto("GTOval_ip_sph", coords_np)

    torch.allclose(ao_value, torch.tensor(ao_value_scf).transpose(-2, -1))

@pytest.mark.parametrize(
    "eval_type",
    ["", "grad", "lapl"]
)
def test_eval_gto_grad_pos(eval_type):
    dtype = torch.double

    atomenv = get_atom_env(dtype, ngrid=3)
    pos1 = atomenv.poss[0]
    pos2 = atomenv.poss[1]
    allbases = [
        loadbasis("%d:%s" % (atomz, atomenv.basis), dtype=dtype, requires_grad=False)
        for atomz in atomenv.atomzs
    ]
    rgrid = atomenv.rgrid

    def evalgto(pos1, pos2, rgrid, name):
        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=allbases[0], pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=allbases[1], pos=pos2)
        env = LibcintWrapper([atombasis1, atombasis2], spherical=True)
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
