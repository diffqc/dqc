from collections import namedtuple
import itertools
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

def get_atom_env(dtype, basis="3-21G", ngrid=0, pos_requires_grad=True, atomz=1):
    d = 0.8
    pos1 = torch.tensor([0.0, 0.0, d], dtype=dtype, requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0, 0.0, -d], dtype=dtype, requires_grad=pos_requires_grad)
    poss = [pos1, pos2]
    atomzs = [atomz, atomz]

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

def get_int_type_and_frac(int_type):
    # given the integral type, returns the actual integral type and whether
    # it is a fractional z integral
    is_z_frac = False
    if "-frac" in int_type:
        int_type = int_type[:-5]
        is_z_frac = True
    return int_type, is_z_frac

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
    "intc_type",
    ["int2c", "int4c"]
)
def test_integral_with_subset(intc_type):
    # check if the integral with the subsets agrees with the subset of the full integrals

    dtype = torch.double
    atomenv = get_atom_env(dtype)
    allbases = [
        loadbasis("%d:%s" % (atomz, atomenv.basis), dtype=dtype, requires_grad=False)
        for atomz in atomenv.atomzs
    ]

    atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=allbases[0], pos=atomenv.poss[0])
    atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=allbases[1], pos=atomenv.poss[1])
    env = LibcintWrapper([atombasis1, atombasis2], spherical=True)
    env1 = env[: len(env) // 2]
    nenv1 = env1.nao()
    if intc_type == "int2c":
        mat_full = env.overlap()
        mat = env.overlap(env1)
        mat1 = env1.overlap()
        mat2 = env1.overlap(other=env)

        assert torch.allclose(mat_full[:, :nenv1], mat)
        assert torch.allclose(mat_full[:nenv1, :nenv1], mat1)
        assert torch.allclose(mat_full[:nenv1, :], mat2)

    elif intc_type == "int4c":
        mat_full = env.elrep()
        mat = env.elrep(other1=env1, other2=env1)
        mat1 = env1.elrep(other1=env, other2=env, other3=env1)
        mat2 = env1.elrep(other1=env1, other2=env1, other3=env1)

        assert torch.allclose(mat_full[:, :nenv1, :nenv1, :], mat)
        assert torch.allclose(mat_full[:nenv1, :, :, :nenv1], mat1)
        assert torch.allclose(mat_full[:nenv1, :nenv1, :nenv1, :nenv1], mat2)

    else:
        raise RuntimeError("Unknown integral type: %s" % intc_type)

def test_nuc_integral_frac_atomz():
    # test the nuclear integral with fractional atomz
    dtype = torch.double

    atomenv1 = get_atom_env(dtype, atomz=1)
    atomenv2 = get_atom_env(dtype, atomz=2)
    atomenv1f = get_atom_env(dtype, atomz=1.0)
    atomenv2f = get_atom_env(dtype, atomz=2.0)
    atomenv15f = get_atom_env(dtype, atomz=1.5)
    natoms = len(atomenv1.atomzs)
    allbases = [
        loadbasis("%d:%s" % (1, atomenv1.basis), dtype=dtype, requires_grad=False)
        for i in range(natoms)
    ]

    def get_nuc_int1e(atomenv):
        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=allbases[0], pos=atomenv.poss[0])
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=allbases[1], pos=atomenv.poss[1])
        env = LibcintWrapper([atombasis1, atombasis2], spherical=True)
        return env.nuclattr()

    nuc1 = get_nuc_int1e(atomenv1)
    nuc2 = get_nuc_int1e(atomenv2)
    nuc1f = get_nuc_int1e(atomenv1f)
    nuc2f = get_nuc_int1e(atomenv2f)
    nuc15f = get_nuc_int1e(atomenv15f)
    assert torch.allclose(nuc1, nuc1f)
    assert torch.allclose(nuc2, nuc2f)
    nuc15 = (nuc1 + nuc2) * 0.5
    assert torch.allclose(nuc15, nuc15f)

def test_nuc_integral_frac_atomz_grad():
    # test the gradient w.r.t. Z for nuclear integral with fractional Z

    dtype = torch.double
    atomz = torch.tensor(2.1, dtype=dtype, requires_grad=True)

    def get_nuc_int1e(atomz):
        atomenv = get_atom_env(dtype, atomz=atomz)
        basis = loadbasis("%d:%s" % (2, atomenv.basis), dtype=dtype, requires_grad=False)

        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=basis, pos=atomenv.poss[0])
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=basis, pos=atomenv.poss[1])
        env = LibcintWrapper([atombasis1, atombasis2], spherical=True)
        return env.nuclattr()

    torch.autograd.gradcheck(get_nuc_int1e, (atomz,))
    torch.autograd.gradgradcheck(get_nuc_int1e, (atomz,))

@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic", "nuclattr", "nuclattr-frac", "elrep"]
)
def test_integral_grad_pos(int_type):
    int_type, is_z_frac = get_int_type_and_frac(int_type)
    dtype = torch.double

    atomz = 1.2 if is_z_frac else 1
    atomenv = get_atom_env(dtype, atomz=atomz)
    pos1 = atomenv.poss[0]
    pos2 = atomenv.poss[1]
    allbases = [
        loadbasis("%d:%s" % (int(atomz), atomenv.basis), dtype=dtype, requires_grad=False)
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
    "intc_type,allsubsets",
    list(itertools.product(
        ["int2c", "int4c"],
        [False, True]
    ))
)
def test_integral_subset_grad_pos(intc_type, allsubsets):
    dtype = torch.double

    atomz = 1
    atomenv = get_atom_env(dtype, atomz=atomz)
    pos1 = atomenv.poss[0]
    pos2 = atomenv.poss[1]
    allbases = [
        loadbasis("%d:%s" % (int(atomz), atomenv.basis), dtype=dtype, requires_grad=False)
        for atomz in atomenv.atomzs
    ]

    def get_int1e(pos1, pos2, name):
        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=allbases[0], pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=allbases[1], pos=pos2)
        env = LibcintWrapper([atombasis1, atombasis2], spherical=True)
        env1 = env[: len(env) // 2]
        env2 = env[len(env) // 2:] if allsubsets else env
        if name == "int2c":
            return env2.nuclattr(other=env1)
        elif name == "int4c":
            return env2.elrep(other1=env1, other2=env2, other3=env1)
        else:
            raise RuntimeError()

    # integrals gradcheck
    torch.autograd.gradcheck(get_int1e, (pos1, pos2, intc_type))
    torch.autograd.gradgradcheck(get_int1e, (pos1, pos2, intc_type))

@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic", "nuclattr", "nuclattr-frac", "elrep"]
)
def test_integral_grad_basis(int_type):
    int_type, is_z_frac = get_int_type_and_frac(int_type)
    dtype = torch.double
    torch.manual_seed(123)

    atomz = 1.2 if is_z_frac else 1
    atomenv = get_atom_env(dtype, atomz=atomz, pos_requires_grad=False)
    pos1 = atomenv.poss[0]
    pos2 = atomenv.poss[1]

    def get_int1e(alphas1, alphas2, coeffs1, coeffs2, name):
        # alphas*: (nangmoms, ngauss)
        bases1 = [
            CGTOBasis(angmom=i, alphas=alphas1[i], coeffs=coeffs1[i])
            for i in range(len(alphas1))
        ]
        bases2 = [
            CGTOBasis(angmom=i, alphas=alphas2[i], coeffs=coeffs2[i])
            for i in range(len(alphas2))
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
    if int_type != "elrep":
        ncontr, nangmom = (2, 2)
    else:
        ncontr, nangmom = (1, 1)  # saving time
    alphas1 = torch.rand((nangmom, ncontr), dtype=dtype, requires_grad=True)
    alphas2 = torch.rand((nangmom, ncontr), dtype=dtype, requires_grad=True)
    coeffs1 = torch.rand((nangmom, ncontr), dtype=dtype, requires_grad=True)
    coeffs2 = torch.rand((nangmom, ncontr), dtype=dtype, requires_grad=True)

    torch.autograd.gradcheck(get_int1e, (alphas1, alphas2, coeffs1, coeffs2, int_type))
    torch.autograd.gradgradcheck(get_int1e, (alphas1, alphas2, coeffs1, coeffs2, int_type))

@pytest.mark.parametrize(
    "intc_type,allsubsets",
    list(itertools.product(
        ["int2c", "int4c"],
        [False, True]
    ))
)
def test_integral_subset_grad_basis(intc_type, allsubsets):
    dtype = torch.double
    torch.manual_seed(123)

    atomz = 1
    atomenv = get_atom_env(dtype, atomz=atomz, pos_requires_grad=False)
    pos1 = atomenv.poss[0]
    pos2 = atomenv.poss[1]

    def get_int1e(alphas1, alphas2, coeffs1, coeffs2, name):
        # alphas*: (nangmoms, ngauss)
        bases1 = [
            CGTOBasis(angmom=i, alphas=alphas1[i], coeffs=coeffs1[i])
            for i in range(len(alphas1))
        ]
        bases2 = [
            CGTOBasis(angmom=i, alphas=alphas2[i], coeffs=coeffs2[i])
            for i in range(len(alphas2))
        ]
        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=bases1, pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=bases2, pos=pos2)
        env = LibcintWrapper([atombasis1, atombasis2], spherical=True, basis_normalized=True)
        env1 = env[: len(env) // 2]
        env2 = env[len(env) // 2:] if allsubsets else env
        if name == "int2c":
            return env2.nuclattr(other=env1)
        elif name == "int4c":
            return env2.elrep(other1=env1, other2=env1, other3=env1)
        else:
            raise RuntimeError()

    # change the numbers to 1 for debugging
    if intc_type != "int4c":
        ncontr, nangmom = (2, 2)
    else:
        ncontr, nangmom = (1, 1)  # saving time

    alphas1 = torch.rand((ncontr, nangmom), dtype=dtype, requires_grad=True)
    alphas2 = torch.rand((ncontr, nangmom), dtype=dtype, requires_grad=True)
    coeffs1 = torch.rand((ncontr, nangmom), dtype=dtype, requires_grad=True)
    coeffs2 = torch.rand((ncontr, nangmom), dtype=dtype, requires_grad=True)

    torch.autograd.gradcheck(get_int1e, (alphas1, alphas2, coeffs1, coeffs2, intc_type))
    torch.autograd.gradgradcheck(get_int1e, (alphas1, alphas2, coeffs1, coeffs2, intc_type))

@pytest.mark.parametrize(
    "eval_type",
    ["", "grad"]
)
def test_eval_gto_vs_pyscf(eval_type):
    # check if our eval_gto produces the same results as pyscf
    # also check the partial eval_gto

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
    wrapper1 = wrapper[:len(wrapper)]
    if eval_type == "":
        ao_value = wrapper.eval_gto(rgrid)
        ao_value1 = wrapper1.eval_gto(rgrid)
    elif eval_type == "grad":
        ao_value = wrapper.eval_gradgto(rgrid)
        ao_value1 = wrapper1.eval_gradgto(rgrid)

    # check the partial eval_gto
    assert torch.allclose(ao_value[..., :len(wrapper1), :], ao_value1)

    # system in pyscf
    mol = get_mol_pyscf(dtype)

    coords_np = rgrid.detach().numpy()
    if eval_type == "":
        ao_value_scf = mol.eval_gto("GTOval_sph", coords_np)
    elif eval_type == "grad":
        ao_value_scf = mol.eval_gto("GTOval_ip_sph", coords_np)
    ao_value_scf = torch.as_tensor(ao_value_scf).transpose(-2, -1)

    assert torch.allclose(ao_value, ao_value_scf)

@pytest.mark.parametrize(
    "eval_type,partial",
    list(itertools.product(
        ["", "grad", "lapl"],
        [False, True],
    ))
)
def test_eval_gto_grad_pos(eval_type, partial):
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
        env1 = env[:len(env) // 2] if partial else env
        if name == "":
            return env1.eval_gto(rgrid)
        elif name == "grad":
            return env1.eval_gradgto(rgrid)
        elif name == "lapl":
            return env1.eval_laplgto(rgrid)
        else:
            raise RuntimeError("Unknown name: %s" % name)

    # evals gradcheck
    torch.autograd.gradcheck(evalgto, (pos1, pos2, rgrid, eval_type))
    torch.autograd.gradgradcheck(evalgto, (pos1, pos2, rgrid, eval_type))
