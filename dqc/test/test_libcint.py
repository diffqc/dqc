from collections import namedtuple
import itertools
import torch
import pytest
import numpy as np
import warnings
from dqc.api.loadbasis import loadbasis
import dqc.hamilton.intor as intor
from dqc.utils.datastruct import AtomCGTOBasis, CGTOBasis
from dqc.system.tools import Lattice

# import pyscf
try:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pyscf
    import pyscf.pbc
except ImportError:
    raise ImportError("pyscf is needed for this test")

AtomEnv = namedtuple("AtomEnv", ["poss", "basis", "rgrid", "atomzs"])
dtype = torch.double

def get_atom_env(dtype, basis="3-21G", ngrid=0, pos_requires_grad=True, atomz=1, d=0.8):
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

def get_wrapper(atomenv, spherical=True, lattice=None):
    # get the wrapper from the given atom environment (i.e. the output of the
    # get_atom_env function)
    allbases = [
        loadbasis("%d:%s" % (max(atomz, 1), atomenv.basis), dtype=dtype, requires_grad=False)
        for atomz in atomenv.atomzs
    ]
    atombases = [
        AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=allbases[0], pos=atomenv.poss[0]),
        AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=allbases[1], pos=atomenv.poss[1]),
    ]
    wrap = intor.LibcintWrapper(atombases, spherical=spherical, lattice=lattice)
    return wrap

def get_mol_pyscf(dtype, basis="3-21G"):
    d = 0.8
    mol = pyscf.gto.M(atom="H 0 0 {d}; H 0 0 -{d}".format(d=d), basis=basis, unit="Bohr")
    return mol

def get_cell_pyscf(dtype, a, basis="3-21G"):
    d = 0.8
    mol = pyscf.pbc.gto.C(atom="H 0 0 {d}; H 0 0 -{d}".format(d=d), a=a, basis=basis, unit="Bohr")
    return mol

def get_int_type_and_frac(int_type):
    # given the integral type, returns the actual integral type and whether
    # it is a fractional z integral
    is_z_frac = False
    if "-frac" in int_type:
        int_type = int_type[:-5]
        is_z_frac = True
    return int_type, is_z_frac

#################### intors ####################
@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic", "nuclattr", "elrep", "coul2c", "coul3c"]
)
def test_integral_vs_pyscf(int_type):
    # check if the integrals from dqc agrees with pyscf

    atomenv = get_atom_env(dtype)
    env = get_wrapper(atomenv, spherical=True)

    if int_type == "overlap":
        mat = intor.overlap(env)
    elif int_type == "kinetic":
        mat = intor.kinetic(env)
    elif int_type == "nuclattr":
        mat = intor.nuclattr(env)
    elif int_type == "elrep":
        mat = intor.elrep(env)
    elif int_type == "coul2c":
        mat = intor.coul2c(env)
    elif int_type == "coul3c":
        mat = intor.coul3c(env)

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
    elif int_type == "coul2c":
        int_name = "int2c2e_sph"
    elif int_type == "coul3c":
        int_name = "int3c2e_sph"
    mat_scf = pyscf.gto.moleintor.getints(int_name, mol._atm, mol._bas, mol._env)

    assert torch.allclose(torch.tensor(mat_scf, dtype=dtype), mat)

@pytest.mark.parametrize(
    "intc_type",
    ["int2c", "int3c", "int4c"]
)
def test_integral_with_subset(intc_type):
    # check if the integral with the subsets agrees with the subset of the full integrals

    atomenv = get_atom_env(dtype)
    env = get_wrapper(atomenv, spherical=True)
    env1 = env[: len(env) // 2]
    nenv1 = env1.nao()
    if intc_type == "int2c":
        mat_full = intor.overlap(env)
        mat = intor.overlap(env, other=env1)
        mat1 = intor.overlap(env1)
        mat2 = intor.overlap(env1, other=env)

        assert torch.allclose(mat_full[:, :nenv1], mat)
        assert torch.allclose(mat_full[:nenv1, :nenv1], mat1)
        assert torch.allclose(mat_full[:nenv1, :], mat2)

    elif intc_type == "int3c":
        mat_full = intor.coul3c(env)
        mat = intor.coul3c(env, other1=env1, other2=env1)
        mat1 = intor.coul3c(env1, other1=env, other2=env)
        mat2 = intor.coul3c(env1, other1=env1, other2=env1)

        assert torch.allclose(mat_full[:, :nenv1, :nenv1], mat)
        assert torch.allclose(mat_full[:nenv1, :, :], mat1)
        assert torch.allclose(mat_full[:nenv1, :nenv1, :nenv1], mat2)

    elif intc_type == "int4c":
        mat_full = intor.elrep(env)
        mat = intor.elrep(env, other1=env1, other2=env1)
        mat1 = intor.elrep(env1, other1=env, other2=env, other3=env1)
        mat2 = intor.elrep(env1, other1=env1, other2=env1, other3=env1)

        assert torch.allclose(mat_full[:, :nenv1, :nenv1, :], mat)
        assert torch.allclose(mat_full[:nenv1, :, :, :nenv1], mat1)
        assert torch.allclose(mat_full[:nenv1, :nenv1, :nenv1, :nenv1], mat2)

    else:
        raise RuntimeError("Unknown integral type: %s" % intc_type)

def test_nuc_integral_frac_atomz():
    # test the nuclear integral with fractional atomz
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
        env = intor.LibcintWrapper([atombasis1, atombasis2], spherical=True)
        return intor.nuclattr(env)

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

    atomz = torch.tensor(2.1, dtype=dtype, requires_grad=True)

    def get_nuc_int1e(atomz):
        atomenv = get_atom_env(dtype, atomz=atomz)
        basis = loadbasis("%d:%s" % (2, atomenv.basis), dtype=dtype, requires_grad=False)

        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=basis, pos=atomenv.poss[0])
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=basis, pos=atomenv.poss[1])
        env = intor.LibcintWrapper([atombasis1, atombasis2], spherical=True)
        return intor.nuclattr(env)

    torch.autograd.gradcheck(get_nuc_int1e, (atomz,))
    torch.autograd.gradgradcheck(get_nuc_int1e, (atomz,))

@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic", "nuclattr", "nuclattr-frac",
     "elrep", "coul2c", "coul3c"]
)
def test_integral_grad_pos(int_type):
    int_type, is_z_frac = get_int_type_and_frac(int_type)

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
        env = intor.LibcintWrapper([atombasis1, atombasis2], spherical=True)
        if name == "overlap":
            return intor.overlap(env)
        elif name == "kinetic":
            return intor.kinetic(env)
        elif name == "nuclattr":
            return intor.nuclattr(env)
        elif name == "elrep":
            return intor.elrep(env)
        elif name == "coul2c":
            return intor.coul2c(env)
        elif name == "coul3c":
            return intor.coul3c(env)
        else:
            raise RuntimeError()

    # integrals gradcheck
    torch.autograd.gradcheck(get_int1e, (pos1, pos2, int_type))
    torch.autograd.gradgradcheck(get_int1e, (pos1, pos2, int_type))

@pytest.mark.parametrize(
    "intc_type,allsubsets",
    list(itertools.product(
        ["int2c", "int3c", "int4c"],
        [False, True]
    ))
)
def test_integral_subset_grad_pos(intc_type, allsubsets):

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
        env = intor.LibcintWrapper([atombasis1, atombasis2], spherical=True)
        env1 = env[: len(env) // 2]
        env2 = env[len(env) // 2:] if allsubsets else env
        if name == "int2c":
            return intor.nuclattr(env2, other=env1)
        elif name == "int3c":
            return intor.coul3c(env2, other1=env1, other2=env2)
        elif name == "int4c":
            return intor.elrep(env2, other1=env1, other2=env2, other3=env1)
        else:
            raise RuntimeError()

    # integrals gradcheck
    torch.autograd.gradcheck(get_int1e, (pos1, pos2, intc_type))
    torch.autograd.gradgradcheck(get_int1e, (pos1, pos2, intc_type))

@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic", "nuclattr", "nuclattr-frac",
     "elrep", "coul2c", "coul3c"]
)
def test_integral_grad_basis(int_type):
    int_type, is_z_frac = get_int_type_and_frac(int_type)
    torch.manual_seed(123)

    atomz = 1.2 if is_z_frac else 1
    atomenv = get_atom_env(dtype, atomz=atomz, pos_requires_grad=False)
    pos1 = atomenv.poss[0]
    pos2 = atomenv.poss[1]

    def get_int1e(alphas1, alphas2, coeffs1, coeffs2, name):
        # alphas*: (nangmoms, ngauss)
        bases1 = [
            CGTOBasis(angmom=i, alphas=alphas1[i], coeffs=coeffs1[i], normalized=True)
            for i in range(len(alphas1))
        ]
        bases2 = [
            CGTOBasis(angmom=i, alphas=alphas2[i], coeffs=coeffs2[i], normalized=True)
            for i in range(len(alphas2))
        ]
        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=bases1, pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=bases2, pos=pos2)
        env = intor.LibcintWrapper([atombasis1, atombasis2], spherical=True)
        if name == "overlap":
            return intor.overlap(env)
        elif name == "kinetic":
            return intor.kinetic(env)
        elif name == "nuclattr":
            return intor.nuclattr(env)
        elif name == "elrep":
            return intor.elrep(env)
        elif name == "coul2c":
            return intor.coul2c(env)
        elif name == "coul3c":
            return intor.coul3c(env)
        else:
            raise RuntimeError()

    # change the numbers to 1 for debugging
    if not int_type.startswith("elrep"):
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
        ["int2c", "int3c", "int4c"],
        [False, True]
    ))
)
def test_integral_subset_grad_basis(intc_type, allsubsets):
    torch.manual_seed(123)

    atomz = 1
    atomenv = get_atom_env(dtype, atomz=atomz, pos_requires_grad=False)
    pos1 = atomenv.poss[0]
    pos2 = atomenv.poss[1]

    def get_int1e(alphas1, alphas2, coeffs1, coeffs2, name):
        # alphas*: (nangmoms, ngauss)
        bases1 = [
            CGTOBasis(angmom=i, alphas=alphas1[i], coeffs=coeffs1[i], normalized=True)
            for i in range(len(alphas1))
        ]
        bases2 = [
            CGTOBasis(angmom=i, alphas=alphas2[i], coeffs=coeffs2[i])
            for i in range(len(alphas2))
        ]
        atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=bases1, pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=bases2, pos=pos2)
        env = intor.LibcintWrapper([atombasis1, atombasis2], spherical=True)
        env1 = env[: len(env) // 2]
        env2 = env[len(env) // 2:] if allsubsets else env
        if name == "int2c":
            return intor.nuclattr(env2, other=env1)
        elif name == "int3c":
            return intor.coul3c(env2, other1=env1, other2=env1)
        elif name == "int4c":
            return intor.elrep(env2, other1=env1, other2=env1, other3=env1)
        else:
            raise RuntimeError()

    # change the numbers to 1 for debugging
    if intc_type == "int2c":
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
    d = 0.8

    # setup the system for dqc
    atomenv = get_atom_env(dtype, ngrid=100)
    rgrid = atomenv.rgrid
    wrapper = get_wrapper(atomenv, spherical=True)
    wrapper1 = wrapper[:len(wrapper)]
    if eval_type == "":
        ao_value = intor.eval_gto(wrapper, rgrid)
        ao_value1 = intor.eval_gto(wrapper1, rgrid)
    elif eval_type == "grad":
        ao_value = intor.eval_gradgto(wrapper, rgrid)
        ao_value1 = intor.eval_gradgto(wrapper1, rgrid)

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
        env = intor.LibcintWrapper([atombasis1, atombasis2], spherical=True)
        env1 = env[:len(env) // 2] if partial else env
        if name == "":
            return intor.eval_gto(env1, rgrid)
        elif name == "grad":
            return intor.eval_gradgto(env1, rgrid)
        elif name == "lapl":
            return intor.eval_laplgto(env1, rgrid)
        else:
            raise RuntimeError("Unknown name: %s" % name)

    # evals gradcheck
    torch.autograd.gradcheck(evalgto, (pos1, pos2, rgrid, eval_type))
    torch.autograd.gradgradcheck(evalgto, (pos1, pos2, rgrid, eval_type))

################ pbc intor ################
@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic"]
)
def test_pbc_integral_1e_vs_pyscf(int_type):
    # check if the pbc 1-electron integrals from dqc agrees with pyscf's pbc_intor

    atomenv = get_atom_env(dtype)
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    env = get_wrapper(atomenv, spherical=True, lattice=Lattice(a))
    kpts = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.2, 0.1, 0.3],
    ], dtype=dtype)
    if int_type == "overlap":
        mat = intor.pbc_overlap(env, kpts=kpts)
    elif int_type == "kinetic":
        mat = intor.pbc_kinetic(env, kpts=kpts)
    else:
        raise RuntimeError("Unknown int_type: %s" % int_type)

    # get the matrix from pyscf
    cell = get_cell_pyscf(dtype, a.detach().numpy())
    if int_type == "overlap":
        int_name = "int1e_ovlp_sph"
    elif int_type == "kinetic":
        int_name = "int1e_kin_sph"
    else:
        raise RuntimeError("Unknown int_type: %s" % int_type)
    mat_scf = cell.pbc_intor(int_name, kpts=kpts.detach().numpy())

    print(mat)
    print(mat_scf)
    assert torch.allclose(torch.as_tensor(mat_scf, dtype=mat.dtype), mat, atol=2e-6)

@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic"]
)
def test_pbc_integral_1e_vs_pyscf_subset(int_type):
    # check if the pbc 1-electron integrals from dqc agrees with pyscf's pbc_intor
    # this function specifically test the integral with a subset

    atomenv = get_atom_env(dtype)
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    env = get_wrapper(atomenv, spherical=True, lattice=Lattice(a))
    env1 = env[: len(env) // 2]
    env2 = env[len(env) // 2:]
    kpts = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.2, 0.1, 0.3],
    ], dtype=dtype)
    if int_type == "overlap":
        fcn = intor.pbc_overlap
    elif int_type == "kinetic":
        fcn = intor.pbc_kinetic
    else:
        raise RuntimeError("Unknown int_type: %s" % int_type)
    mat_full = fcn(env, kpts=kpts)
    mat1_0 = fcn(env , other=env1, kpts=kpts)
    mat1_1 = fcn(env1, other=env1, kpts=kpts)
    mat1_2 = fcn(env1, other=env , kpts=kpts)
    mat2_0 = fcn(env , other=env2, kpts=kpts)
    mat2_1 = fcn(env2, other=env2, kpts=kpts)
    mat2_2 = fcn(env2, other=env , kpts=kpts)
    nenv1 = env1.nao()
    nenv2 = env2.nao()

    assert torch.allclose(mat_full[:, :, :nenv1], mat1_0)
    assert torch.allclose(mat_full[:, :nenv1, :nenv1], mat1_1)
    assert torch.allclose(mat_full[:, :nenv1, :], mat1_2)
    assert torch.allclose(mat_full[:, :, nenv2:], mat2_0)
    assert torch.allclose(mat_full[:, nenv2:, nenv2:], mat2_1)
    assert torch.allclose(mat_full[:, nenv2:, :], mat2_2)

def test_pbc_integral_2c2e_vs_pyscf():
    # check if the pbc 2-centre 2-electron integrals from dqc agrees with
    # pyscf's int2c2e_sph
    # this test uses compensating charge to make the integral converge

    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype) * 3
    latt = Lattice(a)
    kpts = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.2, 0.1, 0.3],
    ], dtype=dtype)

    normfcn = lambda alphas: 1.4366969770013325 * alphas ** 1.5
    alpha1 = 1.0  # dummy alpha
    alpha2 = 0.2  # compensating basis

    # create the neutral aux wrapper (hydrogen at 0.0)
    basis_h0 = CGTOBasis(angmom=0, alphas=torch.tensor([alpha1], dtype=dtype),
                         coeffs=torch.tensor([normfcn(alpha1)], dtype=dtype),
                         normalized=True)
    basis_h1 = CGTOBasis(angmom=0, alphas=torch.tensor([2 * alpha1], dtype=dtype),
                         coeffs=torch.tensor([normfcn(2 * alpha1)], dtype=dtype),
                         normalized=True)
    basis_hcomp0 = CGTOBasis(angmom=0, alphas=torch.tensor([alpha2], dtype=dtype),
                             coeffs=torch.tensor([normfcn(alpha2)], dtype=dtype),
                             normalized=True)
    basis_hcomp1 = CGTOBasis(angmom=0, alphas=torch.tensor([alpha2], dtype=dtype),
                             coeffs=torch.tensor([normfcn(alpha2)], dtype=dtype),
                             normalized=True)
    aux_atombases = [
        # real atom
        AtomCGTOBasis(atomz=1, bases=[basis_h0, basis_h1],
                      pos=torch.tensor([0.0, 0.0, 0.0], dtype=dtype)),
        # compensating basis
        AtomCGTOBasis(atomz=-1, bases=[basis_hcomp0, basis_hcomp1],
                      pos=torch.tensor([0.0, 0.0, 0.0], dtype=dtype)),
    ]
    n = 2  # number of nao per atom
    auxwrapper = intor.LibcintWrapper(aux_atombases, spherical=True,
                                      lattice=latt)

    mat_c = intor.pbc_coul2c(auxwrapper, other=auxwrapper, kpts=kpts)  # (nkpts, ni, nj)
    # get the sum of charge (+compensating basis to make it converge)
    mat = mat_c[..., :n, :n] - mat_c[..., :n, n:] - mat_c[..., n:, :n] + mat_c[..., n:, n:]

    # code to generate the pyscf_mat
    auxbasis = pyscf.gto.basis.parse("""
    H     S
          %f       1.0
    H     S
          %f       1.0
    H     S
          %f       1.0
    H     S
          %f       1.0
    """ % (alpha1, 2 * alpha1, alpha2, alpha2))
    int_name = "int2c2e_sph"
    auxcell = pyscf.pbc.gto.C(atom="H 0 0 0", a=a.detach().numpy(), spin=1, basis=auxbasis, unit="Bohr")
    # manually change the coefficients of the basis
    auxcell._env[-1] = normfcn(alpha2)
    auxcell._env[-3] = normfcn(alpha2)
    auxcell._env[-5] = normfcn(2 * alpha1)
    auxcell._env[-7] = normfcn(alpha1)
    pyscf_mat_c = np.asarray(auxcell.pbc_intor(int_name, kpts=kpts.detach().numpy()))
    pyscf_mat = pyscf_mat_c[..., :n, :n] - pyscf_mat_c[..., :n, n:] \
                - pyscf_mat_c[..., n:, :n] + pyscf_mat_c[..., n:, n:]

    print(mat.view(-1))
    assert torch.allclose(mat.view(-1), torch.as_tensor(pyscf_mat, dtype=mat.dtype).view(-1))

def test_pbc_integral_3c_vs_pyscf():
    # check if the pbc 3-centre integrals from dqc agrees with pyscf's aux_e2

    atomenv = get_atom_env(dtype)
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype) * 3
    latt = Lattice(a)
    env = get_wrapper(atomenv, spherical=True, lattice=latt)
    kpts_ij = torch.tensor([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.2, 0.1, 0.3]],
        [[0.2, 0.1, 0.3], [0.0, 0.0, 0.0]],
        [[0.2, 0.1, 0.3], [0.2, 0.1, 0.3]],
    ], dtype=dtype)

    normfcn = lambda alphas: 1.4366969770013325 * alphas ** 1.5
    alpha1 = 1e8  # fake nuclear gauss
    coeff1 = normfcn(alpha1)
    alpha2 = 0.2  # compensating basis
    coeff2 = normfcn(alpha2)

    # create the neutral aux wrapper (hydrogen at 0.0)
    basis_h = CGTOBasis(angmom=0, alphas=torch.tensor([alpha1], dtype=dtype),
                        coeffs=torch.tensor([coeff1], dtype=dtype),
                        normalized=True)
    basis_hcomp = CGTOBasis(angmom=0, alphas=torch.tensor([alpha2], dtype=dtype),
                            coeffs=torch.tensor([coeff2], dtype=dtype),
                            normalized=True)
    aux_atombases = [
        # real atom
        AtomCGTOBasis(atomz=1, bases=[basis_h],
                      pos=torch.tensor([0.0, 0.0, 0.0], dtype=dtype)),
        # compensating basis
        AtomCGTOBasis(atomz=-1, bases=[basis_hcomp],
                      pos=torch.tensor([0.0, 0.0, 0.0], dtype=dtype)),
    ]
    auxwrapper = intor.LibcintWrapper(aux_atombases, spherical=True,
                                      lattice=latt)

    env, auxwrapper = intor.LibcintWrapper.concatenate(env, auxwrapper)
    mat_c = intor.pbc_coul3c(env, auxwrapper=auxwrapper, kpts_ij=kpts_ij)
    mat = mat_c[..., 0] - mat_c[..., 1]  # get the sum of charge (+compensating basis to make it converge)

    # matrix generated from pyscf (code to generate is below)

    pyscf_mat = np.array(
        [[2.10365731+0.00000000e+00j, 3.65363178+0.00000000e+00j,
          1.46468006+0.00000000e+00j, 3.64358572+0.00000000e+00j,
          3.65363178+0.00000000e+00j, 9.80184173+0.00000000e+00j,
          3.64358572+0.00000000e+00j, 9.80160806+0.00000000e+00j,
          1.46468006+0.00000000e+00j, 3.64358572+0.00000000e+00j,
          2.10365731+0.00000000e+00j, 3.65363178+0.00000000e+00j,
          3.64358572+0.00000000e+00j, 9.80160806+0.00000000e+00j,
          3.65363178+0.00000000e+00j, 9.80184173+0.00000000e+00j],
         [1.9332373 -2.20446277e-01j, 2.7657514 -5.59129356e-01j,
          1.29893831+2.30596505e-01j, 2.69558993+7.53783995e-01j,
          3.2646191 -4.79168666e-01j, 7.23089875-1.73984640e+00j,
          3.25474826+4.79265696e-01j, 7.22941127+1.74381109e+00j,
          1.29893831-2.30596505e-01j, 2.69558993-7.53783995e-01j,
          1.9332373 +2.20446277e-01j, 2.7657514 +5.59129356e-01j,
          3.25474826-4.79265696e-01j, 7.22941127-1.74381109e+00j,
          3.2646191 +4.79168666e-01j, 7.23089875+1.73984640e+00j],
         [1.9332373 +2.20446277e-01j, 3.2646191 +4.79168666e-01j,
          1.29893831+2.30596505e-01j, 3.25474826+4.79265696e-01j,
          2.7657514 +5.59129356e-01j, 7.23089875+1.73984640e+00j,
          2.69558993+7.53783995e-01j, 7.22941127+1.74381109e+00j,
          1.29893831-2.30596505e-01j, 3.25474826-4.79265696e-01j,
          1.9332373 -2.20446277e-01j, 3.2646191 -4.79168666e-01j,
          2.69558993-7.53783995e-01j, 7.22941127-1.74381109e+00j,
          2.7657514 -5.59129356e-01j, 7.23089875-1.73984640e+00j],
         [2.05974021-1.11716192e-15j, 2.9256911 -1.36705691e-01j,
          1.22676707+4.73067139e-01j, 2.63603451+1.20083173e+00j,
          2.9256911 +1.36705691e-01j, 6.69912144+7.10542736e-15j,
          2.63603451+1.20083173e+00j, 5.95491913+3.06327919e+00j,
          1.22676707-4.73067139e-01j, 2.63603451-1.20083173e+00j,
          2.05974021-1.11022302e-15j, 2.9256911 +1.36705691e-01j,
          2.63603451-1.20083173e+00j, 5.95491913-3.06327919e+00j,
          2.9256911 -1.36705691e-01j, 6.69912144+8.88178420e-16j]])

    # # code to generate the pyscf_mat
    # cell = get_cell_pyscf(dtype, a.detach().numpy())
    # auxbasis = pyscf.gto.basis.parse("""
    # H     S
    #       %f       1.0
    # H     S
    #       %f       1.0
    # """ % (alpha1, alpha2))
    # auxcell = pyscf.pbc.gto.C(atom="H 0 0 0", a=a.detach().numpy(), spin=1, basis=auxbasis, unit="Bohr")
    # # manually change the coefficients of the basis
    # auxcell._env[-1] = coeff2
    # auxcell._env[-3] = coeff1
    # pyscf_mat_c = pyscf.pbc.df.incore.aux_e2(cell, auxcell, kptij_lst=kpts_ij.numpy())
    # pyscf_mat = pyscf_mat_c[..., 0] - pyscf_mat_c[..., 1]

    print(mat.view(-1))
    assert torch.allclose(mat.view(-1), torch.as_tensor(pyscf_mat, dtype=mat.dtype).view(-1))

#################### misc properties of LibcintWrapper ####################
def test_wrapper_concat():
    # get the wrappers
    atomenv1 = get_atom_env(dtype, d=2.0)
    env1 = get_wrapper(atomenv1, spherical=True)
    env1s = env1[: len(env1) // 2]

    atomenv2 = get_atom_env(dtype, atomz=0, d=1.0)
    env2 = get_wrapper(atomenv2, spherical=True)
    env2s = env2[: len(env2) // 2]

    env3 = get_wrapper(atomenv2, spherical=False)

    # concatenate the wrappers (with the same spherical)
    wrap1, wrap2 = intor.LibcintWrapper.concatenate(env1, env2)
    assert len(wrap1) == len(env1)
    assert len(wrap2) == len(env2)
    assert wrap1.shell_idxs == (0, len(wrap1))
    assert wrap2.shell_idxs == (len(wrap1), len(wrap1) + len(wrap2))
    assert isinstance(wrap1, intor.SubsetLibcintWrapper)
    assert wrap1.parent is wrap2.parent
    assert wrap1.parent.atombases == env1.atombases + env2.atombases
    for i in range(3):
        assert len(wrap1.params[i]) == len(env1.params[i]) + len(env2.params[i])

    # concatenate the subset with another subset
    wrap1s, wrap2s = intor.LibcintWrapper.concatenate(env1s, env2s)
    assert len(wrap1s) == len(env1s)
    assert len(wrap2s) == len(env2s)
    assert wrap1s.shell_idxs == env1s.shell_idxs
    assert wrap2s.shell_idxs == (env2s.shell_idxs[0] + len(env1), env2s.shell_idxs[1] + len(env1))
    assert isinstance(wrap1s, intor.SubsetLibcintWrapper)
    assert wrap1s.parent is wrap2s.parent
    assert wrap1s.parent.atombases == env1s.atombases + env2s.atombases
    for i in range(3):
        assert len(wrap1s.params[i]) == len(env1.params[i]) + len(env2.params[i])

    # concatenate 3 wrappers with some share the same parent
    wrap1, wrap2s, wrap1s = intor.LibcintWrapper.concatenate(env1, env2s, env1s)
    assert len(wrap1) == len(env1)
    assert len(wrap2s) == len(env2s)
    assert len(wrap1s) == len(env1s)
    off1 = len(env1)
    assert wrap1.shell_idxs == env1.shell_idxs
    assert wrap2s.shell_idxs == (env2s.shell_idxs[0] + off1, env2s.shell_idxs[1] + off1)
    assert wrap1s.shell_idxs == env1s.shell_idxs  # it belongs to the first parent
    for i in range(3):
        assert len(wrap1.params[i]) == len(env1.params[i]) + len(env2.params[i])

    # the case with the same parent
    wrap1, wrap1s = intor.LibcintWrapper.concatenate(env1, env1s)
    assert wrap1 is env1
    assert wrap1s is env1s
    for i in range(3):
        assert len(wrap1.params[i]) == len(env1.params[i])

    # concatenate with the unequal spherical
    try:
        wrap2, wrap3 = intor.LibcintWrapper.concatenate(env2, env3)
        assert False
    except AssertionError:  # TODO: change into ValueError
        pass
