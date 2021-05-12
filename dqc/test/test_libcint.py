from collections import namedtuple
import itertools
import torch
import pytest
import numpy as np
import warnings
from dqc.api.loadbasis import loadbasis
import dqc.hamilton.intor as intor
from dqc.utils.datastruct import AtomCGTOBasis, CGTOBasis
from dqc.hamilton.intor.lattice import Lattice

# import pyscf
try:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pyscf
    import pyscf.pbc
except ImportError:
    raise ImportError("pyscf is needed for this test")

AtomEnv = namedtuple("AtomEnv", ["poss", "basis", "rgrid", "atomzs"])
dtype = torch.double

def get_atom_env(dtype, basis="3-21G", ngrid=0, pos_requires_grad=True, atomz=1, d=1.0):
    # non-symmetric, spin-neutral atomic configuration
    pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    poss = [pos1, pos2, pos3]
    atomzs = [atomz, atomz, atomz]
    return _construct_atom_env(poss, atomzs, dtype, basis, ngrid)

def _construct_atom_env(poss, atomzs, dtype, basis, ngrid):
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
        AtomCGTOBasis(atomz=atomenv.atomzs[i], bases=allbases[i], pos=atomenv.poss[i]) \
        for i in range(len(allbases))
    ]
    wrap = intor.LibcintWrapper(atombases, spherical=spherical, lattice=lattice)
    return wrap

def get_mol_pyscf(dtype, basis="3-21G"):
    mol = pyscf.gto.M(atom="H 0.1 0.0 0.2; H 0.0 1.0 -0.4; H 0.2 -1.4 -0.9",
                      basis=basis, unit="Bohr", spin=1)
    return mol

def get_cell_pyscf(dtype, a, basis="3-21G"):
    mol = pyscf.pbc.gto.C(atom="H 0.1 0.0 0.2; H 0.0 1.0 -0.4; H 0.2 -1.4 -0.9", a=a,
                          basis=basis, unit="Bohr", spin=1)
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
    ["overlap", "kinetic", "nuclattr", "elrep", "coul2c", "coul3c", "r0",
     "r0r0", "r0r0r0"]
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
    elif int_type == "r0":
        mat = intor.int1e("r0", env)
    elif int_type == "r0r0":
        mat = intor.int1e("r0r0", env)
    elif int_type == "r0r0r0":
        mat = intor.int1e("r0r0r0", env)

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
    elif int_type == "r0":
        int_name = "int1e_r_sph"
    elif int_type == "r0r0":
        int_name = "int1e_rr_sph"
    elif int_type == "r0r0r0":
        int_name = "int1e_rrr_sph"
    mat_scf = pyscf.gto.moleintor.getints(int_name, mol._atm, mol._bas, mol._env)
    mat_scf = torch.tensor(mat_scf, dtype=dtype)

    if not int_type.startswith("r0"):
        assert torch.allclose(mat_scf, mat)
    else:
        assert torch.allclose(mat_scf, mat, atol=5e-7)

@pytest.mark.parametrize(
    "intc_type",
    ["int2c", "int3c", "int4c"]
)
def test_integral_with_subset(intc_type):
    # check if the integral with the subsets agrees with the subset of the full integrals

    atomenv = get_atom_env(dtype)
    env = get_wrapper(atomenv, spherical=True)
    env1 = env[: len(env) // 2]
    env2 = env[: len(env) - 1]
    env3 = env[len(env) - 1:]
    nenv1 = env1.nao()
    nenv2 = env2.nao()
    nenv3 = env3.nao()
    if intc_type == "int2c":
        mat_full = intor.overlap(env)
        mat = intor.overlap(env, other=env1)
        mat1 = intor.overlap(env1)
        mat2 = intor.overlap(env1, other=env)
        mat3 = intor.overlap(env, other=env3)

        assert torch.allclose(mat_full[:, :nenv1], mat)
        assert torch.allclose(mat_full[:nenv1, :nenv1], mat1)
        assert torch.allclose(mat_full[:nenv1, :], mat2)
        assert torch.allclose(mat_full[:, -nenv3:], mat3)

    elif intc_type == "int3c":
        mat_full = intor.coul3c(env)
        mat = intor.coul3c(env, other1=env1, other2=env1)
        mat1 = intor.coul3c(env1, other1=env, other2=env)
        mat2 = intor.coul3c(env1, other1=env1, other2=env1)
        mat3 = intor.coul3c(env, other1=env1, other2=env2)
        mat4 = intor.coul3c(env, other1=env1, other2=env3)

        assert torch.allclose(mat_full[:, :nenv1, :nenv1], mat)
        assert torch.allclose(mat_full[:nenv1, :, :], mat1)
        assert torch.allclose(mat_full[:nenv1, :nenv1, :nenv1], mat2)
        assert torch.allclose(mat_full[:, :nenv1, :nenv2], mat3)
        assert torch.allclose(mat_full[:, :nenv1, -nenv3:], mat4)

    elif intc_type == "int4c":
        mat_full = intor.elrep(env)
        mat = intor.elrep(env, other1=env1, other2=env1)
        mat1 = intor.elrep(env1, other1=env, other2=env, other3=env1)
        mat2 = intor.elrep(env1, other1=env1, other2=env1, other3=env1)
        mat3 = intor.elrep(env1, other1=env1, other2=env, other3=env1)
        mat4 = intor.elrep(env, other1=env1, other2=env, other3=env1)
        mat5 = intor.elrep(env, other1=env1, other2=env2, other3=env3)

        assert torch.allclose(mat_full[:, :nenv1, :nenv1, :], mat)
        assert torch.allclose(mat_full[:nenv1, :, :, :nenv1], mat1)
        assert torch.allclose(mat_full[:nenv1, :nenv1, :nenv1, :nenv1], mat2)
        assert torch.allclose(mat_full[:nenv1, :nenv1, :, :nenv1], mat3)
        assert torch.allclose(mat_full[:, :nenv1, :, :nenv1], mat4)
        assert torch.allclose(mat_full[:, :nenv1, :nenv2, -nenv3:], mat5)

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
     "elrep", "coul2c", "coul3c", "r0", "r0r0"]
)
def test_integral_grad_pos(int_type):
    int_type, is_z_frac = get_int_type_and_frac(int_type)

    atomz = 1.2 if is_z_frac else 1
    atomenv = get_atom_env(dtype, atomz=atomz)
    poss = atomenv.poss
    allbases = [
        loadbasis("%d:%s" % (int(atomz), atomenv.basis), dtype=dtype, requires_grad=False)
        for atomz in atomenv.atomzs
    ]

    def get_int1e(name, *poss):
        atombases = [
            AtomCGTOBasis(atomz=atomenv.atomzs[i], bases=allbases[i], pos=poss[i]) \
            for i in range(len(poss))
        ]
        env = intor.LibcintWrapper(atombases, spherical=True)
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
        elif name == "r0":
            return intor.int1e("r0", env)
        elif name == "r0r0":
            return intor.int1e("r0r0", env)
        else:
            raise RuntimeError()

    # integrals gradcheck
    torch.autograd.gradcheck(get_int1e, (int_type, *poss))
    torch.autograd.gradgradcheck(get_int1e, (int_type, *poss))

@pytest.mark.parametrize(
    "intc_type,allsubsets",
    list(itertools.product(
        ["int2c", "int2cr", "int3c", "int4c"],
        [False, True]
    ))
)
def test_integral_subset_grad_pos(intc_type, allsubsets):

    atomz = 1
    atomenv = get_atom_env(dtype, atomz=atomz)
    poss = atomenv.poss
    allbases = [
        loadbasis("%d:%s" % (int(atomz), atomenv.basis), dtype=dtype, requires_grad=False)
        for atomz in atomenv.atomzs
    ]

    def get_int1e(name, *poss):
        atombases = [
            AtomCGTOBasis(atomz=atomenv.atomzs[i], bases=allbases[i], pos=poss[i]) \
            for i in range(len(poss))
        ]
        env = intor.LibcintWrapper(atombases, spherical=True)
        env1 = env[: len(env) // 2]
        env2 = env[len(env) // 2:] if allsubsets else env
        if name == "int2c":
            return intor.nuclattr(env2, other=env1)
        elif name == "int2cr":
            return intor.int1e("r0", env2, other=env1)
        elif name == "int3c":
            return intor.coul3c(env2, other1=env1, other2=env2)
        elif name == "int4c":
            return intor.elrep(env2, other1=env1, other2=env2, other3=env1)
        else:
            raise RuntimeError()

    # integrals gradcheck
    torch.autograd.gradcheck(get_int1e, (intc_type, *poss))
    torch.autograd.gradgradcheck(get_int1e, (intc_type, *poss))

@pytest.mark.parametrize(
    "int_type",
    ["overlap", "kinetic", "nuclattr", "nuclattr-frac",
     "elrep", "coul2c", "coul3c", "r0", "r0r0"]
)
def test_integral_grad_basis(int_type):
    int_type, is_z_frac = get_int_type_and_frac(int_type)
    torch.manual_seed(123)

    atomz = 1.2 if is_z_frac else 1
    atomenv = get_atom_env(dtype, atomz=atomz, pos_requires_grad=False)
    poss = atomenv.poss
    natoms = len(poss)
    # pos1 = atomenv.poss[0]
    # pos2 = atomenv.poss[1]

    def get_int1e(alphas, coeffs, name):
    # def get_int1e(alphas1, alphas2, coeffs1, coeffs2, name):
        # alphas*: (nangmoms, ngauss)
        bases = [
            [
                CGTOBasis(angmom=i, alphas=alphas[j][i], coeffs=coeffs[j][i], normalized=True)
                for i in range(len(alphas[j]))
            ]
            for j in range(len(alphas))
        ]
        # bases1 = [
        #     CGTOBasis(angmom=i, alphas=alphas1[i], coeffs=coeffs1[i], normalized=True)
        #     for i in range(len(alphas1))
        # ]
        # bases2 = [
        #     CGTOBasis(angmom=i, alphas=alphas2[i], coeffs=coeffs2[i], normalized=True)
        #     for i in range(len(alphas2))
        # ]
        atombases = [
            AtomCGTOBasis(atomz=atomenv.atomzs[i], bases=bases[i], pos=poss[i]) \
            for i in range(len(bases))
        ]
        # atombasis1 = AtomCGTOBasis(atomz=atomenv.atomzs[0], bases=bases1, pos=pos1)
        # atombasis2 = AtomCGTOBasis(atomz=atomenv.atomzs[1], bases=bases2, pos=pos2)
        # env = intor.LibcintWrapper([atombasis1, atombasis2], spherical=True)
        env = intor.LibcintWrapper(atombases, spherical=True)
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
        elif name == "r0":
            return intor.int1e("r0", env)
        elif name == "r0r0":
            return intor.int1e("r0r0", env)
        else:
            raise RuntimeError()

    if int_type in ["elrep", "r0", "r0r0", "r0r0r0", "coul3c"]:
        ncontr, nangmom = (1, 1)  # saving time
    # change the numbers to 1 for debugging
    else:
        ncontr, nangmom = (2, 2)
    alphas = torch.rand((natoms, nangmom, ncontr), dtype=dtype, requires_grad=True)
    coeffs = torch.rand((natoms, nangmom, ncontr), dtype=dtype, requires_grad=True)
    # alphas1 = torch.rand((nangmom, ncontr), dtype=dtype, requires_grad=True)
    # alphas2 = torch.rand((nangmom, ncontr), dtype=dtype, requires_grad=True)
    # coeffs1 = torch.rand((nangmom, ncontr), dtype=dtype, requires_grad=True)
    # coeffs2 = torch.rand((nangmom, ncontr), dtype=dtype, requires_grad=True)

    # torch.autograd.gradcheck(get_int1e, (alphas1, alphas2, coeffs1, coeffs2, int_type))
    # torch.autograd.gradgradcheck(get_int1e, (alphas1, alphas2, coeffs1, coeffs2, int_type))

    torch.autograd.gradcheck(get_int1e, (alphas, coeffs, int_type))
    torch.autograd.gradgradcheck(get_int1e, (alphas, coeffs, int_type))

@pytest.mark.parametrize(
    "intc_type,allsubsets",
    list(itertools.product(
        ["int2c", "int2cr", "int3c", "int4c"],
        [False, True]
    ))
)
def test_integral_subset_grad_basis(intc_type, allsubsets):
    torch.manual_seed(123)

    atomz = 1
    atomenv = get_atom_env(dtype, atomz=atomz, pos_requires_grad=False)
    poss = atomenv.poss
    natoms = len(poss)

    def get_int1e(alphas, coeffs, name):
        # alphas*: (nangmoms, ngauss)
        bases = [
            [
                CGTOBasis(angmom=i, alphas=alphas[j][i], coeffs=coeffs[j][i], normalized=True)
                for i in range(len(alphas[j]))
            ]
            for j in range(len(alphas))
        ]
        atombases = [
            AtomCGTOBasis(atomz=atomenv.atomzs[i], bases=bases[i], pos=poss[i]) \
            for i in range(len(bases))
        ]
        env = intor.LibcintWrapper(atombases, spherical=True)
        env1 = env[: len(env) // 2]
        env2 = env[len(env) // 2:] if allsubsets else env
        if name == "int2c":
            return intor.nuclattr(env2, other=env1)
        elif name == "int2cr":
            return intor.int1e("r0", env2, other=env1)
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

    alphas = torch.rand((natoms, ncontr, nangmom), dtype=dtype, requires_grad=True)
    coeffs = torch.rand((natoms, ncontr, nangmom), dtype=dtype, requires_grad=True)

    torch.autograd.gradcheck(get_int1e, (alphas, coeffs, intc_type))
    torch.autograd.gradgradcheck(get_int1e, (alphas, coeffs, intc_type))

@pytest.mark.parametrize(
    "eval_type",
    ["", "grad", "lapl"]
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
    elif eval_type == "lapl":
        ao_value = intor.eval_laplgto(wrapper, rgrid)
        ao_value1 = intor.eval_laplgto(wrapper1, rgrid)

    # check the partial eval_gto
    assert torch.allclose(ao_value[..., :len(wrapper1), :], ao_value1)

    # system in pyscf
    mol = get_mol_pyscf(dtype)

    coords_np = rgrid.detach().numpy()
    if eval_type == "":
        ao_value_scf = mol.eval_gto("GTOval_sph", coords_np)
    elif eval_type == "grad":
        ao_value_scf = mol.eval_gto("GTOval_ip_sph", coords_np)
    elif eval_type == "lapl":
        ao_deriv2 = mol.eval_gto("GTOval_sph_deriv2", coords_np)
        ao_value_scf = ao_deriv2[4] + ao_deriv2[7] + ao_deriv2[9]  # 4: xx, 7: yy, 9: zz
    ao_value_scf = torch.as_tensor(ao_value_scf).transpose(-2, -1)

    assert torch.allclose(ao_value, ao_value_scf, atol=2e-7)

@pytest.mark.parametrize(
    "eval_type",
    ["", "grad", "lapl"]
)
def test_eval_gto_transpose(eval_type):
    # check if our eval_gto transpose produces the correct transposed results

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
        ao_valueT = intor.eval_gto(wrapper, rgrid, to_transpose=True)
        ao_value1T = intor.eval_gto(wrapper1, rgrid, to_transpose=True)
    elif eval_type == "grad":
        ao_value = intor.eval_gradgto(wrapper, rgrid)
        ao_value1 = intor.eval_gradgto(wrapper1, rgrid)
        ao_valueT = intor.eval_gradgto(wrapper, rgrid, to_transpose=True)
        ao_value1T = intor.eval_gradgto(wrapper1, rgrid, to_transpose=True)
    elif eval_type == "lapl":
        ao_value = intor.eval_laplgto(wrapper, rgrid)
        ao_value1 = intor.eval_laplgto(wrapper1, rgrid)
        ao_valueT = intor.eval_laplgto(wrapper, rgrid, to_transpose=True)
        ao_value1T = intor.eval_laplgto(wrapper1, rgrid, to_transpose=True)

    # make sure they are contiguous (TODO: any better way to check is contiguous?)
    ao_value.view(-1)
    ao_value1.view(-1)
    ao_valueT.view(-1)
    ao_value1T.view(-1)

    assert torch.allclose(ao_value, ao_valueT.transpose(-2, -1))
    assert torch.allclose(ao_value1, ao_value1T.transpose(-2, -1))

@pytest.mark.parametrize(
    "eval_type,partial,to_transpose",
    list(itertools.product(
        ["", "grad", "lapl"],
        [False, True],
        [False, True],
    ))
)
def test_eval_gto_grad_pos(eval_type, partial, to_transpose):

    atomenv = get_atom_env(dtype, ngrid=3)
    poss = atomenv.poss
    allbases = [
        loadbasis("%d:%s" % (atomz, atomenv.basis), dtype=dtype, requires_grad=False)
        for atomz in atomenv.atomzs
    ]
    rgrid = atomenv.rgrid

    def evalgto(rgrid, name, *poss):
        atombases = [
            AtomCGTOBasis(atomz=atomenv.atomzs[i], bases=allbases[i], pos=poss[i]) \
            for i in range(len(poss))
        ]
        env = intor.LibcintWrapper(atombases, spherical=True)
        env1 = env[:len(env) // 2] if partial else env
        if name == "":
            return intor.eval_gto(env1, rgrid, to_transpose=to_transpose)
        elif name == "grad":
            return intor.eval_gradgto(env1, rgrid, to_transpose=to_transpose)
        elif name == "lapl":
            return intor.eval_laplgto(env1, rgrid, to_transpose=to_transpose)
        else:
            raise RuntimeError("Unknown name: %s" % name)

    # evals gradcheck
    torch.autograd.gradcheck(evalgto, (rgrid, eval_type, *poss))
    torch.autograd.gradgradcheck(evalgto, (rgrid, eval_type, *poss))

@pytest.mark.parametrize(
    "eval_type,partial,to_transpose",
    list(itertools.product(
        ["", "grad", "lapl"],
        [False, True],
        [False, True],
    ))
)
def test_eval_gto_grad_basis(eval_type, partial, to_transpose):
    name = eval_type
    alphas = torch.tensor([1.3098, 0.2331], dtype=torch.double).requires_grad_()
    coeffs = torch.tensor([1.3305, 0.5755], dtype=torch.double).requires_grad_()
    rgrid = torch.tensor([[-0.75, 0.0, 0.0],
                          [-0.50, 0.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [0.50, 0.0, 0.0],
                          [0.75, 0.0, 0.0]], dtype=torch.double)

    def evalgto(alphas, coeffs):
        bases = [CGTOBasis(angmom=0, alphas=alphas, coeffs=coeffs, normalized=True)]
        atombases = [
            AtomCGTOBasis(atomz=1, bases=bases, pos=torch.tensor([-0.5, 0, 0])),
            AtomCGTOBasis(atomz=1, bases=bases, pos=torch.tensor([0.5, 0, 0])),
        ]
        env = intor.LibcintWrapper(atombases, spherical=True)
        env1 = env[:len(env) // 2] if partial else env
        if name == "":
            return intor.eval_gto(env1, rgrid, to_transpose=to_transpose)
        elif name == "grad":
            return intor.eval_gradgto(env1, rgrid, to_transpose=to_transpose)
        elif name == "lapl":
            return intor.eval_laplgto(env1, rgrid, to_transpose=to_transpose)
        else:
            raise RuntimeError("Unknown name: %s" % name)

    # evals gradcheck
    torch.autograd.gradcheck(evalgto, (alphas, coeffs))
    torch.autograd.gradgradcheck(evalgto, (alphas, coeffs))

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
    mat_scf = torch.as_tensor(mat_scf, dtype=mat.dtype)

    print(mat)
    print(mat_scf)
    print(mat - mat_scf)
    print((mat - mat_scf).abs().max())
    assert torch.allclose(torch.as_tensor(mat_scf, dtype=mat.dtype), mat, atol=8e-6)

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
    mat_c = intor.pbc_coul3c(env, other2=auxwrapper, kpts_ij=kpts_ij)
    mat = mat_c[..., 0] - mat_c[..., 1]  # get the sum of charge (+compensating basis to make it converge)

    # matrix generated from pyscf (code to generate is below)

    pyscf_mat = np.array(
        [[3.18415189+0.00000000e+00j, 4.34634154+0.00000000e+00j,
          1.61959981+0.00000000e+00j, 4.32458705+0.00000000e+00j,
          1.02216289+0.00000000e+00j, 4.30799199+0.00000000e+00j,
          4.34634154+0.00000000e+00j, 9.81979759+0.00000000e+00j,
          3.31315702+0.00000000e+00j, 9.80401607+0.00000000e+00j,
          2.79936732+0.00000000e+00j, 9.78998751+0.00000000e+00j,
          1.61959981+0.00000000e+00j, 3.31315702+0.00000000e+00j,
          1.72505442+0.00000000e+00j, 3.31707926+0.00000000e+00j,
          1.18555734+0.00000000e+00j, 3.30613642+0.00000000e+00j,
          4.32458705+0.00000000e+00j, 9.80401607+0.00000000e+00j,
          3.31707926+0.00000000e+00j, 9.78879831+0.00000000e+00j,
          2.80473336+0.00000000e+00j, 9.77484017+0.00000000e+00j,
          1.02216289+0.00000000e+00j, 2.79936732+0.00000000e+00j,
          1.18555734+0.00000000e+00j, 2.80473336+0.00000000e+00j,
          1.28777369+0.00000000e+00j, 2.80627827+0.00000000e+00j,
          4.30799199+0.00000000e+00j, 9.78998751+0.00000000e+00j,
          3.30613642+0.00000000e+00j, 9.77484017+0.00000000e+00j,
          2.80627827+0.00000000e+00j, 9.7612235 +0.00000000e+00j],
         [3.061388  -6.34041547e-02j, 3.41351142-2.05466963e-01j,
          1.52385377-3.43588405e-02j, 3.38966595+1.08988377e-01j,
          0.88778206+2.38180367e-01j, 3.11643251+1.25105936e+00j,
          4.05708002-1.40352261e-01j, 7.4503628 -5.82496647e-01j,
          3.0191812 -2.58553342e-02j, 7.45165654+1.34532940e-01j,
          2.33988439+7.44624433e-01j, 6.92508674+2.66410092e+00j,
          1.53106238-2.93697784e-02j, 2.54910517-2.23740473e-01j,
          1.60170258-1.77300967e-02j, 2.56449735+1.48841044e-02j,
          1.02099669+3.05158096e-01j, 2.3878365 +8.80563319e-01j,
          4.03614068-1.39477897e-01j, 7.43717154-5.83397287e-01j,
          3.02246396-2.54313341e-02j, 7.43934726+1.32448164e-01j,
          2.34433946+7.46386342e-01j, 6.91447079+2.65786726e+00j,
          0.9444924 -2.43365468e-02j, 2.10687816-2.14824032e-01j,
          1.07714184-1.42467612e-02j, 2.12679772-1.26646016e-02j,
          1.09190724+3.32572269e-01j, 2.00295594+7.11789051e-01j,
          4.01986534-1.39352250e-01j, 7.42511225-5.83105549e-01j,
          3.01185597-2.53355916e-02j, 7.42745756+1.31751106e-01j,
          2.34526536+7.46968551e-01j, 6.90400801+2.65333074e+00j],
         [3.061388  +6.34041547e-02j, 4.05708002+1.40352261e-01j,
          1.53106238+2.93697784e-02j, 4.03614068+1.39477897e-01j,
          0.9444924 +2.43365468e-02j, 4.01986534+1.39352250e-01j,
          3.41351142+2.05466963e-01j, 7.4503628 +5.82496647e-01j,
          2.54910517+2.23740473e-01j, 7.43717154+5.83397287e-01j,
          2.10687816+2.14824032e-01j, 7.42511225+5.83105549e-01j,
          1.52385377+3.43588405e-02j, 3.0191812 +2.58553342e-02j,
          1.60170258+1.77300967e-02j, 3.02246396+2.54313341e-02j,
          1.07714184+1.42467612e-02j, 3.01185597+2.53355916e-02j,
          3.38966595-1.08988377e-01j, 7.45165654-1.34532940e-01j,
          2.56449735-1.48841044e-02j, 7.43934726-1.32448164e-01j,
          2.12679772+1.26646016e-02j, 7.42745756-1.31751106e-01j,
          0.88778206-2.38180367e-01j, 2.33988439-7.44624433e-01j,
          1.02099669-3.05158096e-01j, 2.34433946-7.46386342e-01j,
          1.09190724-3.32572269e-01j, 2.34526536-7.46968551e-01j,
          3.11643251-1.25105936e+00j, 6.92508674-2.66410092e+00j,
          2.3878365 -8.80563319e-01j, 6.91447079-2.65786726e+00j,
          2.00295594-7.11789051e-01j, 6.90400801-2.65333074e+00j],
         [3.14714285-2.84494650e-16j, 3.52300635-8.22119122e-02j,
          1.56973162-5.89339425e-03j, 3.48460877+2.40005038e-01j,
          0.90309237+2.68491282e-01j, 3.15865335+1.40601700e+00j,
          3.52300635+8.22119122e-02j, 6.73471982+4.88498131e-15j,
          2.65224685+2.16858070e-01j, 6.6869355 +6.52053364e-01j,
          2.03045716+8.96761460e-01j, 6.0388224 +2.89224807e+00j,
          1.56973162+5.89339425e-03j, 2.65224685-2.16858070e-01j,
          1.6926784 -8.18789481e-16j, 2.67023551+3.03918811e-02j,
          1.08994827+3.39792411e-01j, 2.48196126+9.29463048e-01j,
          3.48460877-2.40005038e-01j, 6.6869355 -6.52053364e-01j,
          2.67023551-3.03918811e-02j, 6.70480217+9.32587341e-15j,
          2.12144113+7.02009818e-01j, 6.27899604+2.28693981e+00j,
          0.90309237-2.68491282e-01j, 2.03045716-8.96761460e-01j,
          1.08994827-3.39792411e-01j, 2.12144113-7.02009818e-01j,
          1.25577244-4.99600361e-16j, 2.23671933+6.02728622e-02j,
          3.15865335-1.40601700e+00j, 6.0388224 -2.89224807e+00j,
          2.48196126-9.29463048e-01j, 6.27899604-2.28693981e+00j,
          2.23671933-6.02728622e-02j, 6.66183656-5.32907052e-15j]])

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

@pytest.mark.parametrize(
    "int_type",
    ["overlap"]
)
def test_pbc_ft_integral_1e(int_type):
    # test various aspects of the pbcft integrator, including comparing it with
    # pyscf value

    atomenv = get_atom_env(dtype)
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    env = get_wrapper(atomenv, spherical=True, lattice=Lattice(a))
    kpts = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.2, 0.1, 0.3],
    ], dtype=dtype)
    ngrid = 3
    gvgrid = torch.zeros(ngrid, 3, dtype=dtype)  # (ngrid, ndim)
    gvgrid[:, 0] = torch.linspace(-5, 5, ngrid, dtype=dtype)

    if int_type == "overlap":
        opft = intor.pbcft_overlap
        op = intor.pbc_overlap
    else:
        raise RuntimeError("Unknown int_type: %s" % int_type)

    # check the shape of the integral
    mat_ft = opft(env, kpts=kpts, gvgrid=gvgrid)
    assert mat_ft.shape[0] == kpts.shape[0]
    assert mat_ft.shape[-1] == ngrid
    assert mat_ft.shape[-2] == env.nao()
    assert mat_ft.shape[-3] == env.nao()

    # check the subset
    env1 = env[: len(env) // 2]
    mat_ft1 = opft(env, other=env1, kpts=kpts, gvgrid=gvgrid)
    mat_ft2 = opft(env1, other=env1, kpts=kpts, gvgrid=gvgrid)
    nenv1 = env1.nao()
    assert torch.allclose(mat_ft[:, :, :nenv1], mat_ft1)
    assert torch.allclose(mat_ft[:, :nenv1, :nenv1], mat_ft2)

    # check if the results are equal if gvgrid is all zeros (i.e. ignored)
    mat0ft = opft(env, kpts=kpts).squeeze(-1)
    mat0 = op(env, kpts=kpts)
    assert torch.allclose(mat0ft, mat0)

    # compare it with precomputed values from pyscf
    # the code to obtain the tensor is written below
    mat_scf_t = torch.tensor([[[[ 2.8582e-01+1.5615e-01j,  1.8695e+01+0.0000e+00j,
            2.8582e-01-1.5615e-01j],
          [ 7.1320e-01+3.8963e-01j,  6.1272e+01+0.0000e+00j,
            7.1320e-01-3.8963e-01j],
          [ 2.9326e-01+1.0914e-01j,  1.8692e+01+0.0000e+00j,
            2.9326e-01-1.0914e-01j],
          [ 7.1320e-01+3.8963e-01j,  6.1272e+01+0.0000e+00j,
            7.1320e-01-3.8963e-01j],
          [ 2.5018e-01+1.8771e-01j,  1.8692e+01+0.0000e+00j,
            2.5018e-01-1.8771e-01j],
          [ 7.1320e-01+3.8963e-01j,  6.1272e+01+0.0000e+00j,
            7.1320e-01-3.8963e-01j]],

         [[ 3.7394e-02+2.0428e-02j,  6.1272e+01+0.0000e+00j,
            3.7394e-02-2.0428e-02j],
          [ 2.6876e-13+1.4683e-13j,  2.0087e+02+0.0000e+00j,
            2.6876e-13-1.4683e-13j],
          [ 4.2260e-02-5.4527e-03j,  6.1272e+01+0.0000e+00j,
            4.2260e-02+5.4527e-03j],
          [ 2.6875e-13+1.4676e-13j,  2.0087e+02+0.0000e+00j,
            2.6875e-13-1.4676e-13j],
          [ 1.8245e-02+3.8506e-02j,  6.1272e+01+0.0000e+00j,
            1.8245e-02-3.8506e-02j],
          [ 2.6867e-13+1.4684e-13j,  2.0087e+02+0.0000e+00j,
            2.6867e-13-1.4684e-13j]],

         [[ 3.0968e-01+4.4814e-02j,  1.8692e+01+0.0000e+00j,
            3.0968e-01-4.4814e-02j],
          [ 8.1269e-01+2.6394e-18j,  6.1272e+01+0.0000e+00j,
            8.1269e-01-2.6394e-18j],
          [ 3.2569e-01-1.2613e-19j,  1.8695e+01+0.0000e+00j,
            3.2569e-01+1.2613e-19j],
          [ 8.1269e-01+4.4251e-18j,  6.1272e+01+0.0000e+00j,
            8.1269e-01-4.4251e-18j],
          [ 2.7042e-01+7.2065e-02j,  1.8688e+01+0.0000e+00j,
            2.7042e-01-7.2065e-02j],
          [ 8.1269e-01-5.2337e-17j,  6.1272e+01+0.0000e+00j,
            8.1269e-01+5.2337e-17j]],

         [[ 3.4472e-02+2.5046e-02j,  6.1272e+01+0.0000e+00j,
            3.4472e-02-2.5046e-02j],
          [ 3.0621e-13+4.7671e-17j,  2.0087e+02+0.0000e+00j,
            3.0621e-13-4.7671e-17j],
          [ 4.2610e-02+5.0912e-18j,  6.1272e+01+0.0000e+00j,
            4.2610e-02-5.0912e-18j],
          [ 3.0625e-13-4.9440e-23j,  2.0087e+02+0.0000e+00j,
            3.0625e-13+4.9440e-23j],
          [ 1.3167e-02+4.0525e-02j,  6.1272e+01+0.0000e+00j,
            1.3167e-02-4.0525e-02j],
          [ 3.0596e-13-6.0847e-17j,  2.0087e+02+0.0000e+00j,
            3.0596e-13+6.0847e-17j]],

         [[ 2.0494e-01+2.3628e-01j,  1.8692e+01+0.0000e+00j,
            2.0494e-01-2.3628e-01j],
          [ 4.3910e-01+6.8386e-01j,  6.1272e+01+0.0000e+00j,
            4.3910e-01-6.8386e-01j],
          [ 2.0675e-01+1.8861e-01j,  1.8688e+01+0.0000e+00j,
            2.0675e-01-1.8861e-01j],
          [ 4.3910e-01+6.8386e-01j,  6.1272e+01+0.0000e+00j,
            4.3910e-01-6.8386e-01j],
          [ 1.7597e-01+2.7406e-01j,  1.8695e+01+0.0000e+00j,
            1.7597e-01-2.7406e-01j],
          [ 4.3910e-01+6.8386e-01j,  6.1272e+01+0.0000e+00j,
            4.3910e-01-6.8386e-01j]],

         [[ 3.9701e-02+1.5475e-02j,  6.1272e+01+0.0000e+00j,
            3.9701e-02-1.5475e-02j],
          [ 1.6548e-13+2.5761e-13j,  2.0087e+02+0.0000e+00j,
            1.6548e-13-2.5761e-13j],
          [ 4.1215e-02-1.0816e-02j,  6.1272e+01+0.0000e+00j,
            4.1215e-02+1.0816e-02j],
          [ 1.6526e-13+2.5749e-13j,  2.0087e+02+0.0000e+00j,
            1.6526e-13-2.5749e-13j],
          [ 2.3022e-02+3.5855e-02j,  6.1272e+01+0.0000e+00j,
            2.3022e-02-3.5855e-02j],
          [ 1.6547e-13+2.5770e-13j,  2.0087e+02+0.0000e+00j,
            1.6547e-13-2.5770e-13j]]],


        [[[ 2.7084e-01+1.4796e-01j,  1.7225e+01-5.5628e-17j,
            2.8313e-01-1.5467e-01j],
          [ 5.2570e-01+2.8719e-01j,  4.8584e+01+3.9192e-16j,
            6.5622e-01-3.5850e-01j],
          [ 2.6993e-01+1.1738e-01j,  1.7135e+01+1.7194e+00j,
            2.9872e-01-9.0423e-02j],
          [ 4.9440e-01+3.3824e-01j,  4.8341e+01+4.8503e+00j,
            6.8874e-01-2.9119e-01j],
          [ 1.2298e-01+2.6736e-01j,  1.5507e+01+7.4904e+00j,
            3.0823e-01-4.8532e-02j],
          [ 3.4845e-01+4.8726e-01j,  4.3748e+01+2.1132e+01j,
            7.4683e-01-3.7373e-02j]],

         [[ 6.9450e-02+3.7941e-02j,  4.8584e+01+5.0805e-16j,
            1.3569e-02-7.4126e-03j],
          [ 1.1883e-14+6.4915e-15j,  1.3708e+02+8.1722e-16j,
            2.8098e-12-1.5350e-12j],
          [ 7.9106e-02-2.2408e-03j,  4.8341e+01+4.8503e+00j,
            1.5060e-02+3.4995e-03j],
          [ 1.1139e-14+7.7394e-15j,  1.3639e+02+1.3685e+01j,
            2.9490e-12-1.2470e-12j],
          [-5.9529e-04+7.9136e-02j,  4.3748e+01+2.1132e+01j,
            1.2039e-02-9.7018e-03j],
          [ 7.9335e-15+1.0800e-14j,  1.2343e+02+5.9623e+01j,
            3.1981e-12-1.5976e-13j]],

         [[ 2.9316e-01+2.6402e-02j,  1.7135e+01-1.7194e+00j,
            3.0550e-01-6.3859e-02j],
          [ 5.9604e-01-5.9803e-02j,  4.8341e+01-4.8503e+00j,
            7.4403e-01-7.4652e-02j],
          [ 3.0862e-01-1.9806e-18j,  1.7225e+01-1.7761e-16j,
            3.2262e-01-2.6372e-18j],
          [ 5.9903e-01+4.8546e-19j,  4.8584e+01+9.2440e-17j,
            7.4776e-01-3.0738e-18j],
          [ 1.9531e-01+1.6663e-01j,  1.6173e+01+5.9040e+00j,
            2.8225e-01+4.6761e-02j],
          [ 5.6271e-01+2.0541e-01j,  4.5639e+01+1.6659e+01j,
            7.0243e-01+2.5641e-01j]],

         [[ 6.8348e-02+3.9892e-02j,  4.8341e+01-4.8503e+00j,
            1.1539e-02-1.0291e-02j],
          [ 1.3486e-14-1.4516e-15j,  1.3639e+02-1.3685e+01j,
            3.1858e-12-3.1950e-13j],
          [ 7.9138e-02-2.7286e-18j,  4.8584e+01+2.0851e-16j,
            1.5461e-02+5.7020e-18j],
          [ 1.3540e-14-1.2318e-22j,  1.3708e+02-7.6478e-16j,
            3.2017e-12+4.2798e-23j],
          [-2.8358e-03+7.9087e-02j,  4.5639e+01+1.6659e+01j,
            9.5304e-03-1.2175e-02j],
          [ 1.2804e-14+3.5534e-15j,  1.2877e+02+4.7003e+01j,
            3.0091e-12+1.0997e-12j]],

         [[ 2.7539e-01+1.0376e-01j,  1.5507e+01-7.4904e+00j,
            7.0214e-02-3.0403e-01j],
          [ 5.1069e-01+3.1311e-01j,  4.3748e+01-2.1132e+01j,
            9.0108e-02-7.4231e-01j],
          [ 2.4574e-01+7.4313e-02j,  1.6173e+01-5.9040e+00j,
            1.1315e-01-2.6277e-01j],
          [ 4.7688e-01+3.6253e-01j,  4.5639e+01-1.6659e+01j,
            1.6377e-01-7.2961e-01j],
          [ 1.6675e-01+2.5970e-01j,  1.7225e+01-1.2671e-16j,
            1.7431e-01-2.7148e-01j],
          [ 3.2366e-01+5.0407e-01j,  4.8584e+01+3.0920e-16j,
            4.0402e-01-6.2922e-01j]],

         [[ 7.8895e-02-6.1916e-03j,  4.3748e+01-2.1132e+01j,
            1.0529e-02-1.1322e-02j],
          [ 1.1334e-14+7.1497e-15j,  1.2343e+02-5.9623e+01j,
            3.8558e-13-3.1787e-12j],
          [ 6.5017e-02-4.5117e-02j,  4.5639e+01-1.6659e+01j,
            1.5394e-02-1.4414e-03j],
          [ 9.9079e-15+8.8541e-15j,  1.2877e+02-4.7003e+01j,
            7.0045e-13-3.1262e-12j],
          [ 4.2758e-02+6.6592e-02j,  4.8584e+01+4.0444e-16j,
            8.3538e-03-1.3010e-02j],
          [ 7.3158e-15+1.1394e-14j,  1.3708e+02+3.4013e-16j,
            1.7299e-12-2.6942e-12j]]]], dtype=torch.complex128)

    # # code to generate the matrix above
    # try:
    #     from pyscf.pbc.df.ft_ao import ft_aopair_kpts
    # except ImportError:
    #     # older version of PySCF
    #     from pyscf.pbc.df.ft_ao import _ft_aopair_kpts as ft_aopair_kpts
    #
    # # construct the pyscf system
    # cell = get_cell_pyscf(dtype, a.detach().numpy())
    # if int_type == "overlap":
    #     int_name = "GTO_ft_ovlp"
    # else:
    #     raise RuntimeError("Unknown int_type: %s" % int_type)
    #
    # mat_scf = ft_aopair_kpts(cell, gvgrid, kptjs=kpts.numpy(), intor=int_name)
    # mat_scf = np.rollaxis(mat_scf, 1, 4)
    # mat_scf_t = torch.as_tensor(mat_scf)
    # print(mat_scf_t)
    # raise RuntimeError

    # high rtol because mat_scf_t is precomputed
    assert torch.allclose(mat_ft, mat_scf_t, rtol=5e-5)

@pytest.mark.parametrize(
    "angmom",
    [0, 1]
)
def test_eval_ft_gto(angmom):
    # test the evaluation of fourier transform of the gto

    # setup the system
    alphas = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)[:, None]  # (nb, 1)
    coeffs = torch.tensor([1.5, 2.5, 0.5], dtype=dtype)[:, None]
    all_basis = [
        CGTOBasis(
            angmom = angmom,
            alphas = alphas[i],
            coeffs = coeffs[i],
            normalized = True,
        )
        for i in range(len(alphas))
    ]
    atom_basis = AtomCGTOBasis(
        atomz = 0,
        bases = all_basis,
        pos = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    )
    wrapper = intor.LibcintWrapper([atom_basis], spherical=True)
    ngrid = 100
    gvgrid = torch.zeros(ngrid, 3, dtype=dtype)  # (ngrid, ndim)
    gvgrid[:, 0] = torch.linspace(-5, 5, ngrid, dtype=dtype)
    if angmom == 0:
        c = np.sqrt(4 * np.pi)  # s-normalization
    elif angmom == 1:
        c = np.sqrt(4 * np.pi / 3)  # p-normalization
    ao_ft_value = intor.eval_gto_ft(wrapper, gvgrid) * c

    assert ao_ft_value.shape[0] == wrapper.nao()
    assert ao_ft_value.shape[1] == ngrid

    # compare with analytically calculated values
    exp_part = torch.exp(-gvgrid[:, 0] ** 2 / (4 * alphas))  # (ngrid,)
    if angmom == 0:
        true_value = coeffs * (np.pi / alphas) ** 1.5 * exp_part  # (nb, ngrid)
    elif angmom == 1:
        true_value = -1j * coeffs * np.pi ** 1.5 / 2. / alphas ** 2.5 * exp_part  # (nb, ngrid)
        true_value = true_value.unsqueeze(-2) * gvgrid.transpose(-2, -1)  # (ndim, nb, ngrid)
        true_value = true_value.reshape(-1, ngrid)
    true_value = true_value.to(torch.complex128)

    assert torch.allclose(true_value, ao_ft_value)

################## pbc eval ##################
@pytest.mark.parametrize(
    "eval_type",
    ["", "grad"]
)
def test_pbc_eval_gto_vs_pyscf(eval_type):
    # check if our eval_gto produces the same results as pyscf
    # also check the partial eval_gto

    basis = "6-311++G**"
    d = 0.8

    # setup the system for dqc
    atomenv = get_atom_env(dtype, ngrid=100)
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype) * 3
    kpts = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.15]], dtype=dtype)
    rgrid = atomenv.rgrid
    wrapper = get_wrapper(atomenv, spherical=True, lattice=Lattice(a))
    wrapper1 = wrapper[:len(wrapper)]
    if eval_type == "":
        # (nkpts, nao, ngrid)
        ao_value = intor.pbc_eval_gto(wrapper, rgrid)
        ao_value1 = intor.pbc_eval_gto(wrapper1, rgrid)
    elif eval_type == "grad":
        # (ndim, nkpts, nao, ngrid)
        ao_value = intor.pbc_eval_gradgto(wrapper, rgrid)
        ao_value1 = intor.pbc_eval_gradgto(wrapper1, rgrid)

    # (*ncomp, nao, ngrid)
    wkpts = 1.0 / len(kpts)
    ao_value = ao_value.sum(dim=-3) * wkpts
    ao_value1 = ao_value1.sum(dim=-3) * wkpts

    # check the partial eval_gto
    assert torch.allclose(ao_value[..., :len(wrapper1), :], ao_value1)

    # system in pyscf
    cell = get_cell_pyscf(dtype, a.detach().numpy())

    coords_np = rgrid.detach().numpy()
    if eval_type == "":
        ao_value_scf = cell.pbc_eval_gto("GTOval_sph", coords_np)
    elif eval_type == "grad":
        # (ndim, ngrid, nao)
        ao_value_scf = cell.pbc_eval_gto("GTOval_ip_sph", coords_np)
    ao_value_scf = torch.as_tensor(ao_value_scf).transpose(-2, -1).to(ao_value.dtype)
    ao_value_scf = ao_value_scf * wkpts
    print(ao_value_scf.shape, ao_value.shape)

    assert torch.allclose(ao_value, ao_value_scf, atol=1e-5)

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
