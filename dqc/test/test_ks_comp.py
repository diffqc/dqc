import numpy as np
import torch
import pytest
import itertools
from dqc.qccalc.ks import KS
from dqc.system.mol import Mol
from dqc.utils.datastruct import SpinParam

# check the components of KS against PySCF

BASIS = "6-31G*"
DTYPE = torch.double

def pscfelrepxc(atom: str, spin: int = 0, xc: str = "lda_x", basis: str = BASIS):
    # returns the electron repulsion and xc operator using pyscf
    from pyscf import gto, dft
    mol = gto.M(atom=atom, spin=spin, unit="Bohr", basis=basis)
    m = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    m.xc = xc
    m.grids.level = 4
    # set dm to be an identity matrix
    dm = np.eye(mol.nao)
    if spin != 0:
        dm = np.concatenate((dm[None, :, :], dm[None, :, :]), axis=0)
    return np.asarray(m.get_veff(dm=dm))

def dqcelrepxc(atom: str, spin: int = 0, xc: str = "lda_x", basis: str = BASIS):
    # returns the electron repulsion and xc operator using DQC
    mol = Mol(atom, spin=spin, basis=basis, dtype=DTYPE, grid=4)
    qc = KS(mol, xc=xc)
    hamilt = mol.get_hamiltonian()
    if spin == 0:
        # set dm to be an identity matrix
        dm = torch.eye(hamilt.nao, dtype=DTYPE)
        velrepxc = hamilt.get_vxc(dm) + hamilt.get_elrep(dm)
        return velrepxc.fullmatrix()
    else:
        dmu = torch.eye(hamilt.nao, dtype=DTYPE)
        dm = SpinParam(u=dmu, d=dmu)
        vxc = hamilt.get_vxc(dm)
        elrep = hamilt.get_elrep(dm.u + dm.d)
        return torch.cat(((vxc.u + elrep).fullmatrix().unsqueeze(0),
                          (vxc.d + elrep).fullmatrix().unsqueeze(0)), dim=0)

moldescs_spins = [
    ("H -0.5 0 0; H 0.5 0 0", 0),
    ("H 0.0 0 0", 1),

    # ("Li -2.5 0 0; Li 2.5 0 0", 0),
    # ("F -1.25 0 0; F 1.25 0 0", 0),
    # ("O -1.0 0 0; O 1.0 0 0", 2),
]

@pytest.mark.parametrize(
    "dqc_xc,pscf_xc,moldesc,spin",
    [(dqc_xc, pscf_xc, moldesc, spin) for ((moldesc, spin), (dqc_xc, pscf_xc)) in \
        itertools.product(moldescs_spins, [
                          ("lda_x", "lda_x"),
                          ("lda_x + lda_c_pw", "lda_x,lda_c_pw"),
                          ("gga_x_pbe", "gga_x_pbe"),
                          ("gga_x_pbe + gga_c_pbe", "gga_x_pbe,gga_c_pbe")])]
)
def test_ks_velrepxc(dqc_xc, pscf_xc, moldesc, spin):
    print(dqc_xc, pscf_xc, moldesc, spin)
    elrepxc1 = torch.as_tensor( dqcelrepxc(moldesc, spin=spin, xc=dqc_xc))
    elrepxc2 = torch.as_tensor(pscfelrepxc(moldesc, spin=spin, xc=pscf_xc))

    # compare the eigenvalues because the columns and rows can be swapped
    assert torch.allclose(torch.linalg.eigh(elrepxc1)[0], torch.linalg.eigh(elrepxc2)[0])
