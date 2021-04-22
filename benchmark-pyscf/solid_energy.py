import time
import numpy as np
from pyscf.pbc import gto, dft

def get_solid(solname):
    basis = "321g"
    rcut = None
    if solname == "Li":
        atom_desc = "Li 0 0 0"
        alattice = np.array([[1., 1., -1.], [-1., 1., 1.], [1., -1., 1.]]) * 0.5 * 6.6329387300636
        spin = 1
        rcut = 45.4
    elif solname == "H":
        atom_desc = "H 0 0 0"
        alattice = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) * 3.0
        spin = 1
    else:
        raise RuntimeError("Unknown solid %s" % solname)

    sol = gto.C(atom=atom_desc, basis=basis, unit="Bohr", spin=spin, a=alattice)
    if rcut is not None:
        sol.rcut = rcut
    return sol

def get_solids_energy(xc="lda", with_df=False):
    solnames = ["H"]
    for solname in solnames:
        t0 = time.time()
        sol = get_solid(solname)
        mf = dft.KUKS(sol, kpts=np.zeros(3))
        if with_df:
            auxbasis = "def2-svp-jkfit"
            mf = mf.density_fit(auxbasis=auxbasis)
        mf.xc = xc
        mf.grids.level = 4
        energy = mf.kernel()
        t1 = time.time()
        print("Solid %s: %.8e (%.3e)" % (solname, energy, t1-t0))

if __name__ == "__main__":
    get_solids_energy(xc="gga_x_pbe", with_df=True)
