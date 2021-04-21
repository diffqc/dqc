import time
from pyscf import gto, scf

def get_molecule(molname):
    basis = "321g"
    spin = 0
    if molname == "H2":
        atom_desc = "H -0.5 0 0; H 0.5 0 0"
    elif molname == "Li2":
        atom_desc = "Li -2.5 0 0; Li 2.5 0 0"
    elif molname == "N2":
        atom_desc = "N -1 0 0; N 1 0 0"
    elif molname == "O2":
        atom_desc = "O -1 0 0; O 1 0 0"
        spin = 2
    elif molname == "NO":
        atom_desc = "N -1 0 0; O 1 0 0"
        spin = 1
    elif molname == "CO":
        atom_desc = "C -1 0 0; O 1 0 0"
    elif molname == "F2":
        atom_desc = "F -1.25 0 0; F 1.25 0 0"
    else:
        raise RuntimeError("Unknown molecule %s" % molname)

    mol = gto.M(atom=atom_desc, basis=basis, unit="Bohr", spin=spin)
    return mol

def get_atom(atom, spin):
    basis = "321g"
    mol = gto.M(atom="%s 0 0 0"%atom, spin=spin, basis=basis, unit="Bohr")
    return mol

def get_molecules_energy():
    molnames = ["H2", "Li2", "N2", "NO", "CO", "F2", "O2"]
    for molname in molnames:
        t0 = time.time()
        mol = get_molecule(molname)
        mf = scf.UHF(mol)
        energy = mf.kernel()
        t1 = time.time()
        print("Molecule %s: %.8e (%.3e)" % (molname, energy, t1-t0))

def get_atoms_energy():
    atoms = [
        ("H", 1),
        ("He", 0),
        ("Li", 1),
        ("Be", 0),
        ("B", 1),
        ("O", 2),
        ("Ne", 0),
    ]
    for atomname, spin in atoms:
        t0 = time.time()
        atom = get_atom(atomname, spin)
        mf = scf.UHF(atom)
        energy = mf.kernel()
        t1 = time.time()
        print("Atom %s: %.8e (%.3e)" % (atomname, energy, t1-t0))

if __name__ == "__main__":
    get_molecules_energy()
    # get_atoms_energy()
