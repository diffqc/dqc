# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import dqc
import pyscf.dft

# define molecule descriptions for benchmarking
moldesc_h2o = "O 0.0000 0.0000 0.2217; H 0.0000 1.4309 -0.8867; H 0.0000 -1.4309 -0.8867"  # from CCCBDB, exp data
moldesc_c4h5n = """N 0.0000 0.0000 2.1199; H 0.0000 0.0000 4.0021; C 0.0000 2.1182 0.6314;
        C 0.0000 -2.1182 0.6314; C 0.0000 1.3372 -1.8608; C 0.0000 -1.3372 -1.8608;
        H 0.0000 3.9843 1.4388; H 0.0000 -3.9843 1.4388; H 0.0000 2.5636 -3.4826;
        H 0.0000 -2.5636 -3.4826"""

# create moldesc dictionary so parameter names are easy to understand
molecules = {'H2O': moldesc_h2o, 'C4H5N': moldesc_c4h5n}
molecule_names = ['H2O', 'C4H5N']


class TimeDQC:
    """
    Benchmark engery calculations for DQC with cc-pvdz
    """
    params = molecule_names
    param_names = ['molecule']

    def setup(self, molecule_name):
        moldesc = molecules[molecule_name]
        self.m = dqc.Mol(moldesc, basis="cc-pvdz")

    def time_energy_HF(self, _):
        # Hartree-Fock
        dqc.HF(self.m).run().energy()

    def time_energy_LDA(self, _):
        # LDA
        # QUESTION: do I need to worry about grid level 4?
        dqc.KS(self.m, xc="lda_x+lda_c_pw").run().energy()


class TimePySCF:
    """
    Benchmark engery calculations for PySCF with cc-pvdz
    """
    params = molecule_names
    param_names = ['molecule']

    def setup(self, molecule_name):
        moldesc = molecules[molecule_name]
        self.m = pyscf.gto.M(atom=moldesc, basis="cc-pvdz", unit="Bohr")

    def time_energy_HF(self, _):
        # Hartree-Fock
        pyscf.scf.RHF(self.m).kernel()

    def time_energy_LDA(self, _):
        # LDA
        mf = pyscf.dft.RKS(self.m, xc="lda_x+lda_c_pw")
        mf.grids.level = 4
        mf.kernel()