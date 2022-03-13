# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import dqc
import pyscf.dft
import torch


# create efield
efield = (torch.tensor([0, 0, 0], dtype=torch.double).requires_grad_(),)

# define molecule descriptions for benchmarking
moldesc_h2o = "O 0.0000 0.0000 0.2217; H 0.0000 1.4309 -0.8867; H 0.0000 -1.4309 -0.8867"  # from CCCBDB, exp data
moldesc_c4h5n = """N 0.0000 0.0000 2.1199; H 0.0000 0.0000 4.0021; C 0.0000 2.1182 0.6314;
        C 0.0000 -2.1182 0.6314; C 0.0000 1.3372 -1.8608; C 0.0000 -1.3372 -1.8608;
        H 0.0000 3.9843 1.4388; H 0.0000 -3.9843 1.4388; H 0.0000 2.5636 -3.4826;
        H 0.0000 -2.5636 -3.4826"""

# create moldesc dictionary so parameter names are easy to understand
molecules = {'H2O': moldesc_h2o, 'C4H5N': moldesc_c4h5n}
molecule_names = ['H2O', 'C4H5N']

# define the systems for benchmarking
systems = ['HF', 'LDA']


class TimeCalcDQC:
    """
    Benchmark calculations for dqc with cc-pvdz
    """
    params = (molecule_names, systems)
    param_names = ['molecule', 'system']

    def setup(self, molecule_name, _):
        moldesc = molecules[molecule_name]
        self.m = dqc.Mol(moldesc, basis="cc-pvdz", grid=4, efield=efield)

    def time_calc(self, _, system):
        if system == 'HF':
            # Hartree-Fock
            self.sys = dqc.HF(self.m).run()
        elif system == 'LDA':
            # LDA
            self.sys = dqc.KS(self.m, xc="lda_x+lda_c_pw").run()
        else:
            raise ValueError(f'Unrecognized system {system}, must be one of {systems}')


class TimeCalcPySCF:
    """
    Benchmark engery calculations for PySCF with cc-pvdz
    """
    params = (molecule_names, systems)
    param_names = ['molecule', 'system']

    def setup(self, molecule_name, _):
        moldesc = molecules[molecule_name]
        self.m = pyscf.gto.M(atom=moldesc, basis="cc-pvdz", unit="Bohr")

    def time_calc(self, _, system):
        if system == 'HF':
            # Hartree-Fock
            pyscf.scf.RHF(self.m).kernel()
        elif system == 'LDA':
            # LDA
            mf = pyscf.dft.RKS(self.m, xc="lda_x+lda_c_pw")
            mf.grids.level = 4
            mf.kernel()
        else:
            raise ValueError(f'Unrecognized system {system}, must be one of {systems}')


class TimePropertiesDQC:
    """
    Benchmark property calculations for dqc with cc-pvdz
    """
    params = (molecule_names, systems)
    param_names = ['molecule', 'system']

    def setup(self, molecule_name, system):
        moldesc = molecules[molecule_name]
        self.m = dqc.Mol(moldesc, basis="cc-pvdz", grid=4, efield=efield)

        if system == 'HF':
            # Hartree-Fock
            self.sys = dqc.HF(self.m).run()
        elif system == 'LDA':
            # LDA
            self.sys = dqc.KS(self.m, xc="lda_x+lda_c_pw").run()
        else:
            raise ValueError(f'Unrecognized system {system}, must be one of {systems}')

    def time_energy(self, _, __):
        # calculate energy
        self.sys.energy()

    def time_ir_spectrum(self, _, __):
        # calculate ir spectrum
        self.m.atompos.requires_grad_()
        dqc.ir_spectrum(self.sys, freq_unit='cm^-1')

    def time_optimal_geometry(self, _, __):
        # calculate optimal geometry
        self.m.atompos.requires_grad_()
        dqc.optimal_geometry(self.sys)

    def time_vibration(self, _, __):
        # calculate vibrational frequency
        self.m.atompos.requires_grad_()
        dqc.vibration(self.sys, freq_unit='cm^-1')