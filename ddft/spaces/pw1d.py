import torch
from ddft.spaces.base_space import BaseSpace
from ddft.transforms.base_transform import SymmetricTransform
from ddft.utils.tensor import roll_1

class PlaneWave1D(BaseSpace):
    def __init__(self, length, emax, vext, rgrid=None):
        self.length = length
        self.emax = emax
        self.dx = self.length / self.emax # TODO: correct this

        # TODO: interpolate the vext
        self._vext = vext

        # get the transformation objects
        self._kinetics = _Kinetics1DPlaneWave(self.dx)
        self._hamiltonian = _Hamiltonian1DPlaneWave(self._kinetics, self._vext)

    @property
    def K(self):
        return self._kinetics

    @property
    def H(self):
        return self._hamiltonian

class _Kinetics1DPlaneWave(SymmetricTransform):
    def __init__(self, dx):
        self.inv_dx2 = 1.0/(dx * dx)

    def __call__(self, x):
        # x is (nbatch, nr)
        # TODO: do the proper FT, Q^2, IFT
        ddx = x - (roll_1(x, 1) + roll_1(x, -1)) * 0.5
        return ddx * self.inv_dx2

class _Hamiltonian1DPlaneWave(SymmetricTransform):
    def __init__(self, kinetics, vext):
        self._kinetics = kinetics
        self._vext = vext

    def __call__(self, x):
        return self._kinetics(x) + self._vext
