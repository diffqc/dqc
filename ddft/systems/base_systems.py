from abc import abstractmethod, abstractproperty

class BaseSystem(object):
    @abstractproperty
    def dtype(self):
        pass

    @abstractproperty
    def device(self):
        pass

    @abstractproperty
    def atomzs(self):
        """
        Returns a tensor of the z-number of the atoms in the system.
        Return shape: (natoms,)
        """
        pass

    @abstractproperty
    def atomposs(self):
        """
        Returns a tensor of the positions of the atoms in the system.
        Return shape: (natoms, ndim=3)
        """
        pass

    @abstractmethod
    def get_nuclei_energy(self):
        """
        Return the energy of nuclei interactions.
        Return shape: (,)
        """
        pass

    @abstractmethod
    def get_numel(self, split=False):
        """
        Return the number of electrons. If split, returns 2-elements tuple:
        (n_up, n_dn). Otherwise, just return n_up+n_dn
        """
        pass

    @abstractmethod
    def get_grid_pts(self, with_weights=False):
        """
        Returns the grid points (or, optionally, with weights) for integration.

        Arguments
        ---------
        * with_weights: bool
            If True, returns points as well as the integration weights: (pts, wts).
            Otherwise, just return the points
        """
        pass

    @abstractmethod
    def _get_grid(self):
        pass

    @abstractmethod
    def _get_hamiltonian(self):
        pass
