import torch
from ddft.systems.base_systems import BaseSystem
from ddft.basissets.cartesian_cgto import CartCGTOBasis
from ddft.grids.radialgrid import LegendreLogM3RadGrid
from ddft.grids.sphangulargrid import Lebedev
from ddft.grids.multiatomsgrid import BeckeMultiGrid
from ddft.utils.misc import to_tensor
from ddft.utils.periodictable import get_atomz

__all__ = ["mol"]

class mol(BaseSystem):
    """
    Describe the system of an isolated molecule.

    Arguments
    ---------
    * moldesc: str or 2-elements tuple (List[str or int], List[List[float]])
        Description of the molecule system.
        If string, it can be described like "H 0 0 0; H 0.5 0.5 0.5".
        If tuple, the first element of the tuple is the Z number of the atoms while
        the second element is the position of the atoms.
    * basis: str
        The string describing the gto basis.
    * grid: int
        Describe the grid.
        If it is an integer, then it uses the default grid with specified level
        of accuracy.
        Default: 3
    * spin: int or None
        The difference between spin-up and spin-down electrons.
        It must be an integer or None.
        If None, then it is num_electrons % 2.
        Default: None
    * charge: int
        The charge of the molecule.
        Default: 0
    * dtype: torch.dtype
        The data type of tensors in this class.
        Default: torch.float64
    * device: torch.device
        The device on which the tensors in this class are stored.
        Default: torch.device('cpu')
    """
    def __init__(self, moldesc, basis, grid=3,
                 spin=None, charge=0, requires_grad=True,
                 dtype=torch.float64, device=torch.device('cpu')):
        self._dtype = dtype
        self._device = device

        # initial check of the parameters
        self._atomzs, self._atomposs = self.__eval_moldesc(moldesc) # tensors: (natoms,) and (natoms, ndim)
        if requires_grad:
            self._atomposs = self._atomposs.requires_grad_()
        assert type(basis) == str, "The basis must be a string."
        assert type(spin) == int or spin is None, \
            "The spin argument must be a non-negative integer or None."
        assert type(charge) == int, "The charge must be an integer."

        # setup the grid
        self.grid = self.__setup_grid(grid)

        # setup the basis and get the hamiltonian
        self.hamiltonian = self.__setup_basis_and_H(basis)

        # calculate the occupation number
        self.numel = self._atomzs.sum() - charge
        if spin is None:
            spin = self.numel % 2
        if (self.numel - spin) % 2 != 0:
            raise ValueError("Spin %d cannot be used with systems with %d electrons" % (spin, self.numel))
        self.n_dn = (self.numel - spin) * 0.5
        self.n_up = self.n_dn + spin

        # cache
        self.pp_energy = None

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def atomzs(self):
        return self._atomzs

    @property
    def atomposs(self):
        return self._atomposs

    def get_nuclei_energy(self):
        if self.pp_energy is None:
            # atomzs: (natoms,)
            # atompos: (natoms, ndim)
            r12 = (self._atomposs - self._atomposs.unsqueeze(1)).norm(dim=-1) # (natoms, natoms)
            z12 = self._atomzs * self._atomzs.unsqueeze(1) # (natoms, natoms)
            infdiag = torch.eye(r12.shape[0], dtype=r12.dtype, device=r12.device)
            idiag = infdiag.diagonal()
            idiag[:] = float("inf")
            r12 = r12 + infdiag
            self.pp_energy = (z12 / r12).sum() * 0.5
        return self.pp_energy

    def get_numel(self, split=False):
        if split:
            return (self.n_up, self.n_dn)
        else:
            return self.n_up + self.n_dn

    def get_grid_pts(self, with_weights=False):
        if with_weights:
             # (npts, ndim) and (npts,)
            return self.grid.rgrid, self.grid.get_dvolume()
        else:
            return self.grid.rgrid # (npts, ndim)

    ######################## functions for ddft objects ########################
    def _get_grid(self):
        return self.grid

    def _get_hamiltonian(self):
        return self.hamiltonian

    ######################## private functions ########################
    def __eval_moldesc(self, moldesc):
        if type(moldesc) == str:
            # TODO: use regex!
            elmts = [[get_atomz(c.strip()) if i == 0 else float(c.strip()) for i,c in enumerate(line.split())] for line in moldesc.split(";")]
            atomzs = [line[0] for line in elmts]
            atomposs = [line[1:] for line in elmts]
            moldesc = (atomzs, atomposs)

        if hasattr(moldesc, "__iter__") and len(moldesc) == 2:
            atomzs, atomposs = moldesc
            atomzs = to_tensor(atomzs).to(self.dtype).to(self.device)
            atomposs = to_tensor(atomposs).to(self.dtype).to(self.device)
            return atomzs, atomposs
        else:
            raise TypeError("The molecule descriptor must be a 2-elements-tuple")

    def __setup_grid(self, grid):
        assert type(grid) == int, "The grid must be an integer"
        # TODO: adjust this!
        # level: 0,  1,  2,  3,  4,  5
        nr   = [20, 40, 60, 75,100,125][grid]
        prec = [13, 17, 21, 29, 41, 59][grid]
        lmax = [ 2,  3,  4,  6,  7,  8][grid]

        # TODO: adjust the best ra and atomradius for BeckeMultiGrid
        radgrid = LegendreLogM3RadGrid(nr, ra=1., dtype=self.dtype, device=self.device)
        atomgrid = Lebedev(radgrid, prec, basis_maxangmom=lmax, dtype=self.dtype, device=self.device)
        grid = BeckeMultiGrid(atomgrid, self._atomposs, dtype=self.dtype, device=self.device)
        return grid

    def __setup_basis_and_H(self, basis):
        basis_mem = {}
        bases_list = []
        for atomz in self._atomzs:
            z = int(atomz)
            if z not in basis_mem:
                basisobj = CartCGTOBasis(z, basis, dtype=self.dtype, device=self.device)
                basis_mem[z] = basisobj
            bases_list.append(basis_mem[z])

        hamiltonian = basisobj.construct_hamiltonian(self.grid, bases_list, self._atomposs)
        return hamiltonian
