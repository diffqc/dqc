from typing import Optional, Tuple, Union, List
import torch
from dqc.system.base_system import BaseSystem
from dqc.grid.base_grid import BaseGrid
from dqc.system.mol import _parse_moldesc, _parse_basis, _get_nelecs_spin, \
                           _get_orb_weights, AtomZsType, AtomPosType
from dqc.utils.datastruct import CGTOBasis, AtomCGTOBasis, ZType
from dqc.hamilton.intor.lattice import Lattice

class MolPBC(BaseSystem):
    """
    Describe the system of an isolated molecule.

    Arguments
    ---------
    * moldesc: str or 2-elements tuple (atomzs, atompos)
        Description of the molecule system.
        If string, it can be described like "H 0 0 0; H 0.5 0.5 0.5".
        If tuple, the first element of the tuple is the Z number of the atoms while
        the second element is the position of the atoms.
    * basis: str, CGTOBasis or list of str or CGTOBasis
        The string describing the gto basis. If it is a list, then it must have
        the same length as the number of atoms.
    * grid: int
        Describe the grid.
        If it is an integer, then it uses the default grid with specified level
        of accuracy.
        Default: 3
    * spin: int, float, torch.Tensor, or None
        The difference between spin-up and spin-down electrons.
        It must be an integer or None.
        If None, then it is ``num_electrons % 2``.
        For floating point atomzs and/or charge, the ``spin`` must be specified.
        Default: None
    * charge: int, float, or torch.Tensor
        The charge of the molecule.
        Default: 0
    * orb_weights: SpinParam[torch.Tensor] or None
        Specifiying the orbital occupancy (or weights) directly. If specified,
        ``spin`` and ``charge`` arguments are ignored.
    * dtype: torch.dtype
        The data type of tensors in this class.
        Default: torch.float64
    * device: torch.device
        The device on which the tensors in this class are stored.
        Default: torch.device('cpu')
    """

    def __init__(self,
                 moldesc: Union[str, Tuple[AtomZsType, AtomPosType]],
                 alattice: torch.Tensor,
                 basis: Union[str, List[CGTOBasis], List[str], List[List[CGTOBasis]]],
                 grid: Union[int, str] = "sg3",
                 spin: Optional[ZType] = None,
                 *,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu'),
                 ):
        self._dtype = dtype
        self._device = device
        self._grid_inp = grid
        self._grid: Optional[BaseGrid] = None
        charge = 0  # we can't have charged solids for now

        # get the AtomCGTOBasis & the hamiltonian
        # atomzs: (natoms,) dtype: torch.int or dtype for floating point
        # atompos: (natoms, ndim)
        atomzs, atompos = _parse_moldesc(moldesc, dtype, device)
        allbases = _parse_basis(atomzs, basis)  # list of list of CGTOBasis
        atombases = [AtomCGTOBasis(atomz=atz, bases=bas, pos=atpos)
                     for (atz, bas, atpos) in zip(atomzs, allbases, atompos)]
        self._atompos = atompos  # (natoms, ndim)
        self._atomzs = atomzs  # (natoms,) int-type
        nelecs_tot: torch.Tensor = torch.sum(atomzs)

        # get the number of electrons and spin and orbital weights
        nelecs, spin, frac_mode = _get_nelecs_spin(nelecs_tot, spin, charge)
        assert not frac_mode, "Fractional Z mode for pbc is not supported"
        _orb_weights, _orb_weights_u, _orb_weights_d = _get_orb_weights(
            nelecs, spin, frac_mode, dtype, device)

        # save the system's properties
        self._spin = spin
        self._charge = charge
        self._numel = nelecs
        self._orb_weights = _orb_weights
        self._orb_weights_u = _orb_weights_u
        self._orb_weights_d = _orb_weights_d
        self._lattice = Lattice(alattice)
