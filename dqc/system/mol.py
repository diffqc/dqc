from typing import List, Union, Optional, Tuple
import torch
import numpy as np
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.hamilton.hamilton_cgto import HamiltonCGTO
from dqc.system.base_system import BaseSystem
from dqc.grid.base_grid import BaseGrid
from dqc.grid.radial_grid import RadialGrid
from dqc.grid.lebedev_grid import LebedevGrid
from dqc.grid.becke_grid import BeckeGrid
from dqc.utils.datastruct import CGTOBasis, AtomCGTOBasis
from dqc.utils.periodictable import get_atomz
from dqc.api.loadbasis import loadbasis

AtomZType   = Union[List[str], List[int], torch.Tensor]
AtomPosType = Union[List[List[float]], np.array, torch.Tensor]

class Mol(BaseSystem):
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

    def __init__(self,
                 moldesc: Union[str, Tuple[AtomZType, AtomPosType]],
                 basis: Union[str, List[CGTOBasis], List[str], List[List[CGTOBasis]]],
                 grid: int = 4,
                 spin: Optional[int] = None,
                 charge: int = 0,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu'),
                 ):
        self._dtype = dtype
        self._device = device
        self._grid_inp = grid
        self._grid: Optional[BaseGrid] = None

        # get the AtomCGTOBasis & the hamiltonian
        # atomzs: (natoms,)
        # atompos: (natoms, ndim)
        atomzs, atompos = _parse_moldesc(moldesc, dtype, device)
        allbases = _parse_basis(atomzs, basis)  # list of list of CGTOBasis
        atombases = [AtomCGTOBasis(atomz=atz, bases=bas, pos=atpos)
                     for (atz, bas, atpos) in zip(atomzs, allbases, atompos)]
        self._hamilton = HamiltonCGTO(atombases)
        self._atompos = atompos  # (natoms, ndim)

        # get the orbital weights
        nelecs: int = int(torch.sum(atomzs).item()) - charge
        if spin is None:
            spin = nelecs % 2
        assert spin >= 0
        assert (nelecs - spin) % 2 == 0, \
            "Spin %d is not suited for %d electrons" % (spin, nelecs)
        nspin_dn = (nelecs - spin) // 2
        nspin_up = nspin_dn + spin
        _orb_weights = torch.ones((nspin_up,), dtype=dtype, device=device)
        _orb_weights[:nspin_dn] = 2.0
        self._orb_weights = _orb_weights

    def get_hamiltonian(self) -> BaseHamilton:
        return self._hamilton

    def get_orbweight(self) -> torch.Tensor:
        return self._orb_weights

    def setup_grid(self) -> None:
        grid_inp = self._grid_inp
        #        0,  1,  2,  3,  4,  5
        nr   = [20, 40, 60, 75, 100, 125][grid_inp]
        prec = [13, 17, 21, 29, 41, 59][grid_inp]
        radgrid = RadialGrid(nr, "chebyshev", "logm3",
                             dtype=self._dtype, device=self._device)
        sphgrid = LebedevGrid(radgrid, prec=prec)

        natoms = self._atompos.shape[-2]
        sphgrids = [sphgrid for _ in range(natoms)]
        self._grid = BeckeGrid(sphgrids, self._atompos)

    def get_grid(self) -> BaseGrid:
        if self._grid is None:
            raise RuntimeError("Please run mol.setup_grid() first before calling get_grid()")
        return self._grid

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass

def _parse_moldesc(moldesc: Union[str, Tuple[AtomZType, AtomPosType]],
                   dtype: torch.dtype,
                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(moldesc, str):
        # TODO: use regex!
        elmts = [
            [
                get_atomz(c.strip()) if i == 0 else float(c.strip())
                for i, c in enumerate(line.split())
            ] for line in moldesc.split(";")]
        atomzs = torch.tensor([line[0] for line in elmts], device=device)
        atomposs = torch.tensor([line[1:] for line in elmts], dtype=dtype, device=device)
        return atomzs, atomposs

    else:  # tuple of atomzs, atomposs
        atomzs_raw, atompos_raw = moldesc
        assert len(atomzs_raw) == len(atompos_raw), "Mismatch length of atomz and atompos"
        assert len(atomzs_raw) > 0, "Empty atom list"

        # convert the atomz to tensor
        if not isinstance(atomzs_raw, torch.Tensor):
            atomzs = torch.tensor([get_atomz(at) for at in atomzs_raw])
        else:
            atomzs = atomzs_raw  # already a tensor

        # convert the atompos to tensor
        if not isinstance(atompos_raw, torch.Tensor):
            atompos = torch.as_tensor(atompos_raw, dtype=dtype, device=device)
        else:
            atompos = atompos_raw  # already a tensor

        return atomzs, atompos

def _parse_basis(atomzs: torch.Tensor,
                 basis: Union[str, List[CGTOBasis], List[str], List[List[CGTOBasis]]]) -> \
        List[List[CGTOBasis]]:
    # returns the list of cgto basis for every atoms
    natoms = len(atomzs)

    if isinstance(basis, str):
        return [loadbasis("%d:%s" % (atomz, basis)) for atomz in atomzs]

    else:  # basis is a list
        assert len(atomzs) == len(basis)

        # TODO: safely remove "type: ignore" in this block
        assert len(basis) > 0

        # list of cgtobasis
        if isinstance(basis[0], CGTOBasis):
            return [basis for _ in range(natoms)]  # type: ignore

        # list of str
        elif isinstance(basis[0], str):
            return [loadbasis("%d:%s" % (atz, b)) for (atz, b) in zip(atomzs, basis)]  # type: ignore

        # list of list of cgto basis
        else:
            return basis  # type: ignore
