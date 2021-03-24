from typing import List, Union, Optional, Tuple
import torch
import numpy as np
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.hamilton.hcgto import HamiltonCGTO
from dqc.system.base_system import BaseSystem
from dqc.grid.base_grid import BaseGrid
from dqc.grid.factory import get_grid
from dqc.utils.datastruct import CGTOBasis, AtomCGTOBasis, SpinParam, ZType, \
                                 is_z_float, BasisInpType
from dqc.utils.periodictable import get_atomz
from dqc.utils.safeops import eps as util_eps, occnumber
from dqc.api.loadbasis import loadbasis

AtomZsType  = Union[List[str], List[ZType], torch.Tensor]
AtomPosType = Union[List[List[float]], np.ndarray, torch.Tensor]

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
                 basis: BasisInpType,
                 grid: Union[int, str] = "sg3",
                 spin: Optional[ZType] = None,
                 charge: ZType = 0,
                 orb_weights: Optional[SpinParam[torch.Tensor]] = None,
                 *,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu'),
                 ):
        self._dtype = dtype
        self._device = device
        self._grid_inp = grid
        self._grid: Optional[BaseGrid] = None

        # get the AtomCGTOBasis & the hamiltonian
        # atomzs: (natoms,) dtype: torch.int or dtype for floating point
        # atompos: (natoms, ndim)
        atomzs, atompos = _parse_moldesc(moldesc, dtype, device)
        atomzs_int = torch.round(atomzs).to(torch.int) if atomzs.is_floating_point() else atomzs
        allbases = _parse_basis(atomzs_int, basis)  # list of list of CGTOBasis
        atombases = [AtomCGTOBasis(atomz=atz, bases=bas, pos=atpos)
                     for (atz, bas, atpos) in zip(atomzs, allbases, atompos)]
        self._hamilton = HamiltonCGTO(atombases)
        self._atompos = atompos  # (natoms, ndim)
        self._atomzs = atomzs  # (natoms,) int-type or dtype if floating point
        self._atomzs_int = atomzs_int  # (natoms,) int-type rounded from atomzs
        nelecs_tot: torch.Tensor = torch.sum(atomzs)

        # orb_weights is not specified, so determine it from spin and charge
        if orb_weights is None:
            # get the number of electrons and spin
            nelecs, spin, frac_mode = _get_nelecs_spin(nelecs_tot, spin, charge)
            _orb_weights, _orb_weights_u, _orb_weights_d = _get_orb_weights(
                nelecs, spin, frac_mode, dtype, device)

            # save the system's properties
            self._spin = spin
            self._charge = charge
            self._numel = nelecs
            self._orb_weights = _orb_weights
            self._orb_weights_u = _orb_weights_u
            self._orb_weights_d = _orb_weights_d

        # orb_weights is specified, so calculate the spin and charge from it
        else:
            assert isinstance(orb_weights, SpinParam)
            assert orb_weights.u.ndim == 1
            assert orb_weights.d.ndim == 1
            assert len(orb_weights.u) == len(orb_weights.d)

            utot = orb_weights.u.sum()
            dtot = orb_weights.d.sum()
            self._numel = utot + dtot
            self._spin = utot - dtot
            self._charge = nelecs_tot - self._numel

            self._orb_weights_u = orb_weights.u
            self._orb_weights_d = orb_weights.d
            self._orb_weights = orb_weights.u + orb_weights.d

    def get_hamiltonian(self) -> BaseHamilton:
        return self._hamilton

    def get_orbweight(self, polarized: bool = False) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        if not polarized:
            return self._orb_weights
        else:
            return SpinParam(u=self._orb_weights_u, d=self._orb_weights_d)

    def get_nuclei_energy(self) -> torch.Tensor:
        # atomzs: (natoms,)
        # atompos: (natoms, ndim)
        r12_pair = self._atompos.unsqueeze(-3) - self._atompos.unsqueeze(-2)  # (natoms, natoms, ndim)
        # add the diagonal with a small eps to safeguard from nan
        r12_pair = r12_pair + \
            torch.eye(r12_pair.shape[-2], dtype=self._dtype, device=self._device).unsqueeze(-1) * util_eps
        r12 = r12_pair.norm(dim=-1)  # (natoms, natoms)
        z12 = self._atomzs.unsqueeze(-2) * self._atomzs.unsqueeze(-1)  # (natoms, natoms)
        infdiag = torch.eye(r12.shape[0], dtype=r12.dtype, device=r12.device)
        idiag = infdiag.diagonal()
        idiag[:] = float("inf")
        r12 = r12 + infdiag
        q_by_r = z12 / r12
        return q_by_r.sum() * 0.5

    def setup_grid(self) -> None:
        grid_inp = self._grid_inp
        self._grid = get_grid(self._grid_inp, self._atomzs_int, self._atompos,
                              dtype=self._dtype, device=self._device)

        # #        0,  1,  2,  3,  4,  5
        # nr   = [20, 40, 60, 75, 100, 125][grid_inp]
        # prec = [13, 17, 21, 29, 41, 59][grid_inp]
        # radgrid = RadialGrid(nr, "chebyshev", "logm3",
        #                      dtype=self._dtype, device=self._device)
        # sphgrid = LebedevGrid(radgrid, prec=prec)
        #
        # natoms = self._atompos.shape[-2]
        # sphgrids = [sphgrid for _ in range(natoms)]
        # self._grid = BeckeGrid(sphgrids, self._atompos)

    def get_grid(self) -> BaseGrid:
        if self._grid is None:
            raise RuntimeError("Please run mol.setup_grid() first before calling get_grid()")
        return self._grid

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass

    @property
    def spin(self) -> ZType:
        return self._spin

    @property
    def charge(self) -> ZType:
        return self._charge

    @property
    def numel(self) -> ZType:
        return self._numel

def _parse_moldesc(moldesc: Union[str, Tuple[AtomZsType, AtomPosType]],
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
        atompos = torch.tensor([line[1:] for line in elmts], dtype=dtype, device=device)

    else:  # tuple of atomzs, atomposs
        atomzs_raw, atompos_raw = moldesc
        assert len(atomzs_raw) == len(atompos_raw), "Mismatch length of atomz and atompos"
        assert len(atomzs_raw) > 0, "Empty atom list"

        # convert the atomz to tensor
        if not isinstance(atomzs_raw, torch.Tensor):
            atomzs = torch.tensor([get_atomz(at) for at in atomzs_raw], device=device)
        else:
            atomzs = atomzs_raw.to(device)  # already a tensor

        # convert the atompos to tensor
        if not isinstance(atompos_raw, torch.Tensor):
            atompos = torch.as_tensor(atompos_raw, dtype=dtype, device=device)
        else:
            atompos = atompos_raw.to(dtype).to(device)  # already a tensor

    # convert to dtype if atomzs is a floating point tensor, not an integer tensor
    if atomzs.is_floating_point():
        atomzs = atomzs.to(dtype)

    return atomzs, atompos

def _parse_basis(atomzs: torch.Tensor, basis: BasisInpType) -> List[List[CGTOBasis]]:
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

def _get_nelecs_spin(nelecs_tot: torch.Tensor, spin: Optional[ZType],
                     charge: ZType) -> Tuple[torch.Tensor, ZType, bool]:
    # get the number of electrons and spins

    # a boolean to indicate if working in a fractional mode
    frac_mode = nelecs_tot.is_floating_point() or is_z_float(charge) or \
        (spin is not None and is_z_float(spin))

    assert nelecs_tot >= charge, \
        "Only %f electrons, but needs %f charge" % (nelecs_tot.item(), charge)
    nelecs: torch.Tensor = nelecs_tot - charge

    # if spin is not given, then set it as the remainder if nelecs is an integer
    if spin is None:
        assert not frac_mode, \
            "Fraction case requires the spin argument to be specified"
        spin = nelecs % 2
    else:
        assert spin >= 0
        if not frac_mode:
            # only check if the calculation is not in fraction mode,
            # for fractional mode, unmatched spin is acceptable
            assert (nelecs - spin) % 2 == 0, \
                "Spin %d is not suited for %d electrons" % (spin, nelecs)
    return nelecs, spin, frac_mode

def _get_orb_weights(nelecs: torch.Tensor, spin: ZType, frac_mode: bool,
                     dtype: torch.dtype, device: torch.device) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # returns the orbital weights given the electronic information
    # (total orbital weights, spin-up orb weights, spin-down orb weights)

    # calculate the orbital weights
    nspin_dn: torch.Tensor = (nelecs - spin) * 0.5 if frac_mode else (nelecs - spin) // 2
    nspin_up: torch.Tensor = nspin_dn + spin

    # total orbital weights
    _orb_weights_u = occnumber(nspin_up, dtype=dtype, device=device)
    _orb_weights_d = occnumber(nspin_dn, n=len(_orb_weights_u), dtype=dtype, device=device)
    _orb_weights = _orb_weights_u + _orb_weights_d

    # get the polarized orbital weights
    if nspin_dn > 0:
        _orb_weights_d = occnumber(nspin_dn, dtype=dtype, device=device)
    else:
        _orb_weights_d = occnumber(0, n=1, dtype=dtype, device=device)

    return _orb_weights, _orb_weights_u, _orb_weights_d
