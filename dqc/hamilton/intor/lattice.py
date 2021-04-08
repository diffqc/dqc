from typing import Tuple
import torch
import numpy as np
from scipy.special import erfcinv

class Lattice(object):
    """
    Lattice is an object that describe the periodicity of the lattice.
    Note that this object does not know about atoms.
    For the integrated object between the lattice and atoms, please see MolPBC
    """
    def __init__(self, a: torch.Tensor):
        # 2D or 1D repetition are not implemented yet
        assert a.ndim == 2
        assert a.shape[0] == 3
        assert a.shape[-1] == 3
        self.ndim = a.shape[0]
        self.a = a
        self.device = a.device
        self.dtype = a.dtype

    def lattice_vectors(self) -> torch.Tensor:
        """
        Returns the 3D lattice vectors (nv, ndim) with nv == 3
        """
        return self.a

    def recip_vectors(self) -> torch.Tensor:
        """
        Returns the 3D reciprocal vectors with norm == 2 * pi with shape (nv, ndim)
        with nv == 3
        """
        return torch.inverse(self.a.transpose(-2, -1)) * (2 * np.pi)

    def volume(self) -> torch.Tensor:
        """
        Returns the volume of a lattice.
        """
        return torch.det(self.a)

    @property
    def params(self) -> Tuple[torch.Tensor, ...]:
        """
        Returns the list of parameters of this object
        """
        return (self.a,)

    def get_lattice_ls(self, rcut: float, exclude_zeros: bool = False) -> torch.Tensor:
        """
        Returns a tensor that contains the coordinates of the neighboring
        lattices.

        Arguments
        ---------
        rcut: float
            The threshold of the distance from the main cell to be included
            in the neighbor.
        exclude_zeros: bool
            If True, then it will exclude the vector that are all zeros.

        Returns
        -------
        ls: torch.Tensor
            Tensor with size `(nb, ndim)` containing the coordinates of the
            neighboring cells.
        """
        # largely inspired by pyscf:
        # https://github.com/pyscf/pyscf/blob/e6c569932d5bab5e49994ae3dd365998fc5202b5/pyscf/pbc/tools/pbc.py#L473

        a = self.lattice_vectors()
        b = self.recip_vectors() / (2 * np.pi)  # (nv, ndim)
        heights_inv = torch.max(torch.norm(b, dim=-1)).detach().numpy()  # scalar
        nimgs = int(rcut * heights_inv + 1.1)

        assert isinstance(nimgs, int)
        n1_0 = torch.arange(-nimgs, nimgs + 1, dtype=torch.int32, device=self.device)  # (nimgs2,)
        ls = n1_0[:, None] * a[0, :]  # (nimgs2, ndim)
        ls = ls + n1_0[:, None, None] * a[1, :]  # (nimgs2, nimgs2, ndim)
        ls = ls + n1_0[:, None, None, None] * a[2, :]  # (nimgs2, nimgs2, nimgs2, ndim)
        ls = ls.view(-1, ls.shape[-1])  # (nb, ndim)

        if exclude_zeros:
            ls = ls[torch.any(ls != 0, dim=-1), :]

        return ls

    def get_gvgrids(self, gcut: float, exclude_zeros: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tensor that contains the coordinate in reciprocal space of the
        neighboring Brillouin zones.

        Arguments
        ---------
        gcut: float
            Cut off for generating the G-points.
        exclude_zeros: bool
            If True, then it will exclude the vector that are all zeros.

        Returns
        -------
        gvgrids: torch.Tensor
            Tensor with size `(ng, ndim)` containing the G-coordinates of the
            Brillouin zones.
        weights: torch.Tensor
            Tensor with size `(ng)` representing the weights of the G-points.
        """
        a = self.lattice_vectors()
        heights = torch.max(torch.norm(a, dim=-1)).detach().numpy() / (2 * np.pi)  # scalar
        ng1 = int(gcut * heights + 1.1)

        # generate the frequency data points
        ng1 = 2 * ng1 + 1
        rx = torch.as_tensor(np.fft.fftfreq(ng1, 1.0 / ng1), dtype=self.dtype, device=self.device)  # (ng1,)

        # TODO: check if it is b[i, :] or b[:, i]
        b = self.recip_vectors()  # (ndim, ndim)
        gvgrids = rx[:, None] * b[0, :]  # (ng1, ndim)
        gvgrids = gvgrids + rx[:, None, None] * b[1, :]  # (ng1, ng1, ndim)
        gvgrids = gvgrids + rx[:, None, None, None] * b[2, :]  # (ng1, ng1, ng1, ndim)
        gvgrids = gvgrids.view(-1, gvgrids.shape[-1])  # (ng, ndim)

        # 1 / cell.vol == det(b) / (2 pi)^3
        weights = torch.zeros(gvgrids.shape[0], dtype=self.dtype, device=self.device)
        weights = weights + torch.abs(torch.det(b)) / (2 * np.pi) ** 3

        if exclude_zeros:
            idx = torch.any(gvgrids != 0, dim=-1)
            gvgrids = gvgrids[idx, :]
            weights = weights[idx]

        return gvgrids, weights

    def estimate_ewald_eta(self, precision: float) -> float:
        # estimate the ewald's sum eta for nuclei interaction energy
        # this is from Martin's electronic structure appendix F after F.4
        # eta = Gv_min
        sqrt_pi = np.sqrt(np.pi)
        vol = float(self.volume().detach())
        eta0 = (2 * np.pi / vol ** (2. / 3) / 2) ** .5
        eta = eta0
        for _ in range(1):
            eta2 = erfcinv(vol * eta * eta * precision / (2 * np.pi)) \
                   / erfcinv(precision * sqrt_pi / 2 / eta)
            eta = eta0 * np.sqrt(eta2)
        return round(eta * 10) / 10  # round to 1 d.p.
