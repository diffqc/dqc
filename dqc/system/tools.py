from typing import Optional, Tuple
import torch
import numpy as np

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

    def get_lattice_ls(self, nimgs: Optional[int] = None, rcut: Optional[float] = None) -> torch.Tensor:
        """
        Returns a tensor that contains the coordinates of the neighboring
        lattices.

        Arguments
        ---------
        nimgs: Optional[int]
            Number of neighbors from every side (e.g. for 3D: up, down, left,
            right, front, back). If specified, the `rcut` is ignored.
        rcut: Optional[float]
            The threshold of the distance from the main cell to be included
            in the neighbor.

        Returns
        -------
        ls: torch.Tensor
            Tensor with size `(nb, ndim)` containing the coordinates of the
            neighboring cells.
        """
        # largely inspired by pyscf:
        # https://github.com/pyscf/pyscf/blob/e6c569932d5bab5e49994ae3dd365998fc5202b5/pyscf/pbc/tools/pbc.py#L473

        a = self.lattice_vectors()
        # TODO: do this properly
        if nimgs is None:
            assert rcut is not None, "At least one of nimgs or rcut must be specified"
            b = self.recip_vectors() / (2 * np.pi)  # (nv, ndim)
            heights_inv = torch.max(torch.norm(b, dim=-1)).detach().numpy()  # scalar
            nimgs = int(rcut * heights_inv + 1.1)

        assert isinstance(nimgs, int)
        n1_0 = torch.arange(-nimgs, nimgs + 1, dtype=torch.int32, device=self.device)  # (nimgs2,)
        ls = n1_0[:, None] * a[0, :]  # (nimgs2, ndim)
        ls = ls + n1_0[:, None, None] * a[1, :]  # (nimgs2, nimgs2, ndim)
        ls = ls + n1_0[:, None, None, None] * a[2, :]  # (nimgs2, nimgs2, nimgs2, ndim)
        ls = ls.view(-1, ls.shape[-1])  # (nb, ndim)
        return ls
