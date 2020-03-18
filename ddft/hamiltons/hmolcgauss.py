from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
import numpy as np
import lintorch as lt

from ddft.hamiltons.base_hamilton import BaseHamilton
from ddft.utils.spharmonics import spharmonics
from ddft.utils.gamma import incgamma

class HamiltonMoleculeCGauss(BaseHamilton):
    """
    HamiltonMoleculeCGauss represents the system of multiple atoms with
    all-electrons potential at the centre of coordinate. The chosen basis is
    contracted Cartesian Gaussian centered at given positions.
    As the basis depends on the position of each atom, the batch only works
    for molecules with the same positions.

    Arguments
    ---------
    * grid: BaseGrid
        The integration grid. It should be a spherical harmonics radial grid
        centered at the centre of coordinate.
    * ijks: torch.tensor int (nbasis, nelmts, 3)
        The power of xyz in the basis.
    * alphas: torch.tensor (nbasis, nelmts,)
        The Gaussian exponent of the contracted Gaussians.
    * centres: torch.tensor (nbasis, nelmts, 3)
        The position in Cartesian coordinate of the contracted Gaussians.
    * coeffs: torch.tensor (nbasis, nelmts,)
        The contracted coefficients of each elements in the contracted Gaussians.
    * atompos: torch.tensor (natoms, 3)
        The position of the atoms (to be the central position of Coulomb potential).
    * atomzs: torch.tensor (natoms,)
        The atomic number of the atoms.

    Forward arguments
    -----------------
    * wf: torch.tensor (nbatch, nbasis, ncols)
        The basis coefficients of the wavefunction.
    * vext: torch.tensor (nbatch, nr)
        External potential other than the potential from the atoms.

    Overlap arguments
    -----------------
    * wf: torch.tensor (nbatch, nbasis, ncols)
        The basis coefficients of the wavefunction.
    """

    def __init__(self, grid,
                 ijks, alphas, centres, coeffs,
                 atompos, atomzs):
        self.nbasis, self.nelmts, ndim = centres.shape
        assert ndim == 3, "The centres must be 3 dimensions"
        self.natoms = atompos.shape[0]
        self._grid = grid
        dtype = alphas.dtype
        device = alphas.device
        super(HamiltonMoleculeCGauss, self).__init__(
            shape = (self.nbasis, self.nbasis),
            is_symmetric = True,
            is_real = True,
            dtype = dtype,
            device = device)

        # flatten the information about the basis
        ijks = ijks.view(-1, ndim) # (nbasis*nelmts, 3)
        alphas = alphas.view(-1) # (nbasis*nelmts,)
        centres = centres.view(-1, ndim) # (nbasis*nelmts, 3)
        coeffs = coeffs.view(-1) # (nbasis*nelmts,)
        self.coeffs = coeffs
        self.coeffs2 = coeffs * coeffs.unsqueeze(1) # (nbasis*nelmts, nbasis*nelmts)

        ecoeff_obj = Ecoeff(ijks, centres, alphas, atompos)
        self.olp_elmts = ecoeff_obj.get_overlap().unsqueeze(0) # (1, nbasis*nelmts, nbasis*nelmts)
        self.kin_elmts = ecoeff_obj.get_kinetics().unsqueeze(0) # (1, nbasis*nelmts, nbasis*nelmts)

        # coulomb part
        # ???
        # rcd = (rc - atompos.unsqueeze(-2).unsqueeze(-2) + 1e-12) # (natoms, nbasis*nelmts, nbasis*nelmts, 3)
        # q0 = torch.sqrt((rcd*rcd).sum(dim=-1)) # (natoms, nbasis*nelmts, nbasis*nelmts)
        # coul = -olp * (torch.erf(torch.sqrt(gamma) * (q0+1e-12)) / (q0+1e-12)) # (natoms, nbasis*nelmts, nbasis*nelmts)
        # coul_small = olp * torch.sqrt(gamma) * 2/np.sqrt(np.pi)
        self.coul_elmts = coul * atomzs.unsqueeze(-1).unsqueeze(-1) # (natoms, nbasis*nelmts, nbasis*nelmts)

        # combine the kinetics and coulomb elements
        self.kin_coul_elmts = self.kin_elmts + self.coul_elmts.sum(dim=0, keepdim=True) # (1, nbasis*nelmts, nbasis*nelmts)

        # get the contracted part
        self.kin_coul_mat = self._contract(self.kin_coul_elmts) # (1, nbasis, nbasis)
        self.olp_mat = self._contract(self.olp_elmts) # (1, nbasis, nbasis)

        # get the normalization constant
        norm = 1. / torch.sqrt(self.olp_mat.diagonal(dim1=-2, dim2=-1)) # (1,nbasis)
        norm_mat = norm.unsqueeze(-1) * norm.unsqueeze(1) # (1, nbasis, nbasis)

        # normalize the contracted matrix
        self.kin_coul_mat = self.kin_coul_mat * norm_mat
        self.olp_mat = self.olp_mat * norm_mat

        # get the basis
        self.rgrid = self.grid.rgrid_in_xyz # (nr, 3)
        rab = self.rgrid - centres.unsqueeze(1) # (nbasis*nelmts, nr, 3)
        dist_sq = (rab*rab).sum(dim=-1) # (nbasis*nelmts, nr)
        rab_power = ((rab+1e-15)**ijks.unsqueeze(1)).prod(dim=-1) # (nbasis*nelmts, nr)
        basis_all = rab_power * torch.exp(-alphas.unsqueeze(-1) * dist_sq) # (nbasis*nelmts, nr)
        basis = (basis_all * coeffs.unsqueeze(-1)).view(self.nbasis, self.nelmts, -1).sum(dim=1) # (nbasis, nr)
        norm_basis = basis * norm.squeeze(0).unsqueeze(-1)
        self.basis = norm_basis # (nbasis, nr)
        self.basis_dvolume = self.basis * self.grid.get_dvolume() # (nbasis, nr)

    ############################# basis part #############################
    def forward(self, wf, vext):
        # wf: (nbatch, nbasis, ncols)
        # vext: (nbatch, nr)

        nbatch, ns, ncols = wf.shape
        kin_coul = torch.matmul(self.kin_coul_mat, wf) # (nbatch, nbasis, ncols)

        # vext part
        # self.basis: (nbasis, nr)
        # extpot: (nbatch, nbasis, nbasis)
        extpot = torch.matmul(vext.unsqueeze(1) * self.basis_dvolume, self.basis.transpose(-2,-1))
        extpot = torch.bmm(extpot, wf) # (nbatch, nbasis, ncols)

        hwf = extpot + kin_coul
        return hwf

    def precond(self, y, vext, atomz, biases=None, M=None, mparams=None):
        # y: (nbatch, nbasis, ncols)
        # biases: (nbatch, ncols)
        return y

    def _overlap(self, wf):
        nbatch, nbasis, ncols = wf.shape
        res = torch.matmul(self.olp_mat, wf)  # (nbatch, nbasis, ncols)
        return res

    def torgrid(self, wfs, dim=-2):
        # wfs: (..., nbasis, ...)
        wfs = wfs.transpose(dim, -1) # (..., nbasis)
        wfr = torch.matmul(wfs, self.basis) # (..., nr)
        return wfr.transpose(dim, -1)

    ############################# grid part #############################
    @property
    def grid(self):
        return self._grid

    ############################# helper functions #############################
    def _contract(self, mat):
        # multiply the matrix with the contracted coefficients
        mat = mat * self.coeffs2

        # resize mat to have shape of (-1, nbasis, nelmts, nbasis, nelmts)
        batch_size = mat.shape[:-2]
        mat_size = mat.shape[-2:]
        mat = mat.view(-1, self.nbasis, self.nelmts, self.nbasis, self.nelmts)

        # sum the nelmts to get the basis
        cmat = mat.sum(dim=-1).sum(dim=-2) # (-1, nbasis, nbasis)
        return cmat.view(*batch_size, self.nbasis, self.nbasis)

class Ecoeff(object):
    def __init__(self, ijks, centres, alphas, atompos):
        # ijks: (nbasis*nelmts, 3)
        # centres: (nbasis*nelmts, 3)
        # alphas: (nbasis*nelmts)
        # atompos: (natoms, 3)
        self.max_basis = 8
        self.max_ijkflat = 64
        ijk_left = ijks.unsqueeze(0) # (1, nbasis*nelmts, 3)
        ijk_right = ijks.unsqueeze(1) # (nbasis*nelmts, 1, 3)
        self.ndim = ijk_left.shape[-1]
        ijk_pairs = ijk_left * self.max_basis + ijk_right # (nbasis*nelmts, nbasis*nelmts, 3)
        ijk_pairs2 = ijk_pairs[:,:,0] * self.max_ijkflat*self.max_ijkflat + \
                     ijk_pairs[:,:,1] * self.max_ijkflat + ijk_pairs[:,:,2] # (nbasis*nelmts, nbasis*nelmts)
        qab = centres - centres.unsqueeze(1) # (nbasis*nelmts, nbasis*nelmts, 3)
        qab_sq = qab**2 # (nbasis*nelmts, nbasis*nelmts, 3)
        gamma = (alphas + alphas.unsqueeze(1)) # (nbasis*nelmts, nbasis*nelmts)
        kappa = (alphas * alphas.unsqueeze(1)) / gamma # (nbasis*nelmts, nbasis*nelmts)
        # print(alphas.shape, gamma.shape, kappa.shape, qab_sq.shape)
        mab = torch.exp(-kappa * qab_sq) # (nbasis*nelmts, nbasis*nelmts, 3)
        ra = alphas.unsqueeze(-1) * centres # (nbasis*nelmts, 3)
        rc = (ra + ra.unsqueeze(1)) / gamma.unsqueeze(-1) # (nbasis*nelmts, nbasis*nelmts, 3)
        rcd = rc - atompos.unsqueeze(1).unsqueeze(1) # (natoms, nbasis*nelmts, nbasis*neltms, 3)
        rcd_sq = (rcd*rcd).sum(dim=-1) # (natoms, nbasis*nelmts, nbasis*neltms)

        self.ijk_left = ijk_left
        self.ijk_right = ijk_right
        self.ijk_pairs = ijk_pairs
        self.ijk_pairs2 = ijk_pairs2
        self.alphas = alphas.unsqueeze(0)
        self.betas = alphas.unsqueeze(1)
        self.qab = qab
        self.qab_sq = qab_sq
        self.gamma = gamma
        self.kappa = kappa
        self.mab = mab
        self.ra = ra
        self.rc = rc
        self.rcd = rcd
        self.rcd_sq = rcd_sq

        self.ijk_left_max = self.ijk_left.max()
        self.ijk_right_max = self.ijk_right.max()
        self.ijk_pairs2_unique = torch.unique(self.ijk_pairs2)
        self.idx_ijk = [(self.ijk_pairs2 == ijk_flat_value) for ijk_flat_value in self.ijk_pairs2_unique]

        # the key is: "i,j,t,xyz"
        self.key_format = "%d,%d,%d,%d"
        # the value's shape is: (nbasis*nelmts, nbasis*nelmts)
        self.e_memory = {
            "0,0,0,0": self.mab[:,:,0],
            "0,0,0,1": self.mab[:,:,1],
            "0,0,0,2": self.mab[:,:,2],
        }
        # the key format is: r,s,t,n
        # the value's shape is: (natoms, nbasis*nelmts, nbasis*nelmts)
        self.r_memory = {}

        # cache
        self.overlap = None
        self.kinetics = None
        self.coulomb = None

    def _split_pair(self, ijkpair):
        return ijkpair // self.max_basis, ijk_pair % self.max_basis

    def get_overlap(self):
        if self.overlap is None:
            overlap_dim = torch.empty_like(self.qab).to(self.qab.device) # (nbasis*nelmts, nbasis*nelmts, 3)
            for i in range(self.ijk_left_max):
                for j in range(self.ijk_right_max):
                    idx = (self.ijk_pairs == (i*self.max_basis + j)) # (nbasis*nelmts, nbasis*nelmts, 3)
                    # if idx.sum() == 0: continue
                    for xyz in range(self.ndim):
                        idxx = idx[:,:,xyz]
                        coeff = self.get_coeff(i, j, 0, xyz) # (nbasis*nelmts, nbasis*nelmts)
                        overlap_dim[idxx,xyz] = coeff[idxx]
            self.overlap = overlap_dim.prod(dim=-1) * (np.pi/self.gamma)**1.5 # (nbasis*nelmts, nbasis*nelmts)
        return self.overlap

    def get_kinetics(self):
        if self.kinetics is None:
            kinetics_dim0 = torch.empty_like(self.qab).to(self.qab.device) # (nbasis*nelmts, nbasis*nelmts, 3)
            kinetics_dim1 = torch.empty_like(self.qab).to(self.qab.device) # (nbasis*nelmts, nbasis*nelmts, 3)
            kinetics_dim2 = torch.empty_like(self.qab).to(self.qab.device) # (nbasis*nelmts, nbasis*nelmts, 3)
            for i in range(self.ijk_left_max):
                for j in range(self.ijk_right_max):
                    idx = self.ijk_pairs == (i*self.max_basis + j)
                    for xyz in range(self.ndim):
                        idxx = idx[:,:,xyz]
                        sij = self.get_coeff(i,j,0,xyz)
                        dij = j*(j-1)*self.get_coeff(i,j-2,0,xyz) - \
                              2*(2*j+1)*self.betas*sij + \
                              4*self.betas*self.betas*self.get_coeff(i,j+2,0,xyz) # (nbasis*nelmts, nbasis*nelmts)
                        sij_idxx = sij[idxx]
                        if xyz == 0:
                            kinetics_dim0[idxx,xyz] = dij[idxx]
                            kinetics_dim1[idxx,xyz] = sij_idxx
                            kinetics_dim2[idxx,xyz] = sij_idxx
                        elif xyz == 1:
                            kinetics_dim0[idxx,xyz] = sij_idxx
                            kinetics_dim1[idxx,xyz] = dij[idxx]
                            kinetics_dim2[idxx,xyz] = sij_idxx
                        elif xyz == 2:
                            kinetics_dim0[idxx,xyz] = sij_idxx
                            kinetics_dim1[idxx,xyz] = sij_idxx
                            kinetics_dim2[idxx,xyz] = dij[idxx]
            kinetics = kinetics_dim0.prod(dim=-1) + kinetics_dim1.prod(dim=-1) + \
                       kinetics_dim2.prod(dim=-1)
            self.kinetics = -0.5 * (np.pi/self.gamma)**1.5 * kinetics
        return self.kinetics

    def get_coulomb(self):
        # returns (natoms, nbasis*nelmts, nbasis*nelmts)
        if self.coulomb is None:
            # coulomb: (natoms, nbasis*nelmts, nbasis*nelmts)
            coulomb = torch.zeros_like(self.rcd_sq).to(self.rcd_sq.device)
            for ijk_flat_value, idx in zip(self.ijk_pairs2_unique, self.idx_ijk):
                # idx: (nbasis*nelmts, nbasis*nelmts)
                k, l, m, u, v, w = self._unpack_ijk_flat_value(ijk_flat_value)

                for r in range(k+u):
                    Erku = self.get_coeff(k, u, r, 0)[idx] # flatten tensor
                    for s in range(l+v):
                        Eslv = self.get_coeff(l, v, s, 0)[idx]
                        for t in range(m+w):
                            Etmw = self.get_coeff(m, w, t, 0)[idx]
                            Rrst = self.get_rcoeff(r, s, t, 0)[:,idx] # (natoms, -1)
                            coulomb[:,idx] += (Erku * Eslv * Etmw) * Rrst
            self.coulomb = (2*np.pi/self.gamma) * coulomb
        return self.coulomb

    def get_coeff(self, i, j, t, xyz):
        # return: (nbasis*nelmts, nbasis*nelmts)
        if t < 0 or t > i+j or i < 0 or j < 0:
            return 0.0
        coeff = self._access_coeff(i, j, t, xyz)
        if coeff is not None:
            return coeff

        if i == 0 and j > 0:
            coeff = 1./(2*self.gamma) * self.get_coeff(i, j-1, t-1, xyz) + \
                    self.kappa * self.qab[:,:,xyz] / self.betas * self.get_coeff(i, j-1, t, xyz) + \
                    (t + 1) * self.get_coeff(i, j-1, t+1, xyz)
        elif i > 0:
            coeff = 1./(2*self.gamma) * self.get_coeff(i-1, j, t-1, xyz) - \
                    self.kappa * self.qab[:,:,xyz] / self.alphas * self.get_coeff(i-1, j, t, xyz) + \
                    (t + 1) * self.get_coeff(i-1, j, t+1, xyz)

        # add to the memory
        key = self.key_format % (i,j,t,xyz)
        self.e_memory[key] = coeff
        return coeff

    def get_rcoeff(self, r, s, t, n):
        # rcd: (natoms, nbasis*nelmts, nbasis*nelmts, 3)
        # return: (natoms, nbasis*nelmts, nbasis*nelmts)
        if r < 0 or s < 0 or t < 0:
            return 0.0
        coeff = self._access_rcoeff()
        if coeff is not None:
            return coeff

        if r == 0 and s == 0 and t == 0:
            # (natoms, nbasis*nelmts, nbasis*nelmts)
            coeff = (-2*self.gamma)**n * self._boys(n, self.gamma * self.rcd_sq)
        elif r > 0:
            coeff = (r-1) * self.get_rcoeff(r-2, s, t, n+1) + \
                    self.rcd[:,:,:,0] * self.get_rcoeff(r-1, s, t, n+1)
        elif s > 0:
            coeff = (s-1) * self.get_rcoeff(r, s-2, t, n+1) + \
                    self.rcd[:,:,:,1] * self.get_rcoeff(r, s-1, t, n+1)
        elif t > 0:
            coeff = (t-1) * self.get_rcoeff(r, s, t-2, n+1) + \
                    self.rcd[:,:,:,2] * self.get_rcoeff(r, s, t-1, n+1)

        # save the coefficients
        key = self.key_format % (r,s,t,n)
        self.r_memory[key] = coeff
        return coeff

    def _unpack_ijk_flat_value(self, ijk_flat_value):
        ijk_pair2 = ijk_flat_value % self.max_ijkflat
        ijk_pair1 = (ijk_flat_value // self.max_ijkflat) % self.max_ijkflat
        ijk_pair0 = (ijk_flat_value // self.max_ijkflat) // self.max_ijkflat
        k = ijk_pair0 // self.max_basis
        l = ijk_pair1 // self.max_basis
        m = ijk_pair2 // self.max_basis
        u = ijk_pair0 % self.max_basis
        v = ijk_pair1 % self.max_basis
        w = ijk_pair2 % self.max_basis
        return k, l, m, u, v, w

    def _boys(self, n, T):
        nhalf = n + 0.5
        return incgamma(nhalf, T) / (2*T**nhalf)

    def _access_coeff(self, i, j, t, xyz):
        key = self.key_format % (i,j,t,xyz)
        if key in self.e_memory:
            return self.e_memory[key]
        else:
            return None

    def _access_rcoeff(self, r, s, t, n):
        key = self.key_format % (r,s,t,n)
        if key in self.r_memory:
            return self.r_memory[key]
        else:
            return None
