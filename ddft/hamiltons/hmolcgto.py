import os
import re
import ctypes
from typing import NamedTuple, List, Callable, Tuple, Optional
import torch
import numpy as np
from ddft.basissets.cgtobasis import CGTOBasis, AtomCGTOBasis

NDIM = 3
PTR_RINV_ORIG = 4  # from libcint/src/cint_const.h

# load the libcint
_curpath = os.path.dirname(os.path.abspath(__file__))
_libcint_path = os.path.join(_curpath, "../../submodules/libcint/build/libcint.so")
cint = ctypes.cdll.LoadLibrary(_libcint_path)

class LibcintWrapper(object):
    # this class provides the contracted gaussian integrals
    def __init__(self, atombases: List[AtomCGTOBasis], spherical: bool = True) -> None:
        self._atombases = atombases
        self._spherical = spherical
        self._natoms = len(atombases)

        # the libcint lists
        self._ptr_env = 20
        self._atm: List[int] = []
        self._env: List[float] = [0.0] * self._ptr_env
        self._bas: List[int] = []

        # the list of tensors
        self._allpos: List[torch.Tensor] = []
        self._allalphas: List[torch.Tensor] = []
        self._allcoeffs: List[torch.Tensor] = []
        # offset indicating the starting point of i-th shell in r & (alphas & coeffs)
        # in the flatten list above
        self._r_idx: List[int] = []
        self._basis_offset: List[int] = [0]

        # construct the atom, basis, and env lists
        self._nshells = []
        for i, atombasis in enumerate(atombases):
            # modifying most of the parameters above
            ns = self._add_atom_and_basis(i, atombasis)
            self._nshells.append(ns)
        self._nshells_tot = sum(self._nshells)

        # flatten the params list
        self._allpos_params = torch.cat(self._allpos, dim=0)  # (natom, NDIM)
        self._allalphas_params = torch.cat(self._allalphas, dim=0)  # (ntot_gauss)
        self._allcoeffs_params = torch.cat(self._allcoeffs, dim=0)  # (ntot_gauss)

        # convert the lists to numpy (to make it contiguous, Python list is not contiguous)
        self._atm = np.array(self._atm, dtype=np.int32)
        self._bas = np.array(self._bas, dtype=np.int32)
        self._env = np.array(self._env, dtype=np.float64)

        # get the c-pointer of the numpy array
        self.atm_ctypes = self._atm.ctypes.data_as(ctypes.c_void_p)
        self.bas_ctypes = self._bas.ctypes.data_as(ctypes.c_void_p)
        self.env_ctypes = self._env.ctypes.data_as(ctypes.c_void_p)
        self.natm_ctypes = ctypes.c_int(self._atm.shape[0])
        self.nbas_ctypes = ctypes.c_int(self._bas.shape[0])

        # get dtype and device for torch's tensors
        self.dtype = atombases[0].bases[0].alphas.dtype
        self.device = atombases[0].bases[0].alphas.device

        # get the size of the contracted gaussian
        self._offset = [0]
        for i in range(self._nshells_tot):
            nbasiscontr = self._nbasiscontr(i)
            self._offset.append(self._offset[-1] + nbasiscontr)
        self._nbases_tot = self._offset[-1]

    def overlap(self) -> torch.Tensor:
        return self._int1e("ovlp")

    def kinetic(self) -> torch.Tensor:
        return self._int1e("kin")

    def nuclattr(self) -> torch.Tensor:
        return self._int1e("nuc", True)

    def elrep(self) -> torch.Tensor:
        return self._int2e()

    ################ integrals to construct the operator ################
    def _int1e(self, shortname: str, nuc: bool = False) -> torch.Tensor:
        # one electron integral (overlap, kinetic, and nuclear attraction)

        shape = (self._nbases_tot, self._nbases_tot)
        res = torch.empty(shape, dtype=self.dtype, device=self.device)
        for i1 in range(self._nshells_tot):
            c1, a1, r1 = self._get_params(i1)
            slice1 = self._get_matrix_index(i1)

            for i2 in range(i1, self._nshells_tot):
                # get the parameters for backward propagation
                c2, a2, r2 = self._get_params(i2)
                slice2 = self._get_matrix_index(i2)

                if nuc:
                    # computing the nuclear attraction operator per atom
                    # so that the gradient w.r.t. atom's position can be computed
                    mat = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                    for ia in range(self._natoms):
                        self._set_coord_to_centre_on_atom(ia)
                        atompos = self._allpos_params[ia]
                        z = float(self._atm[ia, 0])
                        mat = mat - z * _Int1eFunction.apply(
                            c1, a1, r1 - atompos,
                            c2, a2, r2 - atompos,
                            self, "rinv", i1, i2)
                        self._restore_coords()

                    # # debugging code
                    # mat2 = _Int1eFunction.apply(
                    #     c1, a1, r1, c2, a2, r2,
                    #     self, shortname, i1, i2)
                    # assert torch.allclose(mat, mat2)

                else:
                    mat = _Int1eFunction.apply(
                        c1, a1, r1, c2, a2, r2,
                        self, shortname, i1, i2)

                # apply symmetry
                res[slice1, slice2] = mat
                res[slice2, slice1] = mat.transpose(-2, -1)
        return res

    def _int2e(self) -> torch.Tensor:
        shape = (self._nbases_tot, self._nbases_tot, self._nbases_tot, self._nbases_tot)
        res = torch.empty(shape, dtype=self.dtype, device=self.device)

        # TODO: add symmetry here
        for i1 in range(self._nshells_tot):
            c1, a1, r1 = self._get_params(i1)
            slice1 = self._get_matrix_index(i1)
            for i2 in range(self._nshells_tot):
                c2, a2, r2 = self._get_params(i2)
                slice2 = self._get_matrix_index(i2)
                for i3 in range(self._nshells_tot):
                    c3, a3, r3 = self._get_params(i3)
                    slice3 = self._get_matrix_index(i3)
                    for i4 in range(self._nshells_tot):
                        c4, a4, r4 = self._get_params(i4)
                        slice4 = self._get_matrix_index(i4)
                        mat = _Int2eFunction.apply(c1, a1, r1, c2, a2, r2,
                                                   c3, a3, r3, c4, a4, r4,
                                                   self, "", i1, i2, i3, i4)
                        res[slice1, slice2, slice3, slice4] = mat
        return res

    def _set_coord_to_centre_on_atom(self, ia: int) -> None:
        # set the coordinates to centre on atoms ia
        idx = self._atm[ia, 1]
        atompos = self._env[idx : idx + NDIM]
        self._env[PTR_RINV_ORIG : PTR_RINV_ORIG + NDIM] = atompos

    def _restore_coords(self) -> None:
        # restore the central coordinate into 0.0
        self._env[PTR_RINV_ORIG : PTR_RINV_ORIG + NDIM] = 0.0

    def _get_params(self, sh1: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # getting the parameters of the sh1-th shell (i.e. coeffs, alphas, and pos)
        ib1l = self._basis_offset[sh1]
        ib1u = self._basis_offset[sh1 + 1]
        c1 = self._allcoeffs_params[ib1l:ib1u]
        a1 = self._allalphas_params[ib1l:ib1u]
        r1 = self._allpos_params[self._r_idx[sh1]]
        # r1 = self._allpos_params[self._r_offset_l[sh1]:self._r_offset_u[sh1]]
        return c1, a1, r1

    def _get_matrix_index(self, sh1: int) -> slice:
        # getting the lower and upper indices of the sh1-th shell in the contracted
        # integral matrix
        return slice(self._offset[sh1], self._offset[sh1 + 1], None)

    ################ one electron integral for a basis pair ################
    def calc_integral_1e(self, shortname: str, sh1: int, sh2: int) -> torch.Tensor:
        # calculate the one-electron integral (type depends on the shortname)
        # for basis in shell sh1 & sh2 using libcint
        shortname, toflip = self._is_int1e_need_flip(shortname)
        toflip = toflip and (sh1 != sh2)
        if toflip:
            sh1, sh2 = sh2, sh1

        # NOTE: this function should only be called from this file only.
        # it does not propagate gradient.
        opname = self._get_int1e_name(shortname)
        operator = getattr(cint, opname)
        outshape = self._get_int1e_outshape(shortname, sh1, sh2)

        c_shls = (ctypes.c_int * 2)(sh1, sh2)
        out = np.empty(outshape, dtype=np.float64)
        out_ctypes = out.ctypes.data_as(ctypes.c_void_p)

        # calculate the integral
        operator.restype = ctypes.c_double
        operator(out_ctypes, c_shls, self.atm_ctypes, self.natm_ctypes,
                 self.bas_ctypes, self.nbas_ctypes, self.env_ctypes)

        out_tensor = torch.tensor(out, dtype=self.dtype, device=self.device)
        if toflip:
            out_tensor = out_tensor.transpose(-2, -1)
        return out_tensor

    def get_int1e_deriv_shortname(self, opshortname: str, derivmode: str) -> str:
        if derivmode == "r1":
            return "ip%s" % opshortname
        elif derivmode == "r2":
            return "%sip" % opshortname
        else:
            raise RuntimeError("Unknown derivmode: %s" % derivmode)

    def _is_int1e_need_flip(self, shortname: str) -> Tuple[str, bool]:
        # check if the integral needs to be flipped (e.g. "nucip" need to be
        # flipped to "ipnuc" to access the integral from libcint)

        # count the number of "ip" at the beginning and end of the string
        n_ip_start = len(re.findall(r"^(?:ip)*(?:ip)?", shortname)[0]) // 2
        n_ip_end = len(re.findall(r"(?:ip)?(?:ip)*$", shortname)[0]) // 2
        flip = n_ip_end > n_ip_start
        if flip:
            sname = shortname[2 * n_ip_start : -2 * n_ip_end]
            shortname = ("ip" * n_ip_end) + sname + ("ip" * n_ip_start)
        return shortname, flip

    def _get_int1e_name(self, shortname: str):
        # get the full name of the integral 1 electron to be called in libcint
        suffix = "sph" if self._spherical else "cart"
        return "cint1e_%s_%s" % (shortname, suffix)

    def _get_int1e_outshape(self, shortname: str, sh1: int, sh2: int) -> List[int]:
        if shortname in ["ovlp", "kin", "nuc", "rinv"]:
            return [self._nbasiscontr(sh1), self._nbasiscontr(sh2)]
        elif shortname in ["ipovlp", "ipkin", "ipnuc", "iprinv"]:
            return [NDIM, self._nbasiscontr(sh1), self._nbasiscontr(sh2)]
        elif shortname in ["ipipovlp", "ipovlpip", "ipipkin", "ipkinip",
                           "ipipnuc", "ipnucip", "ipiprinv", "iprinvip"]:
            return [NDIM, NDIM, self._nbasiscontr(sh1), self._nbasiscontr(sh2)]
        else:
            raise RuntimeError("Unset outshape for %s" % shortname)

    ################ two electrons integral for a basis pair ################
    def calc_integral_2e(self, shortname: str, sh1: int, sh2: int,
                         sh3: int, sh4: int) -> torch.Tensor:
        # calculate the one-electron integral (type depends on the shortname)
        # for basis in shell sh1 & sh2 using libcint

        # NOTE: this function should only be called from this file only.
        # it does not propagate gradient.
        opname = self._get_int1e_name(shortname)
        operator = getattr(cint, opname)
        outshape = self._get_int1e_outshape(shortname, sh1, sh2)

        c_shls = (ctypes.c_int * 2)(sh1, sh2)
        out = np.empty(outshape, dtype=np.float64)
        out_ctypes = out.ctypes.data_as(ctypes.c_void_p)

        # calculate the integral
        operator.restype = ctypes.c_double
        operator(out_ctypes, c_shls, self.atm_ctypes, self.natm_ctypes,
                 self.bas_ctypes, self.nbas_ctypes, self.env_ctypes)

        out_tensor = torch.tensor(out, dtype=self.dtype, device=self.device)
        return out_tensor

    ################ misc functions ################
    def _nbasiscontr(self, sh: int) -> int:
        if self._spherical:
            op = cint.CINTcgto_spheric
        else:
            op = cint.CINTcgto_cart
        return op(ctypes.c_int(sh), self.bas_ctypes)

    def _add_atom_and_basis(self, iatom: int, atombasis: AtomCGTOBasis) -> int:
        # construct the atom first
        assert atombasis.pos.numel() == NDIM, "Please report this bug in Github"
        #                charge           ptr_coord     (unused for standard nucl model)
        self._atm.append([atombasis.atomz, self._ptr_env, 0, 0, 0, 0])
        self._env.extend(atombasis.pos)
        self._ptr_env += NDIM

        # then construct the basis
        for basis in atombasis.bases:
            assert basis.alphas.shape == basis.coeffs.shape and basis.alphas.ndim == 1,\
                   "Please report this bug in Github"

            ngauss = len(basis.alphas)
            #                iatom, angmom,       ngauss, ncontr, kappa, ptr_exp
            self._bas.append([iatom, basis.angmom, ngauss, 1     , 0, self._ptr_env,
            #                ptr_coeffs,           unused
                             self._ptr_env + ngauss, 0])
            self._env.extend(basis.alphas)
            self._env.extend(basis.coeffs)
            self._ptr_env += 2 * ngauss

            # add the basis coeffs and alphas to the flat list and update the offset
            self._allalphas.append(basis.alphas)
            self._allcoeffs.append(basis.coeffs)
            self._basis_offset.append(self._basis_offset[-1] + ngauss)
            self._r_idx.append(iatom)

        # add the atom position
        self._allpos.append(atombasis.pos.unsqueeze(0))
        return len(atombasis.bases)

class _Int1eFunction(torch.autograd.Function):
    # wrapper class to provide the gradient of the one-e integrals
    @staticmethod
    def forward(ctx,
                c1: torch.Tensor, a1: torch.Tensor, r1: torch.Tensor,
                c2: torch.Tensor, a2: torch.Tensor, r2: torch.Tensor,
                env: LibcintWrapper, opshortname: str, sh1: int, sh2: int) -> \
                torch.Tensor:  # type: ignore
        # c_: (nbasis,)
        # a_: (nbasis,)
        # r_: (NDIM,)
        # ratom: (NDIM,)
        # they are not used in forward, but present in the argument so that
        # the gradient can be propagated
        out_tensor = env.calc_integral_1e(opshortname, sh1, sh2)
        ctx.save_for_backward(c1, a1, r1, c2, a2, r2)
        ctx.other_info = (env, sh1, sh2, opshortname)
        return out_tensor  # (*outshape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore
        # grad_out: (*outshape)
        c1, a1, r1, c2, a2, r2 = ctx.saved_tensors
        env, sh1, sh2, opshortname = ctx.other_info

        # TODO: to be completed (???)
        grad_c1: Optional[torch.Tensor] = None
        grad_a1: Optional[torch.Tensor] = None
        grad_c2: Optional[torch.Tensor] = None
        grad_a2: Optional[torch.Tensor] = None

        grad_r1: Optional[torch.Tensor] = None
        if r1.requires_grad:
            opsname = env.get_int1e_deriv_shortname(opshortname, "r1")
            doutdr1 = _Int1eFunction.apply(*ctx.saved_tensors, env, opsname,
                                           sh1, sh2)  # (NDIM, *outshape)
            # negative because the integral calculates the nabla w.r.t. the
            # spatial coordinate, not the basis central position
            grad_r1 = -(grad_out * doutdr1).reshape(NDIM, -1).sum(dim=-1)

        grad_r2: Optional[torch.Tensor] = None
        if r2.requires_grad:
            opsname = env.get_int1e_deriv_shortname(opshortname, "r2")
            doutdr2 = _Int1eFunction.apply(*ctx.saved_tensors, env, opsname,
                                           sh1, sh2)
            grad_r2 = -(grad_out * doutdr2).reshape(NDIM, -1).sum(dim=-1)

        return grad_c1, grad_a1, grad_r1, grad_c2, grad_a2, grad_r2, \
               None, None, None, None

class _Int2eFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                c1: torch.Tensor, a1: torch.Tensor, r1: torch.Tensor,
                c2: torch.Tensor, a2: torch.Tensor, r2: torch.Tensor,
                c3: torch.Tensor, a3: torch.Tensor, r3: torch.Tensor,
                c4: torch.Tensor, a4: torch.Tensor, r4: torch.Tensor,
                env: LibcintWrapper, opshortname: str,
                sh1: int, sh2: int, sh3: int, sh4: int):
        # c_: (nbasis,)
        # a_: (nbasis,)
        # r_: (NDIM,)
        # they are not used in forward, but present in the argument so that
        # the gradient can be propagated
        out_tensor = env.calc_integral_2e(opshortname, sh1, sh2, sh3, sh4)
        ctx.save_for_backward(c1, a1, r1, c2, a2, r2, c3, a3, r3, c4, a4, r4)
        ctx.other_info = (env, sh1, sh2, sh3, sh4, opshortname)
        return out_tensor  # (*outshape)

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: (*outshape)
        c1, a1, r1, c2, a2, r2, c3, a3, r3, c4, a4, r4 = ctx.saved_tensors
        env, sh1, sh2, sh3, sh4, opshortname = ctx.other_info

        # TODO: to be completed (???)
        grad_c1 = None
        grad_a1 = None
        grad_c2 = None
        grad_a2 = None
        grad_c3 = None
        grad_a3 = None
        grad_c4 = None
        grad_a4 = None

        grad_r1 = None
        if r1.requires_grad:
            opsname = env.get_int2e_deriv_shortname(opshortname, "r1")
            doutdr1 = _Int2eFunction.apply(*ctx.saved_tensors, env, opsname,
                                           sh1, sh2, sh3, sh4)  # (*outshape, NDIM)
            grad_r1 = -(grad_out.unsqueeze(-1) * doutdr1).view(-1, NDIM).sum(dim=0)

        grad_r2 = None
        if r2.requires_grad:
            opsname = env.get_int2e_deriv_shortname(opshortname, "r2")
            doutdr2 = _Int2eFunction.apply(*ctx.saved_tensors, env, opsname,
                                           sh1, sh2, sh3, sh4)  # (*outshape, NDIM)
            grad_r2 = -(grad_out.unsqueeze(-1) * doutdr2).view(-1, NDIM).sum(dim=0)

        grad_r3 = None
        if r3.requires_grad:
            opsname = env.get_int2e_deriv_shortname(opshortname, "r3")
            doutdr3 = _Int2eFunction.apply(*ctx.saved_tensors, env, opsname,
                                           sh1, sh2, sh3, sh4)  # (*outshape, NDIM)
            grad_r3 = -(grad_out.unsqueeze(-1) * doutdr3).view(-1, NDIM).sum(dim=0)

        grad_r4 = None
        if r4.requires_grad:
            opsname = env.get_int2e_deriv_shortname(opshortname, "r4")
            doutdr4 = _Int2eFunction.apply(*ctx.saved_tensors, env, opsname,
                                           sh1, sh2, sh3, sh4)  # (*outshape, NDIM)
            grad_r4 = -(grad_out.unsqueeze(-1) * doutdr4).view(-1, NDIM).sum(dim=0)

        return grad_c1, grad_a1, grad_r1, grad_c2, grad_a2, grad_r2, \
               grad_c3, grad_a3, grad_r3, grad_c4, grad_a4, grad_r4, \
               None, None, None, None, None, None

if __name__ == "__main__":
    from ddft.basissets.cgtobasis import loadbasis
    dtype = torch.double
    pos1 = torch.tensor([0.0, 0.0,  0.8], dtype=dtype, requires_grad=True)
    pos2 = torch.tensor([0.0, 0.0, -0.8], dtype=dtype, requires_grad=True)

    def get_int1e(pos1, pos2, name):
        bases = loadbasis("1:3-21G", dtype=dtype, requires_grad=False)
        atombasis1 = AtomCGTOBasis(atomz=1, bases=bases, pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=1, bases=bases, pos=pos2)
        env = LibcintWrapper([atombasis1, atombasis2], spherical=False)
        if name == "overlap":
            return env.overlap()
        elif name == "kinetic":
            return env.kinetic()
        elif name == "nuclattr":
            return env.nuclattr()
        else:
            raise RuntimeError()

    # torch.autograd.gradcheck(get_int1e, (pos1, pos2, "overlap"))
    # torch.autograd.gradcheck(get_int1e, (pos1, pos2, "kinetic"))
    # torch.autograd.gradgradcheck(get_int1e, (pos1, pos2, "overlap"))
    # torch.autograd.gradgradcheck(get_int1e, (pos1, pos2, "kinetic"))
    torch.autograd.gradcheck(get_int1e, (pos1, pos2, "nuclattr"))
