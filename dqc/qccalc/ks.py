from typing import Optional, Mapping, Any, Tuple, List, Union, overload
import torch
import xitorch as xt
import xitorch.linalg
import xitorch.optimize
from dqc.system.base_system import BaseSystem
from dqc.qccalc.base_qccalc import BaseQCCalc
from dqc.xc.base_xc import BaseXC
from dqc.api.getxc import get_xc

class KS(BaseQCCalc):
    """
    Performing Restricted or Unrestricted Kohn-Sham DFT calculation.

    Arguments
    ---------
    system: BaseSystem
        The system to be calculated.
    xc: str
        The exchange-correlation potential and energy to be used.
    vext: torch.Tensor or None
        The external potential applied to the system. It must have the shape of
        ``(*BV, system.get_grid().shape[-2])``
    restricted: bool
        If True, performing restricted Kohn-Sham DFT. If False, it performs
        the unrestricted Kohn-Sham DFT.
    """

    def __init__(self, system: BaseSystem, xc: Union[str, BaseXC],
                 vext: Optional[torch.Tensor] = None,
                 restricted: bool = True):

        self.polarized = not restricted

        # get the xc object
        if isinstance(xc, str):
            self.xc: BaseXC = get_xc(xc, polarized=self.polarized)
        else:
            self.xc = xc
        self.system = system

        # build and setup basis and grid
        self.system.setup_grid()
        self.hamilton = system.get_hamiltonian()
        self.hamilton.build()
        self.hamilton.setup_grid(system.get_grid(), self.xc)

        # get the orbital info
        self.orb_weight = system.get_orbweight(polarized=self.polarized)  # (norb,)
        if self.polarized:
            assert isinstance(self.orb_weight, tuple)
            assert self.orb_weight[0].shape[-1] == self.orb_weight[1].shape[-1]
            self.norb = self.orb_weight[0].shape[-1]
        else:
            assert isinstance(self.orb_weight, torch.Tensor)
            self.norb = self.orb_weight.shape[-1]

        # set up the vext linear operator
        self.knvext_linop = self.hamilton.get_kinnucl()  # kinetic, nuclear, and external potential
        if vext is not None:
            assert vext.shape[-1] == system.get_grid().get_rgrid().shape[-2]
            self.knvext_linop = self.knvext_linop + self.hamilton.get_vext(vext)

        # misc info
        self.dtype = self.knvext_linop.dtype
        self.device = self.knvext_linop.device
        self.has_run = True

    def run(self, dm0: Optional[Union[torch.Tensor, Tuple[torch.Tensor]]] = None,  # type: ignore
            eigen_options: Optional[Mapping[str, Any]] = None,
            fwd_options: Optional[Mapping[str, Any]] = None,
            bck_options: Optional[Mapping[str, Any]] = None) -> BaseQCCalc:

        # setup the default options
        if eigen_options is None:
            eigen_options = {
                # NOTE: temporary solution before the gradient calculation in
                # pytorch's symeig is fixed (for the degenerate case)
                # see https://github.com/pytorch/pytorch/issues/47599
                "method": "custom_exacteig"
            }
        if fwd_options is None:
            fwd_options = {
                "method": "broyden1",
                "alpha": -0.5,
                "maxiter": 50,
                # "verbose": True,
            }
        if bck_options is None:
            bck_options = {
                # NOTE: it seems like in most cases the jacobian matrix is posdef
                # if it is not the case, we can just remove the line below
                "posdef": True,
                # "verbose": True,
            }

        # save the eigen_options for use in diagonalization
        self.eigen_options = eigen_options

        # set up the initial self-consistent param guess
        if dm0 is None:
            if not self.polarized:
                dm0 = torch.zeros(self.knvext_linop.shape, dtype=self.dtype,
                                  device=self.device)
            else:
                dm0_u = torch.zeros(self.knvext_linop.shape, dtype=self.dtype,
                                    device=self.device)
                dm0_d = torch.zeros(self.knvext_linop.shape, dtype=self.dtype,
                                    device=self.device)
                dm0 = (dm0_u, dm0_d)

        scp0 = self.__dm2scp(dm0)

        # do the self-consistent iteration
        scp = xitorch.optimize.equilibrium(
            fcn=self.__scp2scp,
            y0=scp0,
            bck_options={**bck_options},
            **fwd_options)

        # post-process parameters
        self._dm = self.__scp2dm(scp)
        self.has_run = True
        return self

    def energy(self) -> torch.Tensor:
        # calculate the total energy from the diagonalization
        fock = self.__dm2fock(self._dm)

        # get the energy from xc
        e_exc = self.hamilton.get_exc(self._dm)

        if not self.polarized:
            assert isinstance(self.orb_weight, torch.Tensor)
            assert isinstance(self._dm, torch.Tensor)

            # eivals: (..., norb), eivecs: (..., nao, norb)
            eivals, eivecs = self.__diagonalize_fock(fock)
            e_eivals = torch.sum(eivals * self.orb_weight, dim=-1)

            # get the energy from xc potential
            vxc = self.hamilton.get_vxc(self._dm)
            e_vxc = torch.einsum("...rc,c,...rc->...", vxc.mm(eivecs), self.orb_weight, eivecs)

            # get the energy from electron repulsion
            elrep = self.hamilton.get_elrep(self._dm)
            e_elrep = 0.5 * torch.einsum("...rc,c,...rc->...", elrep.mm(eivecs), self.orb_weight, eivecs)
        else:
            assert isinstance(self.orb_weight, tuple)
            assert isinstance(self._dm, tuple)

            # eivals: (..., norb), eivecs: (..., nao, norb)
            eivals_u, eivecs_u = self.__diagonalize_fock(fock[0])
            eivals_d, eivecs_d = self.__diagonalize_fock(fock[1])
            e_eivals_u = torch.sum(eivals_u * self.orb_weight[0], dim=-1)
            e_eivals_d = torch.sum(eivals_d * self.orb_weight[1], dim=-1)
            e_eivals = e_eivals_u + e_eivals_d

            # get the energy from xc potential
            vxc_u, vxc_d = self.hamilton.get_vxc(self._dm)
            e_vxc_u = torch.einsum("...rc,c,...rc->...", vxc_u.mm(eivecs_u), self.orb_weight[0], eivecs_u)
            e_vxc_d = torch.einsum("...rc,c,...rc->...", vxc_d.mm(eivecs_d), self.orb_weight[1], eivecs_d)
            e_vxc = e_vxc_u + e_vxc_d

            # get the energy from electron repulsion
            elrep = self.hamilton.get_elrep(self._dm[0] + self._dm[1])
            e_elrep_u = 0.5 * torch.einsum("...rc,c,...rc->...", elrep.mm(eivecs_u), self.orb_weight[0], eivecs_u)
            e_elrep_d = 0.5 * torch.einsum("...rc,c,...rc->...", elrep.mm(eivecs_d), self.orb_weight[1], eivecs_d)
            e_elrep = e_elrep_u + e_elrep_d

        # compute the total energy
        e_tot = e_eivals + (e_exc - e_vxc) - e_elrep + self.system.get_nuclei_energy()
        return e_tot

    def aodm(self) -> torch.Tensor:
        return self._dm

    @overload
    def __dm2fock(self, dm: torch.Tensor) -> xt.LinearOperator:
        ...

    @overload
    def __dm2fock(self, dm: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[xt.LinearOperator, xt.LinearOperator]:
        ...

    def __dm2fock(self, dm):
        if isinstance(dm, torch.Tensor):
            # construct the fock matrix from the density matrix
            elrep = self.hamilton.get_elrep(dm)  # (..., nao, nao)
            vxc = self.hamilton.get_vxc(dm)
            return self.knvext_linop + elrep + vxc
        else:
            elrep = self.hamilton.get_elrep(dm[0] + dm[1])  # (..., nao, nao)
            vext_elrep = self.knvext_linop + elrep

            vxc_ud = self.hamilton.get_vxc(dm)
            return vext_elrep + vxc_ud[0], vext_elrep + vxc_ud[1]

    @overload
    def __fock2dm(self, fock: xt.LinearOperator) -> torch.Tensor:
        ...

    @overload
    def __fock2dm(self, fock: Tuple[xt.LinearOperator, xt.LinearOperator]) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def __fock2dm(self, fock):
        # diagonalize the fock matrix and obtain the density matrix
        if isinstance(fock, xt.LinearOperator):  # unpolarized
            assert isinstance(self.orb_weight, torch.Tensor)
            eigvals, eigvecs = self.__diagonalize_fock(fock)
            dm = self.hamilton.ao_orb2dm(eigvecs, self.orb_weight)
            return dm
        else:  # polarized
            assert isinstance(self.orb_weight, tuple)
            eigvals_u, eigvecs_u = self.__diagonalize_fock(fock[0])
            eigvals_d, eigvecs_d = self.__diagonalize_fock(fock[1])
            dm_u = self.hamilton.ao_orb2dm(eigvecs_u, self.orb_weight[0])
            dm_d = self.hamilton.ao_orb2dm(eigvecs_d, self.orb_weight[1])
            return dm_u, dm_d

    def __diagonalize_fock(self, fock: xt.LinearOperator) -> Tuple[torch.Tensor, torch.Tensor]:
        return xitorch.linalg.lsymeig(
            A=fock,
            neig=self.norb,
            M=self.hamilton.get_overlap(),
            **self.eigen_options)

    ######### self-consistent-param related #########
    # the functions below are created so that it is easy to change which
    # parameters is involved in the self-consistent iterations

    def __dm2scp(self, dm: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if isinstance(dm, torch.Tensor):  # unpolarized
            # scp is the fock matrix
            return self.__dm2fock(dm).fullmatrix()
        else:  # polarized
            # scp is the concatenated fock matrix
            fock_u, fock_d = self.__dm2fock(dm)
            mat_u = fock_u.fullmatrix().unsqueeze(0)
            mat_d = fock_d.fullmatrix().unsqueeze(0)
            return torch.cat((mat_u, mat_d), dim=0)

    def __scp2dm(self, scp: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.polarized:
            fock = xt.LinearOperator.m(scp, is_hermitian=True)
            return self.__fock2dm(fock)
        else:
            fock_u = xt.LinearOperator.m(scp[0], is_hermitian=True)
            fock_d = xt.LinearOperator.m(scp[1], is_hermitian=True)
            return self.__fock2dm((fock_u, fock_d))

    def __scp2scp(self, scp: torch.Tensor) -> torch.Tensor:
        dm = self.__scp2dm(scp)
        return self.__dm2scp(dm)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "__scp2scp":
            return self.getparamnames("__scp2dm", prefix=prefix) + \
                self.getparamnames("__dm2scp", prefix=prefix)
        elif methodname == "__scp2dm":
            return self.getparamnames("__fock2dm", prefix=prefix)
        elif methodname == "__dm2scp":
            return self.getparamnames("__dm2fock", prefix=prefix)
        elif methodname == "__fock2dm":
            return self.getparamnames("__diagonalize_fock", prefix=prefix) + \
                self.hamilton.getparamnames("ao_orb2dm", prefix=prefix + "hamilton.") + \
                [prefix + "orb_weight"]
        elif methodname == "__dm2fock":
            hprefix = prefix + "hamilton."
            return self.hamilton.getparamnames("get_elrep", prefix=hprefix) + \
                self.hamilton.getparamnames("get_vxc", prefix=hprefix) + \
                self.knvext_linop._getparamnames(prefix=prefix + "knvext_linop.")
        elif methodname == "__diagonalize_fock":
            return self.hamilton.getparamnames("get_overlap", prefix=prefix + "hamilton.")
        else:
            raise KeyError("Method %s has no paramnames set" % methodname)
        return []  # TODO: to complete
