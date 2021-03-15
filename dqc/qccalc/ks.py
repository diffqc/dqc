from typing import Optional, Mapping, Any, Tuple, List, Union, overload
import torch
import xitorch as xt
import xitorch.linalg
import xitorch.optimize
from dqc.system.base_system import BaseSystem
from dqc.qccalc.base_qccalc import BaseQCCalc
from dqc.xc.base_xc import BaseXC
from dqc.api.getxc import get_xc
from dqc.utils.datastruct import SpinParam

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
    restricted: bool or None
        If True, performing restricted Kohn-Sham DFT. If False, it performs
        the unrestricted Kohn-Sham DFT.
        If None, it will choose True if the system is unpolarized and False if
        it is polarized
    """

    def __init__(self, system: BaseSystem, xc: Union[str, BaseXC],
                 vext: Optional[torch.Tensor] = None,
                 restricted: Optional[bool] = None):

        # create the kohn-sham engine
        self.engine = _KSEngine(system, xc, vext, restricted)

        # misc info
        self.polarized = self.engine.polarized
        self.shape = self.engine.shape
        self.dtype = self.engine.dtype
        self.device = self.engine.device
        self.has_run = False

    def get_system(self) -> BaseSystem:
        return self.engine.get_system()

    def run(self, dm0: Optional[Union[torch.Tensor, SpinParam[torch.Tensor]]] = None,  # type: ignore
            eigen_options: Optional[Mapping[str, Any]] = None,
            fwd_options: Optional[Mapping[str, Any]] = None,
            bck_options: Optional[Mapping[str, Any]] = None) -> BaseQCCalc:

        # setup the default options
        if eigen_options is None:
            eigen_options = {
                "method": "exacteig"
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
        self.engine.set_eigen_options(eigen_options)

        # set up the initial self-consistent param guess
        if dm0 is None:
            if not self.polarized:
                dm0 = torch.zeros(self.shape, dtype=self.dtype,
                                  device=self.device)
            else:
                dm0_u = torch.zeros(self.shape, dtype=self.dtype,
                                    device=self.device)
                dm0_d = torch.zeros(self.shape, dtype=self.dtype,
                                    device=self.device)
                dm0 = SpinParam(u=dm0_u, d=dm0_d)

        scp0 = self.engine.dm2scp(dm0)

        # do the self-consistent iteration
        scp = xitorch.optimize.equilibrium(
            fcn=self.engine.scp2scp,
            y0=scp0,
            bck_options={**bck_options},
            **fwd_options)

        # post-process parameters
        self._dm = self.engine.scp2dm(scp)
        self.has_run = True
        return self

    def energy(self) -> torch.Tensor:
        # returns the total energy of the system
        return self.engine.dm2energy(self._dm)

    def aodm(self) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        # returns the density matrix in the atomic-orbital basis
        return self._dm

class _KSEngine(xt.EditableModule):
    """
    Private class of Engine to be used with KS.
    This class provides the calculation of the self-consistency iteration step
    and the calculation of the post-calculation properties.

    The reason of this class' existence is the leak in PyTorch:
    https://github.com/pytorch/pytorch/issues/52140
    which can be solved by making a different class than the class where the
    self-consistent iteration is performed.
    """
    def __init__(self, system: BaseSystem, xc: Union[str, BaseXC],
                 vext: Optional[torch.Tensor] = None,
                 restricted: Optional[bool] = None):

        # decide if this is restricted or not
        if restricted is None:
            self._polarized = bool(system.spin != 0)
        else:
            self._polarized = not restricted

        # get the xc object
        if isinstance(xc, str):
            self.xc: BaseXC = get_xc(xc)
        else:
            self.xc = xc
        self.system = system

        # build and setup basis and grid
        self.system.setup_grid()
        self.hamilton = system.get_hamiltonian()
        self.hamilton.build()
        self.hamilton.setup_grid(system.get_grid(), self.xc)

        # get the orbital info
        self.orb_weight = system.get_orbweight(polarized=self._polarized)  # (norb,)
        if self._polarized:
            assert isinstance(self.orb_weight, SpinParam)
            self.norb: Union[int, SpinParam[int]] = SpinParam(
                u=self.orb_weight.u.shape[-1],
                d=self.orb_weight.d.shape[-1])
        else:
            assert isinstance(self.orb_weight, torch.Tensor)
            self.norb = self.orb_weight.shape[-1]

        # set up the vext linear operator
        self.knvext_linop = self.hamilton.get_kinnucl()  # kinetic, nuclear, and external potential
        if vext is not None:
            assert vext.shape[-1] == system.get_grid().get_rgrid().shape[-2]
            self.knvext_linop = self.knvext_linop + self.hamilton.get_vext(vext)

    def get_system(self) -> BaseSystem:
        return self.system

    @property
    def shape(self):
        # returns the shape of the density matrix
        return self.knvext_linop.shape

    @property
    def dtype(self):
        # returns the dtype of the density matrix
        return self.knvext_linop.dtype

    @property
    def device(self):
        # returns the device of the density matrix
        return self.knvext_linop.device

    @property
    def polarized(self):
        return self._polarized

    def dm2scp(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        # convert from density matrix to a self-consistent parameter (scp)
        if isinstance(dm, torch.Tensor):  # unpolarized
            # scp is the fock matrix
            return self.__dm2fock(dm).fullmatrix()
        else:  # polarized
            # scp is the concatenated fock matrix
            fock = self.__dm2fock(dm)
            mat_u = fock.u.fullmatrix().unsqueeze(0)
            mat_d = fock.d.fullmatrix().unsqueeze(0)
            return torch.cat((mat_u, mat_d), dim=0)

    def scp2dm(self, scp: torch.Tensor) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        # convert the self-consistent parameter (scp) to the density matrix
        def _symm(scp: torch.Tensor):
            # forcely symmetrize the tensor
            return (scp + scp.transpose(-2, -1)) * 0.5

        if not self._polarized:
            fock = xt.LinearOperator.m(_symm(scp), is_hermitian=True)
            return self.__fock2dm(fock)
        else:
            fock_u = xt.LinearOperator.m(_symm(scp[0]), is_hermitian=True)
            fock_d = xt.LinearOperator.m(_symm(scp[1]), is_hermitian=True)
            return self.__fock2dm(SpinParam(u=fock_u, d=fock_d))

    def scp2scp(self, scp: torch.Tensor) -> torch.Tensor:
        # self-consistent iteration step from a self-consistent parameter (scp)
        # to an scp
        dm = self.scp2dm(scp)
        return self.dm2scp(dm)

    def set_eigen_options(self, eigen_options: Mapping[str, Any]) -> None:
        # set the eigendecomposition (diagonalization) option
        self.eigen_options = eigen_options

    def dm2energy(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        # calculate the energy given the density matrix
        fock = self.__dm2fock(dm)

        # get the energy from xc and get the potential
        e_exc = self.hamilton.get_exc(dm)
        vxc = self.hamilton.get_vxc(dm)

        if not self._polarized:
            assert isinstance(vxc, xt.LinearOperator)
            assert isinstance(self.orb_weight, torch.Tensor)
            assert isinstance(dm, torch.Tensor)

            # eivals: (..., norb), eivecs: (..., nao, norb)
            eivals, eivecs = self.__diagonalize(fock)
            e_eivals = torch.sum(eivals * self.orb_weight, dim=-1)

            # get the energy from xc potential
            e_vxc = torch.einsum("...rc,c,...rc->...", vxc.mm(eivecs), self.orb_weight, eivecs)

            # get the energy from electron repulsion
            elrep = self.hamilton.get_elrep(dm)
            e_elrep = 0.5 * torch.einsum("...rc,c,...rc->...", elrep.mm(eivecs), self.orb_weight, eivecs)
        else:
            assert isinstance(vxc, SpinParam)
            assert isinstance(self.orb_weight, SpinParam)
            assert isinstance(dm, SpinParam)

            # eivals: (..., norb), eivecs: (..., nao, norb)
            eivals, eivecs = self.__diagonalize(fock)
            e_eivals_u = torch.sum(eivals.u * self.orb_weight.u, dim=-1)
            e_eivals_d = torch.sum(eivals.d * self.orb_weight.d, dim=-1)
            e_eivals = e_eivals_u + e_eivals_d

            # get the energy from xc potential
            e_vxc_u = torch.einsum("...rc,c,...rc->...", vxc.u.mm(eivecs.u), self.orb_weight.u, eivecs.u)
            e_vxc_d = torch.einsum("...rc,c,...rc->...", vxc.d.mm(eivecs.d), self.orb_weight.d, eivecs.d)
            e_vxc = e_vxc_u + e_vxc_d

            # get the energy from electron repulsion
            elrep = self.hamilton.get_elrep(dm.u + dm.d)
            e_elrep_u = 0.5 * torch.einsum("...rc,c,...rc->...", elrep.mm(eivecs.u), self.orb_weight.u, eivecs.u)
            e_elrep_d = 0.5 * torch.einsum("...rc,c,...rc->...", elrep.mm(eivecs.d), self.orb_weight.d, eivecs.d)
            e_elrep = e_elrep_u + e_elrep_d

        # compute the total energy
        e_tot = e_eivals + (e_exc - e_vxc) - e_elrep + self.system.get_nuclei_energy()
        return e_tot

    @overload
    def __dm2fock(self, dm: torch.Tensor) -> xt.LinearOperator:
        ...

    @overload
    def __dm2fock(self, dm: SpinParam[torch.Tensor]) -> SpinParam[xt.LinearOperator]:
        ...

    def __dm2fock(self, dm):
        if isinstance(dm, torch.Tensor):
            # construct the fock matrix from the density matrix
            elrep = self.hamilton.get_elrep(dm)  # (..., nao, nao)
            vxc = self.hamilton.get_vxc(dm)
            return self.knvext_linop + elrep + vxc
        else:
            elrep = self.hamilton.get_elrep(dm.u + dm.d)  # (..., nao, nao)
            vext_elrep = self.knvext_linop + elrep

            vxc_ud = self.hamilton.get_vxc(dm)
            return SpinParam(u=vext_elrep + vxc_ud.u, d=vext_elrep + vxc_ud.d)

    @overload
    def __fock2dm(self, fock: xt.LinearOperator) -> torch.Tensor:
        ...

    @overload
    def __fock2dm(self, fock: SpinParam[xt.LinearOperator]) -> SpinParam[torch.Tensor]:  # type: ignore
        ...

    def __fock2dm(self, fock):
        # diagonalize the fock matrix and obtain the density matrix
        eigvals, eigvecs = self.__diagonalize(fock)
        if isinstance(eigvecs, torch.Tensor):  # unpolarized
            assert isinstance(self.orb_weight, torch.Tensor)
            dm = self.hamilton.ao_orb2dm(eigvecs, self.orb_weight)
            return dm
        else:  # polarized
            assert isinstance(self.orb_weight, SpinParam)
            dm_u = self.hamilton.ao_orb2dm(eigvecs.u, self.orb_weight.u)
            dm_d = self.hamilton.ao_orb2dm(eigvecs.d, self.orb_weight.d)
            return SpinParam(u=dm_u, d=dm_d)

    @overload
    def __diagonalize(self, fock: xt.LinearOperator) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def __diagonalize(self, fock: SpinParam[xt.LinearOperator]  # type: ignore
                      ) -> Tuple[SpinParam[torch.Tensor], SpinParam[torch.Tensor]]:
        ...

    def __diagonalize(self, fock):
        ovlp = self.hamilton.get_overlap()
        if isinstance(fock, SpinParam):
            assert isinstance(self.norb, SpinParam)
            eivals_u, eivecs_u = xitorch.linalg.lsymeig(
                A=fock.u,
                neig=self.norb.u,
                M=ovlp,
                **self.eigen_options)
            eivals_d, eivecs_d = xitorch.linalg.lsymeig(
                A=fock.d,
                neig=self.norb.d,
                M=ovlp,
                **self.eigen_options)
            return SpinParam(u=eivals_u, d=eivals_d), SpinParam(u=eivecs_u, d=eivecs_d)
        else:
            return xitorch.linalg.lsymeig(
                A=fock,
                neig=self.norb,
                M=ovlp,
                **self.eigen_options)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "scp2scp":
            return self.getparamnames("scp2dm", prefix=prefix) + \
                self.getparamnames("dm2scp", prefix=prefix)
        elif methodname == "scp2dm":
            return self.getparamnames("__fock2dm", prefix=prefix)
        elif methodname == "dm2scp":
            return self.getparamnames("__dm2fock", prefix=prefix)
        elif methodname == "__fock2dm":
            if self._polarized:
                params = [prefix + "orb_weight.u", prefix + "orb_weight.d"]
            else:
                params = [prefix + "orb_weight"]
            return self.getparamnames("__diagonalize", prefix=prefix) + \
                self.hamilton.getparamnames("ao_orb2dm", prefix=prefix + "hamilton.") + \
                params
        elif methodname == "__dm2fock":
            hprefix = prefix + "hamilton."
            return self.hamilton.getparamnames("get_elrep", prefix=hprefix) + \
                self.hamilton.getparamnames("get_vxc", prefix=hprefix) + \
                self.knvext_linop._getparamnames(prefix=prefix + "knvext_linop.")
        elif methodname == "__diagonalize":
            return self.hamilton.getparamnames("get_overlap", prefix=prefix + "hamilton.")
        else:
            raise KeyError("Method %s has no paramnames set" % methodname)
        return []  # TODO: to complete
