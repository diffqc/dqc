from __future__ import annotations
from abc import abstractmethod, abstractproperty
from typing import Optional, Mapping, Any, List, Union
import torch
import xitorch as xt
import xitorch.linalg
import xitorch.optimize
from dqc.system.base_system import BaseSystem
from dqc.qccalc.base_qccalc import BaseQCCalc
from dqc.utils.datastruct import SpinParam

class SCF_QCCalc(BaseQCCalc):
    """
    Performing Restricted or Unrestricted self-consistent field iteration
    (e.g. Hartree-Fock or Density Functional Theory)

    Arguments
    ---------
    engine: BaseSCFEngine
        The SCF engine
    restricted: bool or None
        If True, performing restricted SCF iteration. If False, it performs
        the unrestricted SCF iteration.
        If None, it will choose True if the system is unpolarized and False if
        it is polarized.
    """

    def __init__(self, engine: BaseSCFEngine):
        self._engine = engine
        self._polarized = engine.polarized
        self._shape = self._engine.shape
        self.dtype = self._engine.dtype
        self.device = self._engine.device
        self._has_run = False

    def get_system(self) -> BaseSystem:
        return self._engine.get_system()

    def run(self, dm0: Optional[Union[str, torch.Tensor, SpinParam[torch.Tensor]]] = "1e",  # type: ignore
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
        self._engine.set_eigen_options(eigen_options)

        # set up the initial self-consistent param guess
        if dm0 is None:
            dm = self._get_zero_dm()
        elif isinstance(dm0, str):
            if dm0 == "1e":  # initial density based on 1-electron Hamiltonian
                dm = self._get_zero_dm()
                scp0 = self._engine.dm2scp(dm)
                dm = self._engine.scp2dm(scp0)
            else:
                raise RuntimeError("Unknown dm0: %s" % dm0)
        else:
            dm = dm0

        # making it spin param for polarized and tensor for nonpolarized
        if isinstance(dm, torch.Tensor) and self._polarized:
            dm_u = dm * 0.5
            dm_d = dm * 0.5
            dm = SpinParam(u=dm_u, d=dm_d)
        elif isinstance(dm, SpinParam) and not self._polarized:
            dm = SpinParam.sum(dm)

        scp0 = self._engine.dm2scp(dm)

        # do the self-consistent iteration
        scp = xitorch.optimize.equilibrium(
            fcn=self._engine.scp2scp,
            y0=scp0,
            bck_options={**bck_options},
            **fwd_options)

        # post-process parameters
        self._dm = self._engine.scp2dm(scp)
        self._has_run = True
        return self

    def energy(self) -> torch.Tensor:
        # returns the total energy of the system
        assert self._has_run
        return self._engine.dm2energy(self._dm)

    def aodm(self) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        # returns the density matrix in the atomic-orbital basis
        assert self._has_run
        return self._dm

    def _get_zero_dm(self) -> Union[SpinParam[torch.Tensor], torch.Tensor]:
        # get the initial dm that are all zeros
        if not self._polarized:
            return torch.zeros(self._shape, dtype=self.dtype,
                               device=self.device)
        else:
            dm0_u = torch.zeros(self._shape, dtype=self.dtype,
                                device=self.device)
            dm0_d = torch.zeros(self._shape, dtype=self.dtype,
                                device=self.device)
            return SpinParam(u=dm0_u, d=dm0_d)

class BaseSCFEngine(xt.EditableModule):
    @abstractproperty
    def polarized(self) -> bool:
        """
        Returns if the system is polarized or not
        """
        pass

    @abstractproperty
    def shape(self):
        """
        Returns the shape of the density matrix in this engine.
        """
        pass

    @abstractproperty
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the tensors in this engine.
        """
        pass

    @abstractproperty
    def device(self) -> torch.device:
        """
        Returns the device of the tensors in this engine.
        """
        pass

    @abstractmethod
    def get_system(self) -> BaseSystem:
        """
        Returns the system involved in the engine.
        """
        pass

    @abstractmethod
    def dm2energy(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Calculate the energy from the given density matrix.
        """
        pass

    @abstractmethod
    def dm2scp(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Convert the density matrix into the self-consistent parameter (scp).
        Self-consistent parameter is defined as the parameter that is put into
        the equilibrium function, i.e. y in `y = f(y, x)`.
        """
        pass

    @abstractmethod
    def scp2dm(self, scp: torch.Tensor) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """
        Calculate the density matrix from the given self-consistent parameter (scp).
        """
        pass

    @abstractmethod
    def scp2scp(self, scp: torch.Tensor) -> torch.Tensor:
        """
        Calculate the next self-consistent parameter (scp) for the next iteration
        from the previous scp.
        """
        pass

    @abstractmethod
    def set_eigen_options(self, eigen_options: Mapping[str, Any]) -> None:
        """
        Set the options for the diagonalization (i.e. eigendecomposition).
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        List all the names of parameters used in the given method.
        """
        pass
