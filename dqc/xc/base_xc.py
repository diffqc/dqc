from abc import abstractmethod, abstractproperty
import torch
import xitorch as xt
from typing import List, Union, Tuple, overload
from dqc.utils.datastruct import ValGrad

class BaseXC(xt.EditableModule):
    """
    XC is class that calculates the components of xc potential and energy
    density given the density.
    """
    @abstractproperty
    def family(self) -> int:
        """
        Returns 1 for LDA, 2 for GGA, and 3 for Meta-GGA.
        """
        pass

    @overload
    def get_vxc(self, densinfo: ValGrad) -> ValGrad:
        ...

    @overload
    def get_vxc(self, densinfo: Tuple[ValGrad, ValGrad]) -> Tuple[ValGrad, ValGrad]:
        ...

    @abstractmethod
    def get_vxc(self, densinfo):
        """
        Returns the ValGrad for the xc potential given the density info
        for unpolarized case.
        """
        # densinfo.value & lapl: (*BD, nr)
        # densinfo.grad: (*BD, nr, ndim)
        # return:
        # potentialinfo.value & lapl: (*BD, nr)
        # potentialinfo.grad: (*BD, nr, ndim)
        pass

    @abstractmethod
    def get_edensityxc(self, densinfo: Union[ValGrad, Tuple[ValGrad, ValGrad]]) -> \
            torch.Tensor:
        """
        Returns the xc energy density (energy per unit volume)
        """
        # densinfo.value & lapl: (*BD, nr)
        # densinfo.grad: (*BD, nr, ndim)
        # return: (*BD, nr)
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass

    # special operations
    def __add__(self, other):
        return AddBaseXC(self, other)

class AddBaseXC(BaseXC):
    def __init__(self, a: BaseXC, b: BaseXC) -> None:
        self.a = a
        self.b = b

    @overload
    def get_vxc(self, densinfo: ValGrad) -> ValGrad:
        ...

    @overload
    def get_vxc(self, densinfo: Tuple[ValGrad, ValGrad]) -> Tuple[ValGrad, ValGrad]:
        ...

    def get_vxc(self, densinfo):
        avxc = self.a.get_vxc(densinfo)
        bvxc = self.b.get_vxc(densinfo)

        if isinstance(densinfo, tuple):
            return (avxc[0] + bvxc[0], avxc[1] + bvxc[1])
        else:
            return avxc + bvxc

    def get_edensityxc(self, densinfo: Union[ValGrad, Tuple[ValGrad, ValGrad]]) -> \
            torch.Tensor:
        return self.a.get_edensityxc(densinfo) + self.b.get_edensityxc(densinfo)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        return self.a.getparamnames(methodname, prefix=prefix + "a.") + \
            self.b.getparamnames(methodname, prefix=prefix + "b.")
