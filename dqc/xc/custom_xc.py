from abc import abstractmethod, abstractproperty
from typing import Union, List
import torch
from dqc.xc.base_xc import BaseXC
from dqc.utils.datastruct import ValGrad, SpinParam

class CustomXC(BaseXC, torch.nn.Module):
    """
    Base class of custom xc functional.
    """
    @abstractproperty
    def family(self) -> int:
        pass

    @abstractmethod
    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        pass

    def getparamnames(self, methodname: str = "", prefix: str = "") -> List[str]:
        if methodname == "get_edensityxc":
            pfix = prefix if not prefix.endswith(".") else prefix[:-1]
            names = [name for (name, param) in self.named_parameters(prefix=pfix)]
            return names
        else:
            return super().getparamnames(methodname, prefix=prefix)
