from typing import List
from dqc.hamilton.intor.lcintwrap import LibcintWrapper
from dqc.hamilton.intor.utils import estimate_g_cutoff

# contains functions that works in relation to LibcintWrapper

def get_gcut(precision: float, wrappers: List[LibcintWrapper], reduce: str = "min") -> float:
    # get the G-point cut-off from the given wrappers where the FT
    # eval/integration is going to be performed
    gcuts: List[float] = []
    for wrapper in wrappers:
        # TODO: using params here can be confusing because wrapper.params
        # returns all parameters (even if it is a subset)
        coeffs, alphas, _ = wrapper.params
        gcut_wrap = estimate_g_cutoff(precision, coeffs, alphas)
        gcuts.append(gcut_wrap)
    if len(gcuts) == 1:
        return gcuts[0]
    if reduce == "min":
        return min(*gcuts)
    elif reduce == "max":
        return max(*gcuts)
    else:
        raise ValueError("Unknown reduce: %s" % reduce)
