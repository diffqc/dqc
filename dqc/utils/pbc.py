import torch
import numpy as np

# functions usually used for pbc
# typically helper functions are listed within the same file, but if they are
# used in a multiple files, then it should be put under the dqc.utils folder

def unweighted_coul_ft(gvgrids: torch.Tensor) -> torch.Tensor:
    # Returns the unweighted fourier transform of the coulomb kernel: 4*pi/|gv|^2
    # If |gv| == 0, then it is 0.
    # gvgrids: (ngv, ndim)
    # returns: (ngv,)
    gnorm2 = torch.einsum("xd,xd->x", gvgrids, gvgrids)
    gnorm2[gnorm2 < 1e-12] = float("inf")
    coulft = 4 * np.pi / gnorm2
    return coulft
