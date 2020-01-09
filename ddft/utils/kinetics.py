# This file contains function to generate or apply the Kinetics part of the
# Hamiltonian

import torch

def get_K_matrix_1d(n, dtype=torch.float, kspace=False, periodic=True):
    if not kspace:
        if periodic:
            K0 = torch.eye(n, dtype=dtype)
            K = K0 - 0.5 * roll_1(K0, 1) - 0.5 * roll_1(K0, -1)
            return K
        else:
            raise RuntimeError("Unimplemented option of periodic=False")
    else:
        if not periodic:
            raise ValueError("The boundary must be periodic for k-space calculation.")
        raise RuntimeError("Unimplemented option of kspace=True")

def roll_1(x, n):
    """
    Roll to the right of the first dimension (zero-based).
    """
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)
