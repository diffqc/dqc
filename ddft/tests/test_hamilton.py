import torch
import numpy as np
import matplotlib.pyplot as plt
from ddft.spaces.qspace import QSpace
from ddft.hamiltons.hamiltonpw import HamiltonPlaneWave

def compare_hamilton_pw_kinetics(nx, ndim, boxsize=4.0):
    dtype = torch.float64
    x = torch.linspace(-boxsize/2.0, boxsize/2.0, nx+1)[:-1].to(dtype)
    rgrids = torch.meshgrid(*[x for i in range(ndim)]) # (nx,nx)
    rgrid = torch.cat([rgridx.unsqueeze(-1) for rgridx in rgrids], dim=-1).view(-1,ndim) # (nr,ndim)

    # construct the wavefunction as wf = sumj(sin(2*pi*xj))
    wf = torch.sin(rgrid * 2 * np.pi).unsqueeze(0) # (1,nr,ndim)
    wf = torch.sum(wf, dim=-1, keepdim=True) # (1,nr,1)

    space = QSpace(rgrid, [nx for i in range(ndim)])
    h = HamiltonPlaneWave(space)
    kin = h.kinetics(wf) # (1,nr,1)

    assert torch.allclose(wf-2*kin/(2*np.pi)**2, torch.zeros_like(wf), atol=1e-4)

def test_hamilton_pw_kinetics():
    compare_hamilton_pw_kinetics(100, 1)

def test_hamilton_pw_kinetics2d():
    compare_hamilton_pw_kinetics(50, 2)

def test_hamilton_pw_kinetics3d():
    compare_hamilton_pw_kinetics(30, 3)
