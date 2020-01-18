import torch
import numpy as np
import matplotlib.pyplot as plt
from ddft.spaces.qspace import QSpace
from ddft.hamiltons.hamiltonpw import HamiltonPlaneWave

def test_hamilton_pw_kinetics():
    dtype = torch.float64
    nr = 100
    ndim = 1
    rgrid = torch.linspace(-2, 2, nr+1)[:-1].unsqueeze(-1).to(dtype) # (nr,ndim)
    wf = torch.sin(rgrid * 2 * np.pi).unsqueeze(0) # (1,nr,ndim)

    space = QSpace(rgrid, (rgrid.shape[0],))
    h = HamiltonPlaneWave(space)
    kin = h.kinetics(wf) # (1,nr,ndim)

    dr = rgrid.squeeze()[1] - rgrid.squeeze()[0]
    dq = space.qgrid.squeeze()[1] - space.qgrid.squeeze()[0]
    assert torch.allclose(dr*dq / (2*np.pi/nr), torch.ones_like(dr), rtol=1e-4)
    print(rgrid.squeeze())
    print(space.qgrid.squeeze())

    assert torch.allclose(wf+kin/(2*np.pi)**2, torch.zeros_like(wf), atol=1e-4)
