import torch
import numpy as np
from ddft.spaces.qspace import QSpace

def fcntest_qspace(nx, ndim, boxsize, nfreq=1.0, sin=True):
    dtype = torch.float64
    x = torch.linspace(-boxsize/2.0, boxsize/2.0, nx+1)[:-1].to(dtype)
    rgrids = torch.meshgrid(*[x for i in range(ndim)]) # (nx,nx)
    rgrid = torch.cat([rgridx.unsqueeze(-1) for rgridx in rgrids], dim=-1).view(-1,ndim) # (nr,ndim)

    # construct the space
    space = QSpace(rgrid, [nx for i in range(ndim)])

    # construct the wavefunction as wf = sumj(sin(2*pi*xj))
    if sin:
        wf = torch.sin(rgrid * 2 * np.pi * nfreq).unsqueeze(0) # (1,nr,ndim)
    else:
        wf = torch.cos(rgrid * 2 * np.pi * nfreq).unsqueeze(0) # (1,nr,ndim)
    wf = torch.sum(wf, dim=-1, keepdim=True) # (1,nr,1)
    wfq = space.transformsig(wf, dim=1) # (1,ns,1)

    assert torch.allclose(space.invtransformsig(wfq, dim=1), wf, atol=1e-5)
    assert torch.allclose(space.transformsig(space.invtransformsig(wfq, dim=1), dim=1), wfq, atol=1e-5)

def test_qspace_1d():
    fcntest_qspace(101, 1, 5.0)
    fcntest_qspace(100, 1, 5.0)

def test_qspace_2d():
    fcntest_qspace(51, 2, 5.0)
    fcntest_qspace(50, 2, 5.0)

def test_qspace_3d():
    fcntest_qspace(31, 3, 5.0)
    fcntest_qspace(30, 3, 5.0)
