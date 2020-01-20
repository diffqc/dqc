import torch
import numpy as np
import matplotlib.pyplot as plt
from ddft.spaces.qspace import QSpace
from ddft.hamiltons.hamiltonpw import HamiltonPlaneWave

def setup_hamilton(nx, ndim, boxsize):
    dtype = torch.float64
    x = torch.linspace(-boxsize/2.0, boxsize/2.0, nx+1)[:-1].to(dtype)
    rgrids = torch.meshgrid(*[x for i in range(ndim)]) # (nx,nx)
    rgrid = torch.cat([rgridx.unsqueeze(-1) for rgridx in rgrids], dim=-1).view(-1,ndim) # (nr,ndim)

    # construct the space
    space = QSpace(rgrid, [nx for i in range(ndim)])

    # construct the wavefunction as wf = sumj(sin(2*pi*xj))
    wf = torch.sin(rgrid * 2 * np.pi).unsqueeze(0) # (1,nr,ndim)
    wf = torch.sum(wf, dim=-1, keepdim=True) # (1,nr,1)
    wfq = space.transformsig(wf, dim=1) # (1,ns,1)

    h = HamiltonPlaneWave(space)
    return wf, wfq, h, space


def compare_hamilton_pw_kinetics(nx, ndim, boxsize=4.0):
    wf, wfq, h, space = setup_hamilton(nx, ndim, boxsize)
    vext = torch.zeros_like(wf).squeeze(-1)

    kinq = h(wfq, vext) # (1,ns,1)
    kin = space.invtransformsig(kinq, dim=1)

    a = wf-2*kin/(2*np.pi)**2
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-4)

def compare_hamilton_pw_vext(nx, ndim, boxsize=4.0):
    wf, wfq, h, space = setup_hamilton(nx, ndim, boxsize)
    rgrid = space.rgrid # (nr, ndim)
    rgrid_norm = rgrid.norm(dim=-1) # (nr,)
    vext = (rgrid_norm * rgrid_norm * 0.5).unsqueeze(0) # (1,nr)

    hq = h(wfq, vext) # (1,ns,1)
    hr = space.invtransformsig(hq, dim=1) # (1,nr,1)

    kin = wf*(2*np.pi)**2/2.0
    vext_wf = hr-kin
    vext_wf2 = vext.unsqueeze(-1)*wf
    torch.allclose(vext_wf, vext_wf2, atol=1e-4)

def test_hamilton_pw_kinetics():
    compare_hamilton_pw_kinetics(100, 1)

def test_hamilton_pw_kinetics2d():
    compare_hamilton_pw_kinetics(50, 2)

def test_hamilton_pw_kinetics3d():
    compare_hamilton_pw_kinetics(30, 3)

def test_hamilton_pw_vext():
    compare_hamilton_pw_vext(100, 1)

def test_hamilton_pw_vext2d():
    compare_hamilton_pw_vext(50, 2)

def test_hamilton_pw_vext3d():
    compare_hamilton_pw_vext(30, 3)
