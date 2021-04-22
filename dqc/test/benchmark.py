import torch
from dqc.api.properties import raman_spectrum
from dqc.system.mol import Mol
from dqc.qccalc.ks import KS
from dqc.utils.safeops import safepow
from dqc.xc.base_xc import BaseXC
from dqc.utils.datastruct import ValGrad, SpinParam

dtype = torch.float64

class LDAX(BaseXC):
    def __init__(self):
        self.a = -0.7385587663820223
        self.p = 4.0 / 3

    @property
    def family(self):
        return 1

    def get_edensityxc(self, densinfo):
        if isinstance(densinfo, ValGrad):
            rho = densinfo.value.abs()  # safeguarding from nan
            return self.a * safepow(rho, self.p)
            # return self.a * rho ** self.p
        else:
            return 0.5 * (self.get_edensityxc(densinfo.u * 2) + self.get_edensityxc(densinfo.d * 2))

    def getparamnames(self, methodname, prefix=""):
        return []

def get_qc():
    # run the self-consistent HF iteration for h2o
    atomzs = torch.tensor([8, 1, 1], dtype=torch.int64)
    # from CCCBDB (calculated geometry for H2O)
    atomposs = torch.tensor([
        [0.0, 0.0, 0.2156],
        [0.0, 1.4749, -0.8625],
        [0.0, -1.4749, -0.8625],
    ], dtype=dtype).requires_grad_()
    efield = torch.zeros(3, dtype=dtype).requires_grad_()
    grad_efield = torch.zeros((3, 3), dtype=dtype).requires_grad_()

    efields = (efield, grad_efield)
    mol = Mol(moldesc=(atomzs, atomposs), basis="3-21G", dtype=dtype, efield=efields)
    qc = KS(mol, xc=LDAX()).run()
    # qc = KS(mol, xc="lda_x").run()
    return qc

def get_raman():
    qc = get_qc()
    return raman_spectrum(qc)

def fcn1():
    a = torch.rand((50000,), dtype=dtype).requires_grad_()
    b = torch.rand((15, 50000,), dtype=dtype).requires_grad_()
    c = torch.rand((15, 50000,), dtype=dtype).requires_grad_()
    mat = torch.einsum("r,br,cr->bc", a, b, c)
    loss = mat ** 3

    dldb, = torch.autograd.grad(loss.sum(), (b,), create_graph=True)
    dldb2, = torch.autograd.grad(dldb.sum(), (b,), create_graph=True)
    dldb3, = torch.autograd.grad(dldb2.sum(), (b,), create_graph=True)

def fcn2():
    a = torch.rand((15, 15), dtype=dtype).requires_grad_()
    b = torch.rand((15, 50000,), dtype=dtype).requires_grad_()
    mat = torch.einsum("bc,br,cr->r", a, b, b)
    loss = mat ** 3

    dldb, = torch.autograd.grad(loss.sum(), (b,), create_graph=True)
    dldb2, = torch.autograd.grad(dldb.sum(), (b,), create_graph=True)
    dldb3, = torch.autograd.grad(dldb2.sum(), (b,), create_graph=True)

def fcn3():
    dens = torch.rand((53000,), dtype=dtype).requires_grad_()
    densinfo = ValGrad(value=dens)
    ldax = LDAX()
    vxc = ldax.get_vxc(densinfo)
    loss = vxc.value.sum()

    dldd, = torch.autograd.grad(loss, dens, create_graph=True)
    dldd2, = torch.autograd.grad(dldd.sum(), dens, create_graph=True)
    dldd3, = torch.autograd.grad(dldd2.sum(), dens, create_graph=True)

# fcn3()
get_raman()

# with torch.autograd.profiler.profile(profile_memory=True, record_shapes=True, with_stack=False) as prof:
#     get_raman()

# print(prof.table(sort_by="cpu_memory_usage"))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
