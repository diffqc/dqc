import torch

def set_default_option(defopt, opt=None):
    if opt is None:
        opt = {}
    defopt.update(opt)
    return defopt

def get_uniform_density(rgrid, focc):
    # rgrid: (nr,)
    # focc: (nbatch, nlowest)
    nbatch = focc.shape[0]
    nr = rgrid.shape[0]

    nels = focc.sum(dim=-1, keepdim=True) # (nbatch, 1)
    dr = rgrid[1] - rgrid[0]
    density_val = nels / dr / nr # (nbatch, 1)
    density = torch.zeros((nbatch, nr)).to(rgrid.dtype).to(rgrid.device) + density_val

    return density
