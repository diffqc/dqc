import torch

def set_default_option(defopt, opt=None):
    if opt is None:
        opt = {}
    defopt.update(opt)
    return defopt

def unpack(arr, nums):
    iarr = 0
    res = []
    for num in nums:
        if isinstance(num, int):
            res.append(arr[iarr:iarr+num])
            iarr = iarr + num
        elif isinstance(num, list):
            ns = sum(num)
            res.append(unpack(arr[iarr:iarr+ns], num))
            iarr = iarr + ns
    return res

@torch.jit.script
def cumsum_zero(x:torch.Tensor, dim:int=-1):
    if dim != -1:
        x = x.transpose(dim, -1)
    nx = x.shape[-1]
    res = torch.zeros_like(x).to(x.device)
    res[...,1:] = torch.cumsum(x[...,:-1], dim=-1)
    if dim != -1:
        res = res.transpose(dim, -1)
    return res
