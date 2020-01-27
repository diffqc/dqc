import torch

def rfft(sig, signal_ndim, normalized=False):
    if signal_ndim == 1:
        return rfft1d(sig, normalized=normalized)
    elif signal_ndim == 2:
        return rfft2d(sig, normalized=normalized)
    elif signal_ndim == 3:
        return rfft3d(sig, normalized=normalized)
    else:
        raise RuntimeError("The argument signal_ndim must be an integer between 1 to 3")

def irfft(tsig, signal_ndim, normalized=False):
    if signal_ndim == 1:
        return irfft1d(tsig, normalized=normalized)
    elif signal_ndim == 2:
        return irfft2d(tsig, normalized=normalized)
    elif signal_ndim == 3:
        return irfft3d(tsig, normalized=normalized)
    else:
        raise RuntimeError("The argument signal_ndim must be an integer between 1 to 3")

def rfft1d(sig, normalized):
    # sig: (...,nx)
    sigshape = sig.shape

    # sig now is (nbatch, nx)
    sig = sig.contiguous().view(-1, sigshape[-1])
    sigft = torch.rfft(sig, signal_ndim=1, onesided=True,
            normalized=normalized) # (nbatch,nx//2+1,2)

    # for real fft, there are still redundancy even if we put onesided=True
    # for even n, the redundancies are in imag(X_0)=0 and imag(X_(N/2))=0
    # for odd n, the redundancy is in imag(X_0)=0
    is_even = sigshape[-1] % 2 == 0
    sigft_eff = _remove_redundancy(is_even, sigft) # (nbatch,nx)

    # reshape to the original shape
    return sigft_eff.view(*sigshape)

def irfft1d(tsig, normalized):
    # tsig: (...,nx)
    tsigshape = tsig.shape

    # tsig now is (nbatch, nx)
    tsig = tsig.contiguous().view(-1, tsigshape[-1])

    # add the redundancy
    is_even = tsigshape[-1] % 2 == 0
    tsigred = _add_redundancy(is_even, tsig)

    sig = torch.irfft(tsigred, signal_ndim=1, onesided=True,
            normalized=normalized, signal_sizes=(tsigshape[-1],))
    sig = sig.view(*tsigshape)
    return sig

def rfft2d(sig, normalized):
    sig1 = rfft1d(sig, normalized)
    sig2 = rfft1d(sig1.transpose(-2,-1), normalized)
    return sig2.transpose(-2,-1).contiguous()

def irfft2d(tsig, normalized):
    tsig1 = irfft1d(tsig, normalized)
    tsig2 = irfft1d(tsig1.transpose(-2,-1), normalized)
    return tsig2.transpose(-2,-1).contiguous()

def rfft3d(sig, normalized):
    sig1 = rfft1d(sig, normalized)
    sig2 = rfft1d(sig1.transpose(-2,-1), normalized)
    sig3 = rfft1d(sig2.transpose(-3,-1), normalized)
    return sig3.transpose(-3,-1).transpose(-2,-1).contiguous()

def irfft3d(tsig, normalized):
    tsig1 = irfft1d(tsig, normalized)
    tsig2 = irfft1d(tsig1.transpose(-2,-1), normalized)
    tsig3 = irfft1d(tsig2.transpose(-3,-1), normalized)
    return tsig3.transpose(-3,-1).transpose(-2,-1).contiguous()

def _remove_redundancy(is_even, sig):
    # sig: (nbatch,nx//2+1, 2)
    sigflat = sig.view(sig.shape[0], -1) # (nbatch,(nx//2)*2+2)

    # remove the redundancy
    if is_even:
        sigeff = torch.cat((sigflat[:,:1], sigflat[:,2:-1]), dim=-1)
    else:
        sigeff = torch.cat((sigflat[:,:1], sigflat[:,2:]), dim=-1)
    return sigeff

def _add_redundancy(is_even, tsigbox):
    # tsigbox: (...,nx,ny,nz)
    tsigbox_flat = tsigbox.view(-1,tsigbox.shape[-1]) # (...*nx*ny,nz)
    redundancy = torch.zeros(tsigbox_flat.shape[0], 1).to(tsigbox.dtype).to(tsigbox.device)
    if is_even:
        tsigbox_flat = torch.cat((tsigbox_flat[:,:1], redundancy, tsigbox_flat[:,1:], redundancy), dim=-1) # (...*nx*ny,nz+2)
    else:
        tsigbox_flat = torch.cat((tsigbox_flat[:,:1], redundancy, tsigbox_flat[:,1:]), dim=-1) # (...*nx*ny,nz+1)
    tsigbox = tsigbox_flat.view(*tsigbox.shape[:-1], -1, 2) # (...,nx,ny,nz//2+1,2)
    return tsigbox # (...,nx,ny,nz//2+1,2)

if __name__ == "__main__":
    dtype = torch.float64
    normalized = False
    a = torch.rand(2,4,4).to(dtype)
    aft1 = rfft1d(a, normalized)
    a1 = irfft1d(aft1, normalized)
    assert (a1 - a).abs().sum() < 1e-10
    aft2 = rfft2d(a, normalized)
    a2 = irfft2d(aft2, normalized)
    assert (a2 - a).abs().sum() < 1e-10
    aft3 = rfft3d(a, normalized)
    a3 = irfft3d(aft3, normalized)
    assert (a3 - a).abs().sum() < 1e-10
