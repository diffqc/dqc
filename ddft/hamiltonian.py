# This file contains function that calculates the Hamiltonian and produce the
# density or wavefunction

import torch
from ddft.utils.kinetics import get_K_matrix_1d

class Hamilton1P1D(torch.autograd.Function):
    """
    Solving Schrodinger equation for 1 particle in 1 dimension.
    The function returns either wavefunction or density at the given grid.
    It assumes periodic grid.

    Arguments
    ---------
    * rgrid: torch.tensor (nbatch, nr)
        The spatial grid of the space. It is assumed that it has the regular
        spacing.
    * vext: torch.tensor (nbatch, nr)
        The external potential experienced by the particle.
    * iexc: torch.tensor int (nbatch,)
        The index of the excited state. Choose 0 or None for ground state.
        It cannot exceed (nr-1). Default: None
    * return_wv: bool
        If true, then return the wavefunction. Otherwise, it returns the
        density. Default: True

    Returns
    -------
    * wf_or_dens: torch.tensor (nbatch, nr)
        Wavefunction or density.
    * e: torch.tensor (nbatch,)
        The energy of the wavefunction or density
    """

    @staticmethod
    def forward(ctx, rgrid, vext, iexc=None, return_wf=True):
        if type(iexc) == torch.Tensor and len(iexc.shape) == 1:
            iexc = iexc.unsqueeze(1)

        wf_or_n, e = HamiltonNP1D.forward(ctx, rgrid, vext, iexc, return_wf)
        if return_wf:
            return wf_or_n[:,:,0], e
        else:
            return wf_or_n, e

    @staticmethod
    def backward(ctx, grad_wf_or_n, grad_e):
        if ctx.return_wf:
            grad_wf_or_n = grad_wf_or_n.unsqueeze(-1)
        return HamiltonNP1D.backward(ctx, grad_wf_or_n, grad_e)

class HamiltonNP1D(torch.autograd.Function):
    """
    Solving Schrodinger equation for N non-interacting particles in 1 dimension.
    The function returns either wavefunction or density at the given grid.
    It assumes periodic grid.

    Arguments
    ---------
    * rgrid: torch.tensor (nbatch, nr)
        The spatial grid of the space. It is assumed that it has the regular
        spacing.
    * vext: torch.tensor (nbatch, nr)
        The external potential experienced by the particle.
    * iexc: torch.tensor or int (nbatch, np)
        The indices of the excited state. Choose 0 or None for an electron in
        ground state only. If it is an integer, then it only takes an electron
        in the iexc-th excited state. Each element cannot exceed (nr-1).
        Default: None
    * return_wv: bool
        If true, then return the wavefunction. Otherwise, it returns the total
        density. Default: True
    * return_hamiltonian: bool
        If true, then return the Hamiltonian. Default: False

    Returns
    -------
    * wf_or_dens: torch.tensor (nbatch, nr, np) or (nbatch, nr)
        Wavefunction or total density.
    * e: torch.tensor (nbatch, np)
        The energy of the wavefunction or density
    * H: torch.tensor (nb, nr, nr) (only if return_hamiltonian==True)
        The constructed Hamiltonian of the system.
    """

    @staticmethod
    def forward(ctx, rgrid, vext, iexc=None, return_wf=True,
            return_hamiltonian=False):
        nb = rgrid.shape[0]
        nr = rgrid.shape[-1]
        dr = rgrid[:,1] - rgrid[:,0] # (nb, )
        inb = torch.arange(nb)
        if iexc is None:
            iexc = torch.zeros((nb, 1), dtype=torch.long)
        elif type(iexc) == int:
            iexc = torch.zeros((nb, 1), dtype=torch.long) + iexc

        # construct the hamiltonian
        K = get_K_matrix_1d(nr, dtype=rgrid.dtype, kspace=False, periodic=True).expand(nb, nr, nr) # (nb, nr, nr)
        Vext = (torch.eye(nr, dtype=rgrid.dtype) * vext).expand(nb, nr, nr)
        H = (K / (dr*dr).unsqueeze(-1).unsqueeze(-1)) + Vext # (nb, nr, nr)

        # obtain the eigenvectors and eigenvalues
        eigvals, eigvecs = torch.symeig(H, eigenvectors=True) # eval: (nb, nr), evec: (nb, nr, nr)
        e = torch.gather(eigvals, dim=1, index=iexc) # (nb, np)
        wf = torch.gather(eigvecs, dim=2, index=iexc.unsqueeze(1).expand(-1, nr, -1)) # (nb, nr, np)

        # save the variables needed for backward
        ctx.e = e
        ctx.wf = wf
        ctx.H = H
        ctx.return_wf = return_wf
        ctx.return_hamiltonian = return_hamiltonian

        if return_wf:
            wf_or_n = wf
        else:
            # calculate the density
            n = (wf * wf).sum(dim=-1)
            wf_or_n = n

        if return_hamiltonian:
            return wf_or_n, e, H
        else:
            return wf_or_n, e

    @staticmethod
    def backward(ctx, *grads):
        # unpack the gradients
        if ctx.return_hamiltonian:
            grad_wf_or_n, grad_e, grad_H = grads
        else:
            grad_wf_or_n, grad_e = grads
            grad_H = None

        nb = ctx.e.shape[0]
        np = ctx.e.shape[1]
        nr = ctx.H.shape[1]
        wf = ctx.wf

        grad_vext = _apply_G(ctx.wf, ctx.e, ctx.H, grad_wf_or_n,
                              grad_e = grad_e,
                              grad_H = grad_H,
                              is_grad_wf = ctx.return_wf)
        return (None, grad_vext, None, None)

def _apply_G(wfs, es, H, grad_wf_or_n, grad_e=None, grad_H=None, is_grad_wf=True):
    nb, np = es.shape[:2]
    nr = H.shape[1]

    # iterate over the number of particles
    for b in range(np):
        # get the wavefunction and energy of this particle
        wf = wfs[:,:,b]
        e = es[:,b]

        # construct the matrix kernel
        eI = e.repeat_interleave(nr).view(nb,nr).diag_embed()
        A = eI - H
        Ainv = torch.pinverse(A, rcond=1e-5) # (nb, nr, nr)

        # calculate the grad_wf for this particle
        if not is_grad_wf:
            grad_wfb = grad_wf_or_n * 2 * wf
        else:
            grad_wfb = grad_wf_or_n[:,:,b] # (nb, nr)

        # apply the inverse operation
        # n.b.: the inverse output must be orthogonal to wf
        gradb = grad_wfb.unsqueeze(-1)
        Awf = torch.bmm(Ainv, gradb) # (nb, nr, 1)
        Awf = Awf.squeeze(-1) # (nb, nr)
        # orthogonalize w.r.t. wf
        # wf already has norm == 1
        proj_Awf = (Awf * wf).sum(dim=-1, keepdim=True) * wf
        Awf_ortho = Awf - proj_Awf
        grad_vext_b = Awf_ortho * wf

        # add the contribution from grad_e
        if grad_e is not None:
            grad_vext_e = wf * wf * grad_e[:,b]
            grad_vext_b = grad_vext_b + grad_vext_e

        # accummulate the gradient
        if b == 0:
            grad_vext = grad_vext_b
        else:
            grad_vext = grad_vext + grad_vext_b

    # add the contribution from grad_H
    if grad_H is not None:
        grad_vext_H = torch.diagonal(grad_H, dim1=-2, dim2=-1)
        grad_vext = grad_vext + grad_vext_H

    return grad_vext
