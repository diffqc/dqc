import dqc
import torch
import xitorch as xt
import xitorch.optimize

basis = {
    "H": dqc.loadbasis("1:3-21G"),  # load 3-21G basis for atomz = 1
}
bpacker = xt.Packer(basis)  # use xitorch's Packer to get the tensors within a structure
bparams = bpacker.get_param_tensor()  # get the parameters of the basis as one tensor

def fcn(bparams, bpacker):
    # returns the same structure as basis above, but the parameters (alphas
    # and coeffs) are changed according to values in bparams
    basis = bpacker.construct_from_tensor(bparams)

    m = dqc.Mol("H 1 0 0; H -1 0 0", basis=basis)
    qc = dqc.HF(m).run()
    ene = qc.energy()
    return ene

print("Original basis")
print(basis)
min_bparams = xitorch.optimize.minimize(fcn, bparams, (bpacker,), method="gd",
                                        step=2e-1, maxiter=100, verbose=True)
opt_basis = bpacker.construct_from_tensor(min_bparams)
print("Optimized basis")
print(opt_basis)
