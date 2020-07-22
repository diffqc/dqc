#ifndef DDFT_GTO_COEFFS_H
#define DDFT_GTO_COEFFS_H

#include <torch/torch.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace torch::indexing;

torch::Tensor get_overlap_mat(int ijk_left_max, int ijk_right_max, int max_basis, int ndim,
    torch::Tensor& ijk_pairs,
    // arguments for get_ecoeff
    torch::Tensor& alpha, torch::Tensor& betas, torch::Tensor& gamma,
    torch::Tensor& kappa, torch::Tensor& qab,
    py::dict& e_memory, py::str& key_format);

torch::Tensor get_kinetics_mat(int ijk_left_max, int ijk_right_max, int max_basis, int ndim,
    torch::Tensor& ijk_pairs,
    // arguments for get_ecoeff
    torch::Tensor& alpha, torch::Tensor& betas, torch::Tensor& gamma,
    torch::Tensor& kappa, torch::Tensor& qab,
    py::dict& e_memory, py::str& key_format);

// coefficients
torch::Tensor get_ecoeff(int i, int j, int t, int xyz,
    torch::Tensor& alpha, torch::Tensor& betas, torch::Tensor& gamma,
    torch::Tensor& kappa, torch::Tensor& qab,
    py::dict& e_memory, py::str& key_format);

torch::Tensor get_rcoeff(int r, int s, int t, int n,
    torch::Tensor& rcd, torch::Tensor& rcd_sq, torch::Tensor& gamma,
    py::dict& r_memory, py::str& key_format);

// helper functions
torch::Tensor boys(int n, torch::Tensor t);
torch::Tensor incgamma(double n, torch::Tensor t);
#endif
