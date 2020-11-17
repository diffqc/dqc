#ifndef DDFT_COEFFS_CPP
#define DDFT_COEFFS_CPP

#include <torch/torch.h>
#include <pybind11/pybind11.h>
// #include <iostream>
#include <initializer_list>
#include <cmath>
#include "coeffs.h"

namespace py = pybind11;
using namespace torch::indexing;

/************** basis integral matrices **************/
torch::Tensor get_overlap_mat(int ijk_left_max, int ijk_right_max, int max_basis, int ndim,
    torch::Tensor& ijk_pairs, // (nbasis_tot, nbasis_tot, ndim)
    // arguments for get_ecoeff
    torch::Tensor& alpha, // (1, nbasis_tot)
    torch::Tensor& betas, // (nbasis_tot, 1)
    torch::Tensor& gamma, // (nbasis_tot, nbasis_tot)
    torch::Tensor& kappa, // (nbasis_tot, nbasis_tot)
    torch::Tensor& qab, // (nbasis_tot, nbasis_tot, ndim)
    py::dict& e_memory, py::str& key_format) {

  auto overlap_dim = torch::empty_like(qab); // (nbasis_tot, nbasis_tot, ndim)

  for (int i = 0; i < ijk_left_max+1; ++i) {
    for (int j = 0; j < ijk_right_max+1; ++j) {

      auto idx = (ijk_pairs == (i*max_basis + j)); // (nbasis_tot, nbasis_tot, ndim)
      for (int xyz = 0; xyz < ndim; ++xyz) {

        auto idxx = idx.select(/*dim=*/2,/*index=*/xyz);
        auto coeff = get_ecoeff(i, j, 0, xyz, alpha, betas, gamma, kappa, qab, e_memory, key_format);
        // overlap_dim[:,:,xyz][idxx] = coeff[idxx]
        overlap_dim.select(/*dim=*/2, /*index=*/xyz).masked_scatter_(idxx, coeff.index({idxx}));

      }

    }
  }
  auto res = overlap_dim.prod(/*dim=*/-1) * torch::pow(M_PI / gamma, 1.5); // (nbasis_tot, nbasis_tot)
  return res;
}

torch::Tensor get_kinetics_mat(int ijk_left_max, int ijk_right_max, int max_basis, int ndim,
    torch::Tensor& ijk_pairs,
    // arguments for get_ecoeff (return: (nbasis_tot, nbasis_tot))
    torch::Tensor& alpha, torch::Tensor& betas, torch::Tensor& gamma,
    torch::Tensor& kappa, torch::Tensor& qab,
    py::dict& e_memory, py::str& key_format) {

  auto kinetics_dim0 = torch::empty_like(qab); // (nbasis_tot, nbasis_tot, ndim)
  auto kinetics_dim1 = torch::empty_like(qab); // (nbasis_tot, nbasis_tot, ndim)
  auto kinetics_dim2 = torch::empty_like(qab); // (nbasis_tot, nbasis_tot, ndim)
  for (int i = 0; i < ijk_left_max+1; ++i) {
    for (int j = 0; j < ijk_right_max+1; ++j) {
      auto idx = (ijk_pairs == (i * max_basis + j)); // (nbasis_tot, nbasis_tot, ndim)

      for (int xyz = 0; xyz < ndim; ++xyz) {
        auto idxx = idx.select(2, xyz); // (nbasis_tot, nbasis_tot)
        auto sij = get_ecoeff(i, j  , 0, xyz, alpha, betas, gamma, kappa, qab, e_memory, key_format); // (nbasis_tot, nbasis_tot)
        auto  d1 = get_ecoeff(i, j-2, 0, xyz, alpha, betas, gamma, kappa, qab, e_memory, key_format);
        auto  d2 = get_ecoeff(i, j+2, 0, xyz, alpha, betas, gamma, kappa, qab, e_memory, key_format);
        auto dij = j*(j-1)*d1 - 2*(2*j+1)*betas*sij + 4*betas*betas*d2;
        auto sij_idxx = sij.index({idxx});
        auto dij_idxx = dij.index({idxx});
        if (xyz == 0) {
          kinetics_dim0.select(2,xyz).masked_scatter_(idxx, dij_idxx);
          kinetics_dim1.select(2,xyz).masked_scatter_(idxx, sij_idxx);
          kinetics_dim2.select(2,xyz).masked_scatter_(idxx, sij_idxx);
        }
        else if (xyz == 1) {
          kinetics_dim0.select(2,xyz).masked_scatter_(idxx, sij_idxx);
          kinetics_dim1.select(2,xyz).masked_scatter_(idxx, dij_idxx);
          kinetics_dim2.select(2,xyz).masked_scatter_(idxx, sij_idxx);
        }
        else {
          kinetics_dim0.select(2,xyz).masked_scatter_(idxx, sij_idxx);
          kinetics_dim1.select(2,xyz).masked_scatter_(idxx, sij_idxx);
          kinetics_dim2.select(2,xyz).masked_scatter_(idxx, dij_idxx);
        }
      }
    }
  }
  auto kinetics = kinetics_dim0.prod(-1) + kinetics_dim1.prod(-1) + kinetics_dim2.prod(-1);
  auto res = -0.5 * torch::pow(M_PI / gamma, 1.5) * kinetics;
  return res;
}

torch::Tensor get_coulomb_mat(int max_ijkflat, int max_basis,
    py::list& idx_ijk,
    torch::Tensor& rcd_sq,
    torch::Tensor& ijk_pairs2_unique,
    // arguments for get_ecoeff (return: (nbasis_tot, nbasis_tot))
    torch::Tensor& alpha, torch::Tensor& betas, torch::Tensor& gamma,
    torch::Tensor& kappa, torch::Tensor& qab,
    py::dict& e_memory, py::str& key_format,
    // arguments for get_rcoeff (+gamma and rcd_sq)
    torch::Tensor& rcd,
    py::dict& r_memory, py::str& rkey_format) {

  // coulomb: (natoms, nbasis*nelmts, nbasis*nelmts)
  auto coulomb = torch::zeros_like(rcd_sq);
  auto numel = ijk_pairs2_unique.numel();
  for (int i = 0; i < numel; ++i) {
    auto ijk_flat_value = ijk_pairs2_unique.select(/*dim=*/0, /*index=*/i).item<int>();
    // auto idx = idx_ijk.select(/*dim=*/0, /*index=*/i);
    auto idx = py::cast<torch::Tensor>(idx_ijk[i]);
    auto slice0 = {idx};
    std::initializer_list<at::indexing::TensorIndex> slice1 = {Slice(None, None, None), idx};

    // unpack ijk_flat_value
    auto ijk_pair2 = ijk_flat_value % max_ijkflat;
    auto ijk_pair1 = (ijk_flat_value / max_ijkflat) % max_ijkflat;
    auto ijk_pair0 = (ijk_flat_value / max_ijkflat) / max_ijkflat;
    auto k = ijk_pair0 / max_basis;
    auto l = ijk_pair1 / max_basis;
    auto m = ijk_pair2 / max_basis;
    auto u = ijk_pair0 % max_basis;
    auto v = ijk_pair1 % max_basis;
    auto w = ijk_pair2 % max_basis;

    for (int r = 0; r < k + u + 1; ++r) {
      auto Erku = get_ecoeff(k, u, r, 0, alpha, betas, gamma, kappa, qab,
          e_memory, key_format).index(slice0);
      for (int s = 0; s < l + v + 1; ++s) {
        auto Eslv = get_ecoeff(l, v, s, 1, alpha, betas, gamma, kappa, qab,
            e_memory, key_format).index(slice0);
        for (int t = 0; t < m + w + 1; ++t) {
          auto Etmw = get_ecoeff(m, w, t, 2, alpha, betas, gamma, kappa, qab,
              e_memory, key_format).index(slice0);
          auto Rrst = get_rcoeff(r, s, t, 0, rcd, rcd_sq, gamma,
              r_memory, rkey_format).index(slice1);
          auto val = coulomb.index(slice1) + Erku * Eslv * Etmw * Rrst;
          coulomb.index_put_(slice1, val);
        }
      }
    }
  }
  coulomb *= -(2 * M_PI / gamma);
  return coulomb;
}

/************** coefficients **************/
torch::Tensor get_ecoeff(int i, int j, int t, int xyz,
    torch::Tensor& alpha, torch::Tensor& betas, torch::Tensor& gamma,
    torch::Tensor& kappa, torch::Tensor& qab,
    py::dict& e_memory, py::str& key_format) {

  /*
  alpha: (1, nbasis*nelmts)
  beta: (nbasis*nelmts, 1)
  gamma: (nbasis*nelmts, nbasis*nelmts)
  kappa: (nbasis*nelmts, nbasis*nelmts)
  qab: (nbasis*nelmts, nbasis*nelmts, 3)
  */

  if ((t < 0) || (t > i+j) || (i < 0) || (j < 0)) {
    return torch::zeros_like(qab.select(/*dim=*/2,/*index=*/0));
  }

  // access the coefficients
  auto key = key_format.format(i, j, t, xyz);
  if (e_memory.contains(key)) {
    return py::cast<torch::Tensor>(e_memory[key]);
  }

  // if no coefficients, then calculate it iteratively
  torch::Tensor c1, c2, c3, coeff;
  if ((i == 0) && (j > 0)) {
    c1 = get_ecoeff(i, j-1, t-1, xyz, alpha, betas, gamma, kappa, qab, e_memory, key_format);
    c2 = get_ecoeff(i, j-1, t  , xyz, alpha, betas, gamma, kappa, qab, e_memory, key_format);
    c3 = get_ecoeff(i, j-1, t+1, xyz, alpha, betas, gamma, kappa, qab, e_memory, key_format);
    coeff = 1./(2*gamma) * c1 + kappa * qab.select(/*dim=*/2,/*index=*/xyz) / betas * c2 + (t + 1) * c3;
  }
  else {
    c1 = get_ecoeff(i-1, j, t-1, xyz, alpha, betas, gamma, kappa, qab, e_memory, key_format);
    c2 = get_ecoeff(i-1, j, t  , xyz, alpha, betas, gamma, kappa, qab, e_memory, key_format);
    c3 = get_ecoeff(i-1, j, t+1, xyz, alpha, betas, gamma, kappa, qab, e_memory, key_format);
    coeff = 1./(2*gamma) * c1 - kappa * qab.select(/*dim=*/2,/*index=*/xyz) / alpha * c2 + (t + 1) * c3;
  }
  e_memory[key] = coeff;
  return coeff;
}

torch::Tensor get_rcoeff(int r, int s, int t, int n,
    torch::Tensor& rcd, torch::Tensor& rcd_sq, torch::Tensor& gamma,
    py::dict& r_memory, py::str& rkey_format) {
  /*
  rcd: natoms, nbasis*nelmts, nbasis*nelmts, 3
  rcd_sq: natoms, nbasis*nelmts, nbasis*nelmts
  gamma: nbasis*nelmts, nbasis*nelmts
  */

  auto all = Slice(None,None,None);
  if ((r < 0) || (s < 0) || (t < 0)) {
    return torch::zeros_like(rcd.select(/*dim=*/3, /*index=*/0));
  }

  // access the coefficients
  auto key = rkey_format.format(r, s, t, n);
  if (r_memory.contains(key)) {
    return py::cast<torch::Tensor>(r_memory[key]);
  }

  torch::Tensor c1, c2, coeff;
  if ((r == 0) && (s == 0) && (t == 0)) {
    // (natoms, nelmts_tot, nelmts_tot)
    auto gamma_rcd = gamma * rcd_sq;
    coeff = torch::pow(-2*gamma, n) * boys(n, gamma_rcd);
  }
  else if (r > 0) {
    c1 = get_rcoeff(r-2, s, t, n+1, rcd, rcd_sq, gamma, r_memory, rkey_format);
    c2 = get_rcoeff(r-1, s, t, n+1, rcd, rcd_sq, gamma, r_memory, rkey_format);
    coeff = (r-1) * c1 + rcd.select(/*dim=*/3, /*index=*/0) * c2;
  }
  else if (s > 0) {
    c1 = get_rcoeff(r, s-2, t, n+1, rcd, rcd_sq, gamma, r_memory, rkey_format);
    c2 = get_rcoeff(r, s-1, t, n+1, rcd, rcd_sq, gamma, r_memory, rkey_format);
    coeff = (s-1) * c1 + rcd.select(/*dim=*/3, /*index=*/1) * c2;
  }
  else {
    c1 = get_rcoeff(r, s, t-2, n+1, rcd, rcd_sq, gamma, r_memory, rkey_format);
    c2 = get_rcoeff(r, s, t-1, n+1, rcd, rcd_sq, gamma, r_memory, rkey_format);
    coeff = (t-1) * c1 + rcd.select(/*dim=*/3, /*index=*/2) * c2;
  }
  r_memory[key] = coeff;
  return coeff;
}

/************** helper functions **************/

torch::Tensor boys(int n, torch::Tensor& t) {
  auto options = torch::TensorOptions().dtype(t.dtype()).device(t.device());
  auto nhalf = torch::full(1, /*value=*/n + 0.5, options);
  auto t2 = t + 1e-12; // add small noise
  auto exp_part = -nhalf * torch::log(t2) + torch::lgamma(nhalf);
  return 0.5 * torch::igamma(nhalf, t2) * torch::exp(exp_part);
}

#endif
