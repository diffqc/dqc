#pragma once

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <torch/torch.h>
#include "igamma.h"

/*
The code below follows McMurchie-Davidson by Joshua Goings
https://github.com/jjgoings/McMurchie-Davidson (BSD-3-clause)
For the license, see NOTICE
*/

/*
DDFT modifications:
* nuclear attraction is normalized for atoms at (0, 0, 0)
* use of struct of params for ecoeff and rcoeff
* use of caches for ecoeff and rcoeff
*/

// comment the lines below if you don't want to store the cache
#define WITH_ECACHE
#define WITH_RCACHE

template <typename scalar_t>
struct ecoeff_params {
  scalar_t expUQx2;
  scalar_t uQxa;
  scalar_t uQxb;

  #ifdef WITH_ECACHE
  private:
  // caches
  std::unique_ptr<scalar_t[]> values_;
  std::unique_ptr<bool[]> valid_;
  int max_tn_;

  public:
  #endif

  ecoeff_params(scalar_t Qx, scalar_t u, scalar_t a1, scalar_t a2,
                int max_i, int max_j, bool fullt = false) {
    expUQx2 = std::exp(-u * Qx * Qx);
    uQxa = u * Qx / a1;
    uQxb = u * Qx / a2;

    #ifdef WITH_ECACHE
    int max_ij = (max_i + max_j);
    max_tn_ = 1 + (fullt ? max_ij : (max_ij / 2));
    int size = (1 + max_ij) * max_tn_;

    values_.reset(new scalar_t[size]);
    valid_.reset(new bool[size]()); // initialize with all false
    #endif
  }

  #ifdef WITH_ECACHE
  inline int hasval(int i, int j, int t) {
    return valid_[(i + j) * max_tn_ + t];
  }

  inline scalar_t getval(int i, int j, int t) {
    return values_[(i + j) * max_tn_ + t];
  }

  inline void setval(int i, int j, int t, scalar_t val) {
    int idx = (i + j) * max_tn_ + t;
    values_[idx] = val;
    valid_[idx] = true;
  }
  #endif
};

template <typename scalar_t>
struct rcoeff_params {
  scalar_t T;
  scalar_t Px;
  scalar_t Py;
  scalar_t Pz;
  scalar_t mp2;
  int tmax = 0;
  int umax = 0;
  int vmax = 0;
  int nmax = 0;

  #ifdef WITH_RCACHE
  // caches
  std::unique_ptr<scalar_t[]> values_;
  std::unique_ptr<bool[]> valid_;
  int v_offset_;
  int u_offset_;
  int t_offset_;
  #endif

  rcoeff_params(scalar_t p_, scalar_t Px_, scalar_t Py_, scalar_t Pz_,
                int max_tn, int max_un, int max_vn) {
    Px = Px_;
    Py = Py_;
    Pz = Pz_;
    T = p_ * (Px * Px + Py * Py + Pz * Pz);
    mp2 = -2 * p_;

    #ifdef WITH_RCACHE
    int max_nn = max_tn + max_un + max_vn;
    v_offset_ = (max_nn + 1);
    u_offset_ = (max_vn + 1) * v_offset_;
    t_offset_ = (max_un + 1) * u_offset_;
    int size  = (max_tn + 1) * t_offset_;
    values_.reset(new scalar_t[size]);
    valid_.reset(new bool[size]()); // initialize with false
    #endif
  }

  #ifdef WITH_RCACHE
  inline int hasval(int t, int u, int v, int n) {
    int idx = t * t_offset_ + u * u_offset_ + v * v_offset_ + n;
    return valid_[idx];
  }

  inline scalar_t getval(int t, int u, int v, int n) {
    int idx = t * t_offset_ + u * u_offset_ + v * v_offset_ + n;
    return values_[idx];
  }

  inline void setval(int t, int u, int v, int n, scalar_t val) {
    int idx = t * t_offset_ + u * u_offset_ + v * v_offset_ + n;
    values_[idx] = val;
    valid_[idx] = true;
  }
  #endif
};

template <typename scalar_t>
scalar_t gaussian_product_center(scalar_t a1, scalar_t a2, scalar_t x1, scalar_t x2) {
  return (a1 * x1 + a2 * x2) / (a1 + a2);
}

template <typename scalar_t>
scalar_t boys(int n, scalar_t T) {
  scalar_t nhalf = n + 0.5;
  if (T == 0) {
    return (scalar_t)1.0 / (2 * n + (scalar_t)1.0);
  }
  else {
    scalar_t exp_part = -nhalf * std::log(T) + std::lgamma(nhalf);
    return 0.5 * calc_igamma(nhalf, T) * std::exp(exp_part);
  }
}

template <typename scalar_t>
scalar_t calc_ecoeff(int i, int j, int t,
                     scalar_t half_over_p, ecoeff_params<scalar_t>& ecx,
                     int n = 0, scalar_t Ax = 0.0) {
  if ((i < 0) || (j < 0) || (t < 0)) {
    return (scalar_t) 0.0; // undefined
  }
  if (n == 0) {
    if ((t < 0) || (t > i + j)) {
      return (scalar_t) 0.0;
    }
    else if ((i == 0) && (j == 0) && (t == 0)) {
      return ecx.expUQx2;
    }
    else {
      #ifdef WITH_ECACHE
      // cache
      if (ecx.hasval(i, j, t)) {
        return ecx.getval(i, j, t);
      }
      #endif

      scalar_t res;
      if (j == 0) {
        res = half_over_p * calc_ecoeff(i - 1, j, t - 1, half_over_p, ecx) -
              ecx.uQxa    * calc_ecoeff(i - 1, j, t    , half_over_p, ecx) +
              (t + 1)     * calc_ecoeff(i - 1, j, t + 1, half_over_p, ecx);
      }
      else {
        res = half_over_p * calc_ecoeff(i, j - 1, t - 1, half_over_p, ecx) +
              ecx.uQxb    * calc_ecoeff(i, j - 1, t    , half_over_p, ecx) +
              (t + 1)     * calc_ecoeff(i, j - 1, t + 1, half_over_p, ecx);
      }

      #ifdef WITH_ECACHE
      // store cache
      ecx.setval(i, j, t, res);
      #endif

      return res;
    }
  }
  else {
    return calc_ecoeff(i + 1, j, t, half_over_p, ecx, n - 1, Ax) +
           Ax * calc_ecoeff(i, j, t, half_over_p, ecx, n - 1, Ax);
  }
}

template <typename scalar_t>
scalar_t calc_rcoeff(int t, int u, int v, int n,
                     rcoeff_params<scalar_t>& rp) {
  scalar_t val = 0.0;
  if ((t < 0) || (u < 0) || (v < 0) || (n < 0)) {
    return 0.0; // undefined
  }

  #ifdef WITH_RCACHE
  if (rp.hasval(t, u, v, n)) {
    return rp.getval(t, u, v, n);
  }
  #endif

  if ((t == 0) && (u == 0) && (v == 0)) {
    val += std::pow(rp.mp2, n) * boys(n, rp.T);
  }
  else if ((t == 0) && (u == 0)) {
    if (v > 1) {
      val += (v - 1) * calc_rcoeff(t, u, v - 2, n + 1, rp);
    }
    val += rp.Pz * calc_rcoeff(t, u, v - 1, n + 1, rp);
  }
  else if (t == 0) {
    if (u > 1) {
      val += (u - 1) * calc_rcoeff(t, u - 2, v, n + 1, rp);
    }
    val += rp.Py * calc_rcoeff(t, u - 1, v, n + 1, rp);
  }
  else {
    if (t > 1) {
      val += (t - 1) * calc_rcoeff(t - 2, u, v, n + 1, rp);
    }
    val += rp.Px * calc_rcoeff(t - 1, u, v, n + 1, rp);
  }

  #ifdef WITH_RCACHE
  rp.setval(t, u, v, n, val);
  #endif

  return val;
}

// overlap integral
template <typename scalar_t>
scalar_t calc_overlap(scalar_t a1, scalar_t x1, scalar_t y1, scalar_t z1,
                      int l1, int m1, int n1,
                      scalar_t a2, scalar_t x2, scalar_t y2, scalar_t z2,
                      int l2, int m2, int n2) {
  // bound check
  if ((l1 < 0) || (m1 < 0) || (n1 < 0) ||
      (l2 < 0) || (m2 < 0) || (n2 < 0)) {
    return (scalar_t)0.0;
  }

  scalar_t p = a1 + a2;
  scalar_t u = a1 * a2 / p;
  scalar_t half_over_p = 0.5 / p;
  auto ecx = ecoeff_params<scalar_t>(x1 - x2, u, a1, a2, l1, l2);
  auto ecy = ecoeff_params<scalar_t>(y1 - y2, u, a1, a2, m1, m2);
  auto ecz = ecoeff_params<scalar_t>(z1 - z2, u, a1, a2, n1, n2);
  auto sx = calc_ecoeff(l1, l2, 0, half_over_p, ecx);
  auto sy = calc_ecoeff(m1, m2, 0, half_over_p, ecy);
  auto sz = calc_ecoeff(n1, n2, 0, half_over_p, ecz);
  return sx * sy * sz * std::pow(M_PI / (a1 + a2), 1.5);
}

// kinetics integral
template <typename scalar_t>
scalar_t calc_kinetic(scalar_t a1, scalar_t x1, scalar_t y1, scalar_t z1,
                      int l1, int m1, int n1,
                      scalar_t a2, scalar_t x2, scalar_t y2, scalar_t z2,
                      int l2, int m2, int n2) {
  // bound check
  if ((l1 < 0) || (m1 < 0) || (n1 < 0) ||
      (l2 < 0) || (m2 < 0) || (n2 < 0)) {
    return (scalar_t)0.0;
  }

  scalar_t Ax = (2 * l2 + 1) * a2;
  scalar_t Ay = (2 * m2 + 1) * a2;
  scalar_t Az = (2 * n2 + 1) * a2;
  scalar_t Bx = -2 * a2 * a2;
  scalar_t Cx = -0.5 * l2 * (l2 - 1);
  scalar_t Cy = -0.5 * m2 * (m2 - 1);
  scalar_t Cz = -0.5 * n2 * (n2 - 1);
  scalar_t dx = x1 - x2;
  scalar_t dy = y1 - y2;
  scalar_t dz = z1 - z2;

  // prep for ecoeff
  scalar_t p = a1 + a2;
  scalar_t u = a1 * a2 / p;
  scalar_t half_over_p = 0.5 / p;
  auto ecx = ecoeff_params<scalar_t>(dx, u, a1, a2, l1, l2 + 2);
  auto ecy = ecoeff_params<scalar_t>(dy, u, a1, a2, m1, m2 + 2);
  auto ecz = ecoeff_params<scalar_t>(dz, u, a1, a2, n1, n2 + 2);

  scalar_t Tx = Ax * calc_ecoeff(l1, l2    , 0, half_over_p, ecx) +
                Bx * calc_ecoeff(l1, l2 + 2, 0, half_over_p, ecx) +
                Cx * calc_ecoeff(l1, l2 - 2, 0, half_over_p, ecx);
  Tx *= calc_ecoeff(m1, m2, 0, half_over_p, ecy);
  Tx *= calc_ecoeff(n1, n2, 0, half_over_p, ecz);

  scalar_t Ty = Ay * calc_ecoeff(m1, m2    , 0, half_over_p, ecy) +
                Bx * calc_ecoeff(m1, m2 + 2, 0, half_over_p, ecy) +
                Cy * calc_ecoeff(m1, m2 - 2, 0, half_over_p, ecy);
  Ty *= calc_ecoeff(l1, l2, 0, half_over_p, ecx);
  Ty *= calc_ecoeff(n1, n2, 0, half_over_p, ecz);

  scalar_t Tz = Az * calc_ecoeff(n1, n2    , 0, half_over_p, ecz) +
                Bx * calc_ecoeff(n1, n2 + 2, 0, half_over_p, ecz) +
                Cz * calc_ecoeff(n1, n2 - 2, 0, half_over_p, ecz);
  Tz *= calc_ecoeff(l1, l2, 0, half_over_p, ecx);
  Tz *= calc_ecoeff(m1, m2, 0, half_over_p, ecy);

  return (Tx + Ty + Tz) * std::pow(M_PI / (a1 + a2), 1.5);
}

// nuclear attraction with atom at (0, 0, 0)
template <typename scalar_t>
scalar_t calc_nuclattr(scalar_t a1, scalar_t x1, scalar_t y1, scalar_t z1,
                      int l1, int m1, int n1,
                      scalar_t a2, scalar_t x2, scalar_t y2, scalar_t z2,
                      int l2, int m2, int n2) {
  // bound check
  if ((l1 < 0) || (m1 < 0) || (n1 < 0) ||
      (l2 < 0) || (m2 < 0) || (n2 < 0)) {
    return (scalar_t)0.0;
  }

  scalar_t Px = gaussian_product_center(a1, a2, x1, x2);
  scalar_t Py = gaussian_product_center(a1, a2, y1, y2);
  scalar_t Pz = gaussian_product_center(a1, a2, z1, z2);

  // prep for ecoeff
  scalar_t p = a1 + a2;
  scalar_t uu = a1 * a2 / p;
  scalar_t half_over_p = 0.5 / p;
  auto ecx = ecoeff_params<scalar_t>(x1 - x2, uu, a1, a2,
                                     l1, l2, /*fullt*/true);
  auto ecy = ecoeff_params<scalar_t>(y1 - y2, uu, a1, a2,
                                     m1, m2, /*fullt*/true);
  auto ecz = ecoeff_params<scalar_t>(z1 - z2, uu, a1, a2,
                                     n1, n2, /*fullt*/true);

  // params for rcoeff
  auto rp = rcoeff_params<scalar_t>(p, Px, Py, Pz,
                                    l1 + l2, m1 + m2, n1 + n2);

  scalar_t val = 0.0;
  for (int t = 0; t < l1 + l2 + 1; ++t) {
    scalar_t el = calc_ecoeff(l1, l2, t, half_over_p, ecx);

    for (int u = 0; u < m1 + m2 + 1; ++u) {
      scalar_t em = calc_ecoeff(m1, m2, u, half_over_p, ecy);

      for (int v = 0; v < n1 + n2 + 1; ++v) {
        scalar_t ev = calc_ecoeff(n1, n2, v, half_over_p, ecz);
        val += (el * em * ev) * calc_rcoeff(t, u, v, 0, rp);
      }
    }
  }
  val *= -(2 * M_PI) / p;
  return val;
}

// electron repulsion
template <typename scalar_t>
scalar_t calc_elrep(scalar_t a1, scalar_t x1, scalar_t y1, scalar_t z1,
                    int l1, int m1, int n1,
                    scalar_t a2, scalar_t x2, scalar_t y2, scalar_t z2,
                    int l2, int m2, int n2,
                    scalar_t a3, scalar_t x3, scalar_t y3, scalar_t z3,
                    int l3, int m3, int n3,
                    scalar_t a4, scalar_t x4, scalar_t y4, scalar_t z4,
                    int l4, int m4, int n4) {
  // bound check
  if ((l1 < 0) || (m1 < 0) || (n1 < 0) ||
      (l2 < 0) || (m2 < 0) || (n2 < 0) ||
      (l3 < 0) || (m3 < 0) || (n3 < 0) ||
      (l4 < 0) || (m4 < 0) || (n4 < 0)) {
    return (scalar_t)0.0;
  }
  scalar_t p = a1 + a2;
  scalar_t q = a3 + a4;
  scalar_t alpha = p * q / (p + q);
  scalar_t Px = (a1 * x1 + a2 * x2) / p;
  scalar_t Py = (a1 * y1 + a2 * y2) / p;
  scalar_t Pz = (a1 * z1 + a2 * z2) / p;
  scalar_t Qx = (a3 * x3 + a4 * x4) / q;
  scalar_t Qy = (a3 * y3 + a4 * y4) / q;
  scalar_t Qz = (a3 * z3 + a4 * z4) / q;
  scalar_t RPQ = std::sqrt(std::pow(Px - Qx, 2) +
                           std::pow(Py - Qy, 2) +
                           std::pow(Pz - Qz, 2));

  scalar_t u1 = a1 * a2 / p;
  scalar_t u3 = a3 * a4 / q;
  scalar_t half_over_p = 0.5 / p;
  scalar_t half_over_q = 0.5 / q;
  auto ecx = ecoeff_params<scalar_t>(x1 - x2, u1, a1, a2,
                                     l1, l2, /*fullt*/true);
  auto ecy = ecoeff_params<scalar_t>(y1 - y2, u1, a1, a2,
                                     m1, m2, /*fullt*/true);
  auto ecz = ecoeff_params<scalar_t>(z1 - z2, u1, a1, a2,
                                     n1, n2, /*fullt*/true);
  auto edx = ecoeff_params<scalar_t>(x3 - x4, u3, a3, a4,
                                     l3, l4, /*fullt*/true);
  auto edy = ecoeff_params<scalar_t>(y3 - y4, u3, a3, a4,
                                     m3, m4, /*fullt*/true);
  auto edz = ecoeff_params<scalar_t>(z3 - z4, u3, a3, a4,
                                     n3, n4, /*fullt*/true);
  auto rp = rcoeff_params<scalar_t>(alpha,
                                    Px - Qx, Py - Qy, Pz - Qz,
                                    l1 + l2 + l3 + l4,
                                    m1 + m2 + m3 + m4,
                                    n1 + n2 + n3 + n4);
  int t, u, v, tau, nu, phi;
  scalar_t val = 0.0;
  for (t = 0; t < l1 + l2 + 1; ++t) {
    scalar_t et = calc_ecoeff(l1, l2, t, half_over_p, ecx);

    for (u = 0; u < m1 + m2 + 1; ++u) {
      scalar_t em = calc_ecoeff(m1, m2, u, half_over_p, ecy);

      for (v = 0; v < n1 + n2 + 1; ++v) {
        scalar_t en = calc_ecoeff(n1, n2, v, half_over_p, ecz);

        for (tau = 0; tau < l3 + l4 + 1; ++tau) {
          scalar_t etau = calc_ecoeff(l3, l4, tau, half_over_q, edx);

          for (nu = 0; nu < m3 + m4 + 1; ++nu) {
            scalar_t enu = calc_ecoeff(m3, m4, nu , half_over_q, edy);

            for (phi = 0; phi < n3 + n4 + 1; ++phi) {
              scalar_t ephi = calc_ecoeff(n3, n4, phi, half_over_q, edz);
              scalar_t rcoeff = calc_rcoeff(t + tau, u + nu, v + phi, 0, rp);
              scalar_t sign = ((tau + nu + phi) % 2 == 0) ? 1.0 : -1.0;
              val += (et * em * en * etau * enu * ephi) * sign * rcoeff;
            }
          }
        }
      }
    }
  }
  val *= 2 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q));
  return val;
}
/************************* kernels *************************/

template <typename scalar_t>
static void overlap_kernel(
          torch::Tensor& ret,
    const torch::Tensor& a1,
    const torch::Tensor& x1,
    const torch::Tensor& y1,
    const torch::Tensor& z1,
    const torch::Tensor& l1,
    const torch::Tensor& m1,
    const torch::Tensor& n1,
    const torch::Tensor& a2,
    const torch::Tensor& x2,
    const torch::Tensor& y2,
    const torch::Tensor& z2,
    const torch::Tensor& l2,
    const torch::Tensor& m2,
    const torch::Tensor& n2) {

  auto ret_data = ret.data_ptr<scalar_t>();
  auto a1_data = a1.data_ptr<scalar_t>();
  auto x1_data = x1.data_ptr<scalar_t>();
  auto y1_data = y1.data_ptr<scalar_t>();
  auto z1_data = z1.data_ptr<scalar_t>();
  auto l1_data = l1.data_ptr<int>();
  auto m1_data = m1.data_ptr<int>();
  auto n1_data = n1.data_ptr<int>();
  auto a2_data = a2.data_ptr<scalar_t>();
  auto x2_data = x2.data_ptr<scalar_t>();
  auto y2_data = y2.data_ptr<scalar_t>();
  auto z2_data = z2.data_ptr<scalar_t>();
  auto l2_data = l2.data_ptr<int>();
  auto m2_data = m2.data_ptr<int>();
  auto n2_data = n2.data_ptr<int>();
  int numel = a1.numel();

  for (int i = 0; i < numel; ++i) {
    ret_data[i] = calc_overlap(
      a1_data[i],
      x1_data[i],
      y1_data[i],
      z1_data[i],
      l1_data[i],
      m1_data[i],
      n1_data[i],
      a2_data[i],
      x2_data[i],
      y2_data[i],
      z2_data[i],
      l2_data[i],
      m2_data[i],
      n2_data[i]
    );
  }
}

template <typename scalar_t>
static void kinetic_kernel(
          torch::Tensor& ret,
    const torch::Tensor& a1,
    const torch::Tensor& x1,
    const torch::Tensor& y1,
    const torch::Tensor& z1,
    const torch::Tensor& l1,
    const torch::Tensor& m1,
    const torch::Tensor& n1,
    const torch::Tensor& a2,
    const torch::Tensor& x2,
    const torch::Tensor& y2,
    const torch::Tensor& z2,
    const torch::Tensor& l2,
    const torch::Tensor& m2,
    const torch::Tensor& n2) {

  auto ret_data = ret.data_ptr<scalar_t>();
  auto a1_data = a1.data_ptr<scalar_t>();
  auto x1_data = x1.data_ptr<scalar_t>();
  auto y1_data = y1.data_ptr<scalar_t>();
  auto z1_data = z1.data_ptr<scalar_t>();
  auto l1_data = l1.data_ptr<int>();
  auto m1_data = m1.data_ptr<int>();
  auto n1_data = n1.data_ptr<int>();
  auto a2_data = a2.data_ptr<scalar_t>();
  auto x2_data = x2.data_ptr<scalar_t>();
  auto y2_data = y2.data_ptr<scalar_t>();
  auto z2_data = z2.data_ptr<scalar_t>();
  auto l2_data = l2.data_ptr<int>();
  auto m2_data = m2.data_ptr<int>();
  auto n2_data = n2.data_ptr<int>();
  int numel = a1.numel();

  for (int i = 0; i < numel; ++i) {
    ret_data[i] = calc_kinetic(
      a1_data[i],
      x1_data[i],
      y1_data[i],
      z1_data[i],
      l1_data[i],
      m1_data[i],
      n1_data[i],
      a2_data[i],
      x2_data[i],
      y2_data[i],
      z2_data[i],
      l2_data[i],
      m2_data[i],
      n2_data[i]
    );
  }
}

template <typename scalar_t>
static void nuclattr_kernel(
          torch::Tensor& ret,
    const torch::Tensor& a1,
    const torch::Tensor& x1,
    const torch::Tensor& y1,
    const torch::Tensor& z1,
    const torch::Tensor& l1,
    const torch::Tensor& m1,
    const torch::Tensor& n1,
    const torch::Tensor& a2,
    const torch::Tensor& x2,
    const torch::Tensor& y2,
    const torch::Tensor& z2,
    const torch::Tensor& l2,
    const torch::Tensor& m2,
    const torch::Tensor& n2) {

  auto ret_data = ret.data_ptr<scalar_t>();
  auto a1_data = a1.data_ptr<scalar_t>();
  auto x1_data = x1.data_ptr<scalar_t>();
  auto y1_data = y1.data_ptr<scalar_t>();
  auto z1_data = z1.data_ptr<scalar_t>();
  auto l1_data = l1.data_ptr<int>();
  auto m1_data = m1.data_ptr<int>();
  auto n1_data = n1.data_ptr<int>();
  auto a2_data = a2.data_ptr<scalar_t>();
  auto x2_data = x2.data_ptr<scalar_t>();
  auto y2_data = y2.data_ptr<scalar_t>();
  auto z2_data = z2.data_ptr<scalar_t>();
  auto l2_data = l2.data_ptr<int>();
  auto m2_data = m2.data_ptr<int>();
  auto n2_data = n2.data_ptr<int>();
  int numel = a1.numel();

  for (int i = 0; i < numel; ++i) {
    ret_data[i] = calc_nuclattr(
      a1_data[i],
      x1_data[i],
      y1_data[i],
      z1_data[i],
      l1_data[i],
      m1_data[i],
      n1_data[i],
      a2_data[i],
      x2_data[i],
      y2_data[i],
      z2_data[i],
      l2_data[i],
      m2_data[i],
      n2_data[i]
    );
  }
}

template <typename scalar_t>
static void elrep_kernel(
          torch::Tensor& ret,
    const torch::Tensor& a1,
    const torch::Tensor& x1,
    const torch::Tensor& y1,
    const torch::Tensor& z1,
    const torch::Tensor& l1,
    const torch::Tensor& m1,
    const torch::Tensor& n1,
    const torch::Tensor& a2,
    const torch::Tensor& x2,
    const torch::Tensor& y2,
    const torch::Tensor& z2,
    const torch::Tensor& l2,
    const torch::Tensor& m2,
    const torch::Tensor& n2,
    const torch::Tensor& a3,
    const torch::Tensor& x3,
    const torch::Tensor& y3,
    const torch::Tensor& z3,
    const torch::Tensor& l3,
    const torch::Tensor& m3,
    const torch::Tensor& n3,
    const torch::Tensor& a4,
    const torch::Tensor& x4,
    const torch::Tensor& y4,
    const torch::Tensor& z4,
    const torch::Tensor& l4,
    const torch::Tensor& m4,
    const torch::Tensor& n4) {

  auto ret_data = ret.data_ptr<scalar_t>();
  auto a1_data = a1.data_ptr<scalar_t>();
  auto x1_data = x1.data_ptr<scalar_t>();
  auto y1_data = y1.data_ptr<scalar_t>();
  auto z1_data = z1.data_ptr<scalar_t>();
  auto l1_data = l1.data_ptr<int>();
  auto m1_data = m1.data_ptr<int>();
  auto n1_data = n1.data_ptr<int>();
  auto a2_data = a2.data_ptr<scalar_t>();
  auto x2_data = x2.data_ptr<scalar_t>();
  auto y2_data = y2.data_ptr<scalar_t>();
  auto z2_data = z2.data_ptr<scalar_t>();
  auto l2_data = l2.data_ptr<int>();
  auto m2_data = m2.data_ptr<int>();
  auto n2_data = n2.data_ptr<int>();
  auto a3_data = a3.data_ptr<scalar_t>();
  auto x3_data = x3.data_ptr<scalar_t>();
  auto y3_data = y3.data_ptr<scalar_t>();
  auto z3_data = z3.data_ptr<scalar_t>();
  auto l3_data = l3.data_ptr<int>();
  auto m3_data = m3.data_ptr<int>();
  auto n3_data = n3.data_ptr<int>();
  auto a4_data = a4.data_ptr<scalar_t>();
  auto x4_data = x4.data_ptr<scalar_t>();
  auto y4_data = y4.data_ptr<scalar_t>();
  auto z4_data = z4.data_ptr<scalar_t>();
  auto l4_data = l4.data_ptr<int>();
  auto m4_data = m4.data_ptr<int>();
  auto n4_data = n4.data_ptr<int>();
  int numel = a1.numel();

  for (int i = 0; i < numel; ++i) {
    ret_data[i] = calc_elrep(
      a1_data[i],
      x1_data[i],
      y1_data[i],
      z1_data[i],
      l1_data[i],
      m1_data[i],
      n1_data[i],
      a2_data[i],
      x2_data[i],
      y2_data[i],
      z2_data[i],
      l2_data[i],
      m2_data[i],
      n2_data[i],
      a3_data[i],
      x3_data[i],
      y3_data[i],
      z3_data[i],
      l3_data[i],
      m3_data[i],
      n3_data[i],
      a4_data[i],
      x4_data[i],
      y4_data[i],
      z4_data[i],
      l4_data[i],
      m4_data[i],
      n4_data[i]
    );
  }
}

/************************* dispatchers *************************/

torch::Tensor _overlap(
    const torch::Tensor& a1,
    const torch::Tensor& x1,
    const torch::Tensor& y1,
    const torch::Tensor& z1,
    const torch::Tensor& l1,
    const torch::Tensor& m1,
    const torch::Tensor& n1,
    const torch::Tensor& a2,
    const torch::Tensor& x2,
    const torch::Tensor& y2,
    const torch::Tensor& z2,
    const torch::Tensor& l2,
    const torch::Tensor& m2,
    const torch::Tensor& n2) {

  torch::Tensor result = torch::empty_like(a1);
  at::ScalarType the_type = a1.scalar_type();
  switch (the_type) {
    case at::ScalarType::Double: {
      overlap_kernel<double>(result, a1, x1, y1, z1, l1, m1, n1,
                                     a2, x2, y2, z2, l2, m2, n2);
      break;
    }
    case at::ScalarType::Float: {
      overlap_kernel<float>(result, a1, x1, y1, z1, l1, m1, n1,
                                    a2, x2, y2, z2, l2, m2, n2);
      break;
    }
    default: {
      std::cout << "The function overlap is undefined for type " << the_type << "\n";
      std::exit(1);
    }
  }
  return result;
}

torch::Tensor _kinetic(
    const torch::Tensor& a1,
    const torch::Tensor& x1,
    const torch::Tensor& y1,
    const torch::Tensor& z1,
    const torch::Tensor& l1,
    const torch::Tensor& m1,
    const torch::Tensor& n1,
    const torch::Tensor& a2,
    const torch::Tensor& x2,
    const torch::Tensor& y2,
    const torch::Tensor& z2,
    const torch::Tensor& l2,
    const torch::Tensor& m2,
    const torch::Tensor& n2) {
  //
  torch::Tensor result = torch::empty_like(a1);
  at::ScalarType the_type = a1.scalar_type();
  switch (the_type) {
    case at::ScalarType::Double: {
      kinetic_kernel<double>(result, a1, x1, y1, z1, l1, m1, n1,
                                     a2, x2, y2, z2, l2, m2, n2);
      break;
    }
    case at::ScalarType::Float: {
      kinetic_kernel<float>(result, a1, x1, y1, z1, l1, m1, n1,
                                    a2, x2, y2, z2, l2, m2, n2);
      break;
    }
    default: {
      std::cout << "The function kinetic is undefined for type " << the_type;
      std::exit(1);
    }
  }
  return result;
}

torch::Tensor _nuclattr(
    const torch::Tensor& a1,
    const torch::Tensor& x1,
    const torch::Tensor& y1,
    const torch::Tensor& z1,
    const torch::Tensor& l1,
    const torch::Tensor& m1,
    const torch::Tensor& n1,
    const torch::Tensor& a2,
    const torch::Tensor& x2,
    const torch::Tensor& y2,
    const torch::Tensor& z2,
    const torch::Tensor& l2,
    const torch::Tensor& m2,
    const torch::Tensor& n2) {
  //
  torch::Tensor result = torch::empty_like(a1);
  at::ScalarType the_type = a1.scalar_type();
  switch (the_type) {
    case at::ScalarType::Double: {
      nuclattr_kernel<double>(result, a1, x1, y1, z1, l1, m1, n1,
                                      a2, x2, y2, z2, l2, m2, n2);
      break;
    }
    case at::ScalarType::Float: {
      nuclattr_kernel<float>(result, a1, x1, y1, z1, l1, m1, n1,
                                     a2, x2, y2, z2, l2, m2, n2);
      break;
    }
    default: {
      std::cout << "The function nuclattr is undefined for type " << the_type;
      std::exit(1);
    }
  }
  return result;
}

torch::Tensor _elrep(
    const torch::Tensor& a1,
    const torch::Tensor& x1,
    const torch::Tensor& y1,
    const torch::Tensor& z1,
    const torch::Tensor& l1,
    const torch::Tensor& m1,
    const torch::Tensor& n1,
    const torch::Tensor& a2,
    const torch::Tensor& x2,
    const torch::Tensor& y2,
    const torch::Tensor& z2,
    const torch::Tensor& l2,
    const torch::Tensor& m2,
    const torch::Tensor& n2,
    const torch::Tensor& a3,
    const torch::Tensor& x3,
    const torch::Tensor& y3,
    const torch::Tensor& z3,
    const torch::Tensor& l3,
    const torch::Tensor& m3,
    const torch::Tensor& n3,
    const torch::Tensor& a4,
    const torch::Tensor& x4,
    const torch::Tensor& y4,
    const torch::Tensor& z4,
    const torch::Tensor& l4,
    const torch::Tensor& m4,
    const torch::Tensor& n4) {
  //
  torch::Tensor result = torch::empty_like(a1);
  at::ScalarType the_type = a1.scalar_type();
  switch (the_type) {
    case at::ScalarType::Double: {
      elrep_kernel<double>(result, a1, x1, y1, z1, l1, m1, n1,
                                   a2, x2, y2, z2, l2, m2, n2,
                                   a3, x3, y3, z3, l3, m3, n3,
                                   a4, x4, y4, z4, l4, m4, n4);
      break;
    }
    case at::ScalarType::Float: {
      elrep_kernel<float>(result, a1, x1, y1, z1, l1, m1, n1,
                                  a2, x2, y2, z2, l2, m2, n2,
                                  a3, x3, y3, z3, l3, m3, n3,
                                  a4, x4, y4, z4, l4, m4, n4);
      break;
    }
    default: {
      std::cout << "The function elrep is undefined for type " << the_type;
      std::exit(1);
    }
  }
  return result;
}
