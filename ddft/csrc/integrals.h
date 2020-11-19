#pragma once

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <torch/torch.h>
#include "igamma.h"

/*
The code below follows McMurchie-Davidson by Joshua Goings
https://github.com/jjgoings/McMurchie-Davidson (BSD-3-clause)
For the license, see NOTICE
*/

template <typename scalar_t>
scalar_t gaussian_product_center(scalar_t a1, scalar_t a2, scalar_t x1, scalar_t x2) {
  return (a1 * x1 + a2 * x2) / (a1 + a2);
}

template <typename scalar_t>
scalar_t boys(int n, scalar_t T) {
  scalar_t nhalf = n + 0.5;
  scalar_t exp_part = -nhalf * std::log(T) + std::lgamma(nhalf);
  return 0.5 * calc_igamma(nhalf, T) * std::exp(exp_part);
}

template <typename scalar_t>
scalar_t calc_ecoeff(int i, int j, int t,
                     scalar_t Qx, scalar_t a, scalar_t b,
                     int n = 0, scalar_t Ax = 0.0) {
  auto p = a + b;
  auto u = a * b / p;
  if (n == 0) {
    if ((t < 0) || (t > i + j)) {
      return (scalar_t)0.0;
    }
    else if ((i == 0) && (j == 0) && (t == 0)) {
      return std::exp(-u * Qx * Qx);
    }
    else if (j == 0) {
      return (0.5 / p)    * calc_ecoeff(i - 1, j, t - 1, Qx, a, b) -
             (u * Qx / a) * calc_ecoeff(i - 1, j, t    , Qx, a, b) +
             (t + 1)      * calc_ecoeff(i - 1, j, t + 1, Qx, a, b);
    }
    else {
      return (0.5 / p)    * calc_ecoeff(i, j - 1, t - 1, Qx, a, b) +
             (u * Qx / b) * calc_ecoeff(i, j - 1, t    , Qx, a, b) +
             (t + 1)      * calc_ecoeff(i, j - 1, t + 1, Qx, a, b);
    }
  }
  else {
    return calc_ecoeff(i + 1, j, t, Qx, a, b, n - 1, Ax) +
           Ax * calc_ecoeff(i, j, t, Qx, a, b, n - 1, Ax);
  }
}

template <typename scalar_t>
scalar_t calc_rcoeff(int t, int u, int v, int n,
                     scalar_t p, scalar_t PCx, scalar_t PCy, scalar_t PCz,
                     scalar_t RPC) {
  scalar_t val = 0.0;
  if ((t == 0) && (u == 0) && (v == 0)) {
    scalar_t T = p * RPC * RPC;
    val += std::pow(-2 * p, n) * boys(n, T);
  }
  else if ((t == 0) && (u == 0)) {
    if (v > 1) {
      val += (v - 1) * calc_rcoeff(t, u, v - 2, n + 1, p, PCx, PCy, PCz, RPC);
    }
    val += PCz * calc_rcoeff(t, u, v - 1, n + 1, p, PCx, PCy, PCz, RPC);
  }
  else if (t == 0) {
    if (u > 1) {
      val += (u - 1) * calc_rcoeff(t, u - 2, v, n + 1, p, PCx, PCy, PCz, RPC);
    }
    val += PCy * calc_rcoeff(t, u - 1, v, n + 1, p, PCx, PCy, PCz, RPC);
  }
  else {
    if (t > 1) {
      val += (t - 1) * calc_rcoeff(t - 2, u, v, n + 1, p, PCx, PCy, PCz, RPC);
    }
    val += PCx * calc_rcoeff(t - 1, u, v, n + 1, p, PCx, PCy, PCz, RPC);
  }
  return val;
}

// overlap integral
template <typename scalar_t>
scalar_t calc_overlap(scalar_t a1, scalar_t x1, scalar_t y1, scalar_t z1,
                      int l1, int m1, int n1,
                      scalar_t a2, scalar_t x2, scalar_t y2, scalar_t z2,
                      int l2, int m2, int n2) {
  auto sx = calc_ecoeff(l1, l2, 0, x1 - x2, a1, a2);
  auto sy = calc_ecoeff(m1, m2, 0, y1 - y2, a1, a2);
  auto sz = calc_ecoeff(n1, n2, 0, z1 - z2, a1, a2);
  return sx * sy * sz * std::pow(M_PI / (a1 + a2), 1.5);
}

// kinetics integral
template <typename scalar_t>
scalar_t calc_kinetic(scalar_t a1, scalar_t x1, scalar_t y1, scalar_t z1,
                      int l1, int m1, int n1,
                      scalar_t a2, scalar_t x2, scalar_t y2, scalar_t z2,
                      int l2, int m2, int n2) {
  // pass
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

  scalar_t Tx = Ax * calc_ecoeff(l1, l2    , 0, dx, a1, a2) +
                Bx * calc_ecoeff(l1, l2 + 2, 0, dx, a1, a2) +
                Cx * calc_ecoeff(l1, l2 - 2, 0, dx, a1, a2);
  Tx *= calc_ecoeff(m1, m2, 0, dy, a1, a2);
  Tx *= calc_ecoeff(n1, n2, 0, dz, a1, a2);

  scalar_t Ty = Ay * calc_ecoeff(m1, m2    , 0, dy, a1, a2) +
                Bx * calc_ecoeff(m1, m2 + 2, 0, dy, a1, a2) +
                Cy * calc_ecoeff(m1, m2 - 2, 0, dy, a1, a2);
  Ty *= calc_ecoeff(l1, l2, 0, dx, a1, a2);
  Ty *= calc_ecoeff(n1, n2, 0, dz, a1, a2);

  scalar_t Tz = Az * calc_ecoeff(n1, n2    , 0, dz, a1, a2) +
                Bx * calc_ecoeff(n1, n2 + 2, 0, dz, a1, a2) +
                Cz * calc_ecoeff(n1, n2 - 2, 0, dz, a1, a2);
  Tz *= calc_ecoeff(l1, l2, 0, dx, a1, a2);
  Tz *= calc_ecoeff(m1, m2, 0, dy, a1, a2);

  return (Tx + Ty + Tz) * std::pow(M_PI / (a1 + a2), 1.5);
}

// nuclear attraction with atom at (0, 0, 0)
template <typename scalar_t>
scalar_t calc_nuclattr(scalar_t a1, scalar_t x1, scalar_t y1, scalar_t z1,
                      int l1, int m1, int n1,
                      scalar_t a2, scalar_t x2, scalar_t y2, scalar_t z2,
                      int l2, int m2, int n2) {
  scalar_t p = a1 + a2;
  scalar_t Px = gaussian_product_center(a1, a2, x1, x2);
  scalar_t Py = gaussian_product_center(a1, a2, y1, y2);
  scalar_t Pz = gaussian_product_center(a1, a2, z1, z2);

  scalar_t RPC = std::sqrt(Px * Px + Py * Py + Pz * Pz);
  scalar_t val = 0.0;
  for (int t = 0; t < l1 + l2 + 1; ++t) {
    for (int u = 0; u < m1 + m2 + 1; ++u) {
      for (int v = 0; v < n1 + n2 + 1; ++v) {
        val += calc_ecoeff(l1, l2, t, x1 - x2, a1, a2) *
               calc_ecoeff(m1, m2, t, y1 - y2, a1, a2) *
               calc_ecoeff(n1, n2, t, z1 - z2, a1, a2) *
               calc_rcoeff(t, u, v, 0, p, Px, Py, Pz, RPC);
      }
    }
  }
  val *= (2 * M_PI) / p;
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
