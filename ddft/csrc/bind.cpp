#ifndef DDFT_BIND_CPP
#define DDFT_BIND_CPP

#include "coeffs.h"
#include "integrals.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(csrc, m) {
  m.doc() = "C++ extension of DDFT";
  m.def("get_ecoeff", &get_ecoeff, "Get the E-coefficients for gaussian-type orbitals");
  m.def("get_overlap_mat", &get_overlap_mat, "Get the overlap matrix for gaussian-type orbitals");
  m.def("get_kinetics_mat", &get_kinetics_mat, "Get the kinetics matrix for gaussian-type orbitals");
  m.def("get_coulomb_mat", &get_coulomb_mat, "Get the Coulomb matrix for gaussian-type orbitals");

  m.def("_overlap", &_overlap, "");
  m.def("_kinetic", &_kinetic, "");
  m.def("_nuclattr", &_nuclattr, "");
}

#endif
