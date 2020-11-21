#ifndef DDFT_BIND_CPP
#define DDFT_BIND_CPP

#include "integrals.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(csrc, m) {
  m.doc() = "C++ extension of DDFT";
  m.def("_overlap", &_overlap, "");
  m.def("_kinetic", &_kinetic, "");
  m.def("_nuclattr", &_nuclattr, "");
}

#endif
