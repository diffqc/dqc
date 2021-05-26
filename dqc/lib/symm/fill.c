#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

void fills4(double* out, double* arr, int64_t ni, int64_t nk);
// void fills8(double* out, double* arr, int64_t n);

void fills4(double* out, double* arr, int64_t ni, int64_t nk) {
  // fill out the full matrix from the reduced symmetric array (arr) with s4
  // symmetry: (ijkl) == (ijlk) == (jilk) == (jikl)
  // out should be initialized with all zeros
  int64_t n2 = nk * nk;
  int64_t n3 = ni * n2;
  int64_t iarr = 0;
  for (int64_t i = 0; i < ni; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      int64_t ij = i * n3 + j * n2;
      int64_t ji = j * n3 + i * n2;
      for (int64_t k = 0; k < nk; ++k) {
        for (int64_t l = 0; l <= k; ++l) {
          double elmt = arr[iarr++];
          if (elmt == 0.0) {
            continue;
          }
          int64_t kl = k * nk + l;
          int64_t lk = l * nk + k;
          out[ij + kl] = elmt;
          out[ij + lk] = elmt;
          out[ji + lk] = elmt;
          out[ji + kl] = elmt;
        }
      }
    }
  }
}

// void fills8(double* out, double* arr, int64_t n) {
//   // fill out the full matrix from the reduced symmetric array (arr) with s8
//   // symmetry: (ijkl) == (ijlk) == (jilk) == (jikl) == (klij) == (lkij) == (lkji)
//   // == (klji)
//   // out should be initialized with all zeros
//
//   int64_t n2 = n * n;
//   int64_t n3 = n2 * n;
//   int64_t iarr = 0;
//   // int64_t size1 = n * (n + 1) / 2;
//   // int64_t size = size1 * (size1 + 1) / 2;
//   for (int64_t i = 0; i < n; ++i) {
//     int64_t i_idx = i * (i + 1) / 2;
//
//     for (int64_t j = 0; j <= i; ++j) {
//       int64_t ij_idx = i_idx + j;
//
//       for (int64_t k = 0; k <= i; ++k) {
//         int64_t lmax1 = ij_idx - k * (k + 1) / 2;
//         int64_t lmax = lmax1 < k ? lmax1 : k;
//
//         for (int64_t l = 0; l <= lmax; ++l) {
//           double elmt = arr[iarr];
//           iarr++;
//           if (elmt == 0.0) {
//             continue;
//           }
//
//           int64_t ij0 = i * n3 + j * n2;
//           int64_t ji0 = j * n3 + i * n2;
//           int64_t kl0 = k * n + l;
//           int64_t lk0 = l * n + k;
//           out[ij0 + kl0] = elmt;
//           out[ij0 + lk0] = elmt;
//           out[ji0 + lk0] = elmt;
//           out[ji0 + kl0] = elmt;
//
//           int64_t ij1 = i * n + j;
//           int64_t ji1 = j * n + i;
//           int64_t kl1 = k * n3 + l * n2;
//           int64_t lk1 = l * n3 + k * n2;
//           out[kl1 + ij1] = elmt;
//           out[kl1 + ji1] = elmt;
//           out[lk1 + ji1] = elmt;
//           out[lk1 + ij1] = elmt;
//         }
//       }
//     }
//   }
//   // print64_tf("iarr: %d, size: %d\n", iarr, size);
// }
