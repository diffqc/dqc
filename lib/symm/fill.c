#include <stdlib.h>
#include <stdio.h>

void fills4(double* out, double* arr, int ni, int nk);
// void fills8(double* out, double* arr, int n);

void fills4(double* out, double* arr, int ni, int nk) {
  // fill out the full matrix from the reduced symmetric array (arr) with s4
  // symmetry: (ijkl) == (ijlk) == (jilk) == (jikl)
  // out should be initialized with all zeros
  int n2 = nk * nk;
  int n3 = ni * n2;
  int iarr = 0;
  for (int i = 0; i < ni; ++i) {
    for (int j = 0; j <= i; ++j) {
      int ij = i * n3 + j * n2;
      int ji = j * n3 + i * n2;
      for (int k = 0; k < nk; ++k) {
        for (int l = 0; l <= k; ++l) {
          double elmt = arr[iarr++];
          if (elmt == 0.0) {
            continue;
          }
          int kl = k * nk + l;
          int lk = l * nk + k;
          out[ij + kl] = elmt;
          out[ij + lk] = elmt;
          out[ji + lk] = elmt;
          out[ji + kl] = elmt;
        }
      }
    }
  }
}

// void fills8(double* out, double* arr, int n) {
//   // fill out the full matrix from the reduced symmetric array (arr) with s8
//   // symmetry: (ijkl) == (ijlk) == (jilk) == (jikl) == (klij) == (lkij) == (lkji)
//   // == (klji)
//   // out should be initialized with all zeros
//
//   int n2 = n * n;
//   int n3 = n2 * n;
//   int iarr = 0;
//   // int size1 = n * (n + 1) / 2;
//   // int size = size1 * (size1 + 1) / 2;
//   for (int i = 0; i < n; ++i) {
//     int i_idx = i * (i + 1) / 2;
//
//     for (int j = 0; j <= i; ++j) {
//       int ij_idx = i_idx + j;
//
//       for (int k = 0; k <= i; ++k) {
//         int lmax1 = ij_idx - k * (k + 1) / 2;
//         int lmax = lmax1 < k ? lmax1 : k;
//
//         for (int l = 0; l <= lmax; ++l) {
//           double elmt = arr[iarr];
//           iarr++;
//           if (elmt == 0.0) {
//             continue;
//           }
//
//           int ij0 = i * n3 + j * n2;
//           int ji0 = j * n3 + i * n2;
//           int kl0 = k * n + l;
//           int lk0 = l * n + k;
//           out[ij0 + kl0] = elmt;
//           out[ij0 + lk0] = elmt;
//           out[ji0 + lk0] = elmt;
//           out[ji0 + kl0] = elmt;
//
//           int ij1 = i * n + j;
//           int ji1 = j * n + i;
//           int kl1 = k * n3 + l * n2;
//           int lk1 = l * n3 + k * n2;
//           out[kl1 + ij1] = elmt;
//           out[kl1 + ji1] = elmt;
//           out[lk1 + ji1] = elmt;
//           out[lk1 + ij1] = elmt;
//         }
//       }
//     }
//   }
//   // printf("iarr: %d, size: %d\n", iarr, size);
// }
