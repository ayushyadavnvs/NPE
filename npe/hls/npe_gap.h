#ifndef NPE_GAP_H_
#define NPE_GAP_H_

#include "npe_mem.h"

// Global average pool: the int32 sum of each channel over all H x W pixels,
// written to buffer V (one V word = P sums). The division by H * W is folded
// into the next layer's scale by the compiler.
//
// Each clock cycle reads one input word and adds its P lanes to P running
// sums. Loop order: channel word, then pixel; after the last pixel of a
// channel word, its P sums are written to V.
inline void npe_gap(const npe_dec_t &d, NPE_BUF_PARAMS) {
#pragma HLS INLINE off
  npe_acc_t acc[NPE_P];
#pragma HLS ARRAY_PARTITION variable = acc complete

  const int pixels = d.a_h * d.a_w;
  const int total = d.cout_w * pixels;
  int cw = 0, p = 0;
  int a_addr = d.a_first; // pixel p, channel word cw

gap_loop:
  for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
    const bool first = (p == 0);
    const bool last = (p == pixels - 1);
    npe_word_t av[NPE_NB], bv[NPE_NB];
    npe_read_ab(d, S0, A, B, C, a_addr, a_addr, av, bv);
    npe_word_t in_word = av[0];

    npe_vword_t sums;
  gap_lane:
    for (int c = 0; c < NPE_P; c++) {
      ap_int<9> x = npe_lane(in_word, c, d.a_signed);
      npe_acc_t s = (first ? (npe_acc_t)0 : acc[c]) + x;
      acc[c] = s;
      sums.range(32 * c + 31, 32 * c) = s;
    }
    if (last)
      V[d.o_first + cw] = sums;

    // Advance: pixel, then channel word.
    if (!last) {
      p++;
      a_addr += d.a_step;
    } else {
      p = 0;
      cw++;
      a_addr = d.a_first + cw;
    }
  }
}

#endif // NPE_GAP_H_
