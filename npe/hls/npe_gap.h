#ifndef NPE_GAP_H_
#define NPE_GAP_H_

#include "npe_conv.h" // npe_bank_read

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
    npe_word_t ev[4], od[4];
    npe_bank_read(S0, d.a_buf == NPE_BUF_S0 ? a_addr >> 1 : 0, ev[0], od[0]);
    npe_bank_read(A, d.a_buf == NPE_BUF_A ? a_addr >> 1 : 0, ev[1], od[1]);
    npe_bank_read(B, d.a_buf == NPE_BUF_B ? a_addr >> 1 : 0, ev[2], od[2]);
    npe_bank_read(C, d.a_buf == NPE_BUF_C ? a_addr >> 1 : 0, ev[3], od[3]);
    int a_sel = d.a_buf - NPE_BUF_S0;
    npe_word_t in_word = (a_addr & 1) ? od[a_sel] : ev[a_sel];

    npe_vword_t sums;
  gap_lane:
    for (int c = 0; c < NPE_P; c++) {
      ap_uint<8> raw = in_word.range(8 * c + 7, 8 * c);
      ap_int<9> x = d.a_signed ? (ap_int<9>)(ap_int<8>)raw : (ap_int<9>)raw;
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
