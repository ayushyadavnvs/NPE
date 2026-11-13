#ifndef NPE_POOL_H_
#define NPE_POOL_H_

#include "npe_mem.h"

// Max pool: any window size, stride and padding. Padding positions are
// skipped (they never win), so the window max is over real pixels only.
//
// Each clock cycle reads PC consecutive channel words of one input pixel and
// keeps the lane-wise maximum: P * PC comparators. One group of PC output
// words needs kh * kw cycles. Loop order and addressing are the same as
// npe_dwconv: output pixel, channel group, kernel row, kernel column.
inline void npe_pool_max(const npe_dec_t &d, NPE_BUF_PARAMS) {
#pragma HLS INLINE off
  ap_int<9> best[NPE_PC][NPE_P]; // running max of each lane, 9-bit signed
#pragma HLS ARRAY_PARTITION variable = best complete dim = 0
#pragma HLS DEPENDENCE variable = S0 inter false
#pragma HLS DEPENDENCE variable = A inter false
#pragma HLS DEPENDENCE variable = B inter false
#pragma HLS DEPENDENCE variable = C inter false

  const int taps = d.kh * d.kw;
  const int total = d.out_h * d.out_w * d.ch_g * taps;
  int ox = 0, cg = 0, ky = 0, kx = 0;
  int iy0 = -d.pad_t, ix0 = -d.pad_l; // input position of tap (0, 0)
  const int a_origin = d.a_first - d.pad_t * d.a_line_step - d.pad_l * d.a_step;
  int a_row = a_origin, a_pixel = a_origin, a_line = a_origin, a_tap = a_origin;
  int o_pixel = d.o_first;

pool_loop:
  for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
    const bool last_kx = (kx == d.kw - 1);
    const bool last_ky = (ky == d.kh - 1);
    const bool first_tap = (ky == 0 && kx == 0);
    const bool last_tap = last_ky && last_kx;
    const int iy = iy0 + ky, ix = ix0 + kx;
    const bool inside = (iy >= 0 && iy < d.a_h && ix >= 0 && ix < d.a_w);

    int a_addr = inside ? a_tap : 0;
    npe_word_t av[NPE_NB], bv[NPE_NB];
    npe_read_ab(d, S0, A, B, C, a_addr, a_addr, av, bv);

    npe_word_t result[NPE_NB];
  pool_word:
    for (int k = 0; k < NPE_NB; k++) {
      result[k] = 0;
      if (k < NPE_PC) {
      pool_lane:
        for (int c = 0; c < NPE_P; c++) {
          ap_int<9> x = npe_lane(av[k], c, d.a_signed);
          // The first tap of a window starts the max; padding taps are skipped.
          ap_int<9> m = first_tap ? (ap_int<9>)-256 : best[k][c];
          if (inside && x > m)
            m = x;
          best[k][c] = m;
          result[k].range(8 * c + 7, 8 * c) = (ap_uint<8>)m;
        }
      }
    }
    int n_real = d.cout_w - cg * NPE_PC;
    if (n_real > NPE_PC)
      n_real = NPE_PC;
    npe_write_o(d, S0, A, B, C, last_tap, o_pixel + cg * NPE_PC, n_real, result);

    // Advance: kernel column, kernel row, channel group, output pixel.
    if (!last_kx) {
      kx++;
      a_tap += d.a_step;
    } else {
      kx = 0;
      if (!last_ky) {
        ky++;
        a_line += d.a_line_step;
        a_tap = a_line;
      } else {
        ky = 0;
        if (cg != d.ch_g - 1) {
          cg++;
        } else {
          cg = 0;
          o_pixel += d.o_step;
          if (ox != d.out_w - 1) {
            ox++;
            ix0 += d.stride_w;
            a_pixel += d.a_col_step;
          } else { // next output row
            ox = 0;
            ix0 = -d.pad_l;
            iy0 += d.stride_h;
            a_row += d.a_row_step;
            a_pixel = a_row;
          }
        }
        a_line = a_pixel + cg * NPE_PC;
        a_tap = a_line;
      }
    }
  }
}

#endif // NPE_POOL_H_
