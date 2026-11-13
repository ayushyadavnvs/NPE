#ifndef NPE_POOL_H_
#define NPE_POOL_H_

#include "npe_conv.h" // npe_bank_read / write

// Max pool: any window size, stride and padding. Padding positions are
// skipped (they never win), so the window max is over real pixels only.
//
// Each clock cycle reads one input word (P channels of one input pixel) and
// keeps the lane-wise maximum: P comparators. One output word needs kh * kw
// cycles. Loop order and addressing are the same as npe_dwconv: output
// pixel, channel word, kernel row, kernel column.
inline void npe_pool_max(const npe_dec_t &d, NPE_BUF_PARAMS) {
#pragma HLS INLINE off
  ap_int<9> best[NPE_P]; // running max of each lane, as a 9-bit signed value
#pragma HLS ARRAY_PARTITION variable = best complete
#pragma HLS DEPENDENCE variable = S0 inter false
#pragma HLS DEPENDENCE variable = A inter false
#pragma HLS DEPENDENCE variable = B inter false
#pragma HLS DEPENDENCE variable = C inter false

  const int taps = d.kh * d.kw;
  const int total = d.out_h * d.out_w * d.cout_w * taps;
  int ox = 0, cw = 0, ky = 0, kx = 0;
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
    npe_word_t ev[4], od[4];
    npe_bank_read(S0, d.a_buf == NPE_BUF_S0 ? a_addr >> 1 : 0, ev[0], od[0]);
    npe_bank_read(A, d.a_buf == NPE_BUF_A ? a_addr >> 1 : 0, ev[1], od[1]);
    npe_bank_read(B, d.a_buf == NPE_BUF_B ? a_addr >> 1 : 0, ev[2], od[2]);
    npe_bank_read(C, d.a_buf == NPE_BUF_C ? a_addr >> 1 : 0, ev[3], od[3]);
    int a_sel = d.a_buf - NPE_BUF_S0;
    npe_word_t in_word = (a_addr & 1) ? od[a_sel] : ev[a_sel];

    npe_word_t result;
  pool_lane:
    for (int c = 0; c < NPE_P; c++) {
      ap_uint<8> raw = in_word.range(8 * c + 7, 8 * c);
      ap_int<9> x = d.a_signed ? (ap_int<9>)(ap_int<8>)raw : (ap_int<9>)raw;
      // The first tap of a window starts the max; padding taps are skipped.
      ap_int<9> m = first_tap ? (ap_int<9>)-256 : best[c];
      if (inside && x > m)
        m = x;
      best[c] = m;
      result.range(8 * c + 7, 8 * c) = (ap_uint<8>)m;
    }

    // Write one word, in the last tap cycle, to the bank of its address.
    int o_addr = o_pixel + cw;
    bool odd = o_addr & 1;
    int row = o_addr >> 1;
    bool en0 = last_tap && !odd, en1 = last_tap && odd;
    npe_bank_write(S0, en0 && d.o_buf == NPE_BUF_S0, row, result,
                   en1 && d.o_buf == NPE_BUF_S0, row, result);
    npe_bank_write(A, en0 && d.o_buf == NPE_BUF_A, row, result,
                   en1 && d.o_buf == NPE_BUF_A, row, result);
    npe_bank_write(B, en0 && d.o_buf == NPE_BUF_B, row, result,
                   en1 && d.o_buf == NPE_BUF_B, row, result);
    npe_bank_write(C, en0 && d.o_buf == NPE_BUF_C, row, result,
                   en1 && d.o_buf == NPE_BUF_C, row, result);

    // Advance: kernel column, kernel row, channel word, output pixel.
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
        if (cw != d.cout_w - 1) {
          cw++;
        } else {
          cw = 0;
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
        a_line = a_pixel + cw;
        a_tap = a_line;
      }
    }
  }
}

#endif // NPE_POOL_H_
