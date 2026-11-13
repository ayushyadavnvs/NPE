#ifndef NPE_DWCONV_H_
#define NPE_DWCONV_H_

#include "npe_mem.h"

// Depthwise convolution: every channel is convolved with its own kh x kw
// kernel (groups = channels). Any kernel size, stride and zero padding.
//
// Each clock cycle reads PC consecutive channel words of one input pixel and
// multiplies them lane by lane with one kernel tap of those channels: P * PC
// multipliers. One group of PC output words needs kh * kw cycles. Loop
// order: output pixel, channel group, kernel row, kernel column. The weight
// address (WD) therefore just counts up within a pixel. Words past the
// layer's channels (a partial last group) are masked and not written.
//
// Input pixel of output pixel (oy, ox) and tap (ky, kx), and padding, are as
// in npe_conv. The output words are written once, in the last tap cycle; the
// input and output are in different buffers (checked by the packer).
inline void npe_dwconv(const npe_dec_t &d, NPE_BUF_PARAMS, NPE_MEM_PARAMS) {
#pragma HLS INLINE off
  npe_acc_t acc[NPE_PC][NPE_P]; // running sums of the current channel group
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
  // Reads and writes go to different buffers.
#pragma HLS DEPENDENCE variable = S0 inter false
#pragma HLS DEPENDENCE variable = A inter false
#pragma HLS DEPENDENCE variable = B inter false
#pragma HLS DEPENDENCE variable = C inter false

  const int taps = d.kh * d.kw;
  const int total = d.out_h * d.out_w * d.ch_g * taps;
  int ox = 0;              // output pixel column
  int cg = 0;              // channel group
  int ky = 0, kx = 0;      // kernel position
  int iy0 = -d.pad_t, ix0 = -d.pad_l; // input position of tap (0, 0)
  const int a_origin = d.a_first - d.pad_t * d.a_line_step - d.pad_l * d.a_step;
  int a_row = a_origin;    // address of tap (0, 0), group 0, pixel (oy, 0)
  int a_pixel = a_origin;  // ... pixel (oy, ox)
  int a_line = a_origin;   // address of tap (ky, 0), group cg
  int a_tap = a_origin;    // address of tap (ky, kx), group cg
  int o_pixel = d.o_first; // output word address of the current pixel
  int w_addr = d.w_base;   // next WD word
  int q_addr = d.q_base;   // QD word of the current channel group

dwconv_loop:
  for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
    const bool last_kx = (kx == d.kw - 1);
    const bool last_ky = (ky == d.kh - 1);
    const bool first_tap = (ky == 0 && kx == 0);
    const bool last_tap = last_ky && last_kx;
    const int iy = iy0 + ky, ix = ix0 + kx;
    const bool inside = (iy >= 0 && iy < d.a_h && ix >= 0 && ix < d.a_w);

    // Read PC words (zero at a padding position or past the channels).
    int a_addr = inside ? a_tap : 0;
    npe_word_t av[NPE_NB], bv[NPE_NB];
    npe_read_ab(d, S0, A, B, C, a_addr, a_addr, av, bv);

    npe_dwword_t weights = WD[w_addr];
    npe_qdword_t quant = QD[q_addr];
    npe_word_t result[NPE_NB];
  dwconv_word:
    for (int k = 0; k < NPE_NB; k++) {
      result[k] = 0;
      if (k < NPE_PC) {
        bool real = inside && (cg * NPE_PC + k < d.cout_w);
        npe_word_t x = real ? av[k] : (npe_word_t)0;
      dwconv_lane:
        for (int c = 0; c < NPE_P; c++) {
          int l = NPE_P * k + c; // lane within the group
          ap_int<8> w = weights.range(8 * l + 7, 8 * l);
          npe_acc_t sum = (first_tap ? (npe_acc_t)0 : acc[k][c]) + npe_lane(x, c, d.a_signed) * w;
          acc[k][c] = sum;
          const int q = 72 * NPE_P * k;
          npe_acc_t bias = quant.range(q + 32 * c + 31, q + 32 * c);
          ap_int<32> M = quant.range(q + 32 * NPE_P + 32 * c + 31, q + 32 * NPE_P + 32 * c);
          ap_uint<6> S = quant.range(q + 64 * NPE_P + 8 * c + 5, q + 64 * NPE_P + 8 * c);
          result[k].range(8 * c + 7, 8 * c) = npe_requant(sum + bias, M, S, d.lo, d.hi);
        }
      }
    }
    int n_real = d.cout_w - cg * NPE_PC;
    if (n_real > NPE_PC)
      n_real = NPE_PC;
    npe_write_o(d, S0, A, B, C, last_tap, o_pixel + cg * NPE_PC, n_real, result);

    // Advance, innermost first: kernel column, kernel row, channel group,
    // output pixel. Addresses only ever add.
    w_addr++;
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
          q_addr++;
        } else {
          cg = 0;
          q_addr = d.q_base;
          w_addr = d.w_base;
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
        a_line = a_pixel + cg * NPE_PC; // tap (0, 0) of the new group
        a_tap = a_line;
      }
    }
  }
}

#endif // NPE_DWCONV_H_
