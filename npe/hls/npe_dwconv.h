#ifndef NPE_DWCONV_H_
#define NPE_DWCONV_H_

#include "npe_mem.h"
#include "npe_wstream.h"

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
// in npe_conv. The input and output are in different buffers (checked by
// the packer).
//
// Requantization is shared: the sums of a finished group are copied to a
// pending register in its last tap cycle, and requantized and written RQ
// words per cycle (NPE_RQ, a knob) during the next group's taps. That needs
// kh * kw * RQ >= PC (checked by the packer); the loop runs PC / RQ extra
// cycles at the end to drain the last group.
//
// Each cycle also takes in a weight-stream beat for later streamed conv
// layers, if the ring has room (npe_wstream.h).
inline void npe_dwconv(const npe_dec_t &d, NPE_BUF_PARAMS, NPE_MEM_PARAMS) {
#pragma HLS INLINE off
  npe_acc_t acc[NPE_PC][NPE_P]; // running sums of the current channel group
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
  npe_acc_t pend[NPE_PC][NPE_P]; // sums of the last finished group
#pragma HLS ARRAY_PARTITION variable = pend complete dim = 0
  // Reads and writes go to different buffers.
#pragma HLS DEPENDENCE variable = S0 inter false
#pragma HLS DEPENDENCE variable = A inter false
#pragma HLS DEPENDENCE variable = B inter false
#pragma HLS DEPENDENCE variable = C inter false
#pragma HLS DEPENDENCE variable = W inter false

  const int taps = d.kh * d.kw;
  const int steps = d.out_h * d.out_w * d.ch_g * taps;
  const int drain = (NPE_PC + NPE_RQ - 1) / NPE_RQ; // cycles to requant a group
  const int total = steps + drain;
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
  // Pending group: its QD word, first output word, real words, and the next
  // of its words to requantize (pend_k >= pend_n: nothing pending).
  int pend_q = d.q_base, pend_o = 0, pend_n = 0, pend_k = 0;

dwconv_loop:
  for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
    npe_ws_take(WS, WR, W, WR.pos); // weights of later streamed layers
    const bool last_kx = (kx == d.kw - 1);
    const bool last_ky = (ky == d.kh - 1);
    const bool first_tap = (ky == 0 && kx == 0);
    const bool last_tap = last_ky && last_kx;
    const int iy = iy0 + ky, ix = ix0 + kx;
    const bool inside = (iy >= 0 && iy < d.a_h && ix >= 0 && ix < d.a_w);

    const bool step = (i < steps); // else: draining the last group

    // Read PC words (zero at a padding position or past the channels).
    int a_addr = inside ? a_tap : 0;
    npe_word_t av[NPE_NB], bv[NPE_NB];
    npe_read_ab(d, S0, A, B, C, a_addr, a_addr, av, bv);

    // Requantize and write RQ words of the pending group.
    npe_qdword_t quant = QD[pend_q];
    npe_word_t result[NPE_NB];
  dwconv_rq_word:
    for (int j = 0; j < NPE_NB; j++) {
      result[j] = 0;
      if (j < NPE_RQ) {
      dwconv_rq_lane:
        for (int c = 0; c < NPE_P; c++) {
          // word pend_k + j of the group; pend_k is a multiple of RQ
          npe_acc_t sum = 0;
          ap_int<32> M = 0;
          ap_uint<6> S = 1;
          npe_acc_t bias = 0;
        dwconv_rq_sel:
          for (int k = j; k < NPE_PC; k += NPE_RQ) {
            if (k - j == pend_k) {
              sum = pend[k][c];
              const int q = 72 * NPE_P * k;
              bias = quant.range(q + 32 * c + 31, q + 32 * c);
              M = quant.range(q + 32 * NPE_P + 32 * c + 31, q + 32 * NPE_P + 32 * c);
              S = quant.range(q + 64 * NPE_P + 8 * c + 5, q + 64 * NPE_P + 8 * c);
            }
          }
          result[j].range(8 * c + 7, 8 * c) = npe_requant(sum + bias, M, S, d.lo, d.hi);
        }
      }
    }
    int n_wr = pend_n - pend_k;
    if (n_wr > NPE_RQ)
      n_wr = NPE_RQ;
    npe_write_o(d, S0, A, B, C, n_wr > 0, pend_o + pend_k, n_wr, result);
    if (pend_k < pend_n)
      pend_k += NPE_RQ;

    // Multiply-accumulate one tap of PC words.
    npe_dwword_t weights = WD[w_addr];
  dwconv_word:
    for (int k = 0; k < NPE_PC; k++) {
      bool real = inside && (cg * NPE_PC + k < d.cout_w);
      npe_word_t x = real ? av[k] : (npe_word_t)0;
    dwconv_lane:
      for (int c = 0; c < NPE_P; c++) {
        int l = NPE_P * k + c; // lane within the group
        ap_int<8> w = weights.range(8 * l + 7, 8 * l);
        npe_acc_t sum = (first_tap ? (npe_acc_t)0 : acc[k][c]) + npe_lane(x, c, d.a_signed) * w;
        acc[k][c] = sum;
        if (step && last_tap)
          pend[k][c] = sum;
      }
    }
    if (step && last_tap) { // the group is done: it becomes the pending one
      int n_real = d.cout_w - cg * NPE_PC;
      pend_n = n_real > NPE_PC ? NPE_PC : n_real;
      pend_k = 0;
      pend_q = q_addr;
      pend_o = o_pixel + cg * NPE_PC;
    }
    if (!step)
      continue;

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
