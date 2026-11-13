#ifndef NPE_FC_H_
#define NPE_FC_H_

#include "npe_decode.h"

// Fully connected layer on an int32 vector: out[k] = sum_c V[c] * w[k][c] +
// bias[k], written as raw int32 to the OUT stream (logits, no requant).
//
// Each clock cycle multiplies one V word (P int32) by a P x P weight block
// (same W layout as a dense 1x1 conv) into P running sums. After the input
// words of one output word, its real outputs (the last word can be partly
// padding, e.g. 3 classes) are streamed out one per cycle.
// The sums are 64-bit; the result is the low 32 bits, exact when it fits in
// int32 (the reference checks that it does).
inline void npe_fc(const npe_dec_t &d, NPE_BUF_PARAMS, NPE_MEM_PARAMS) {
#pragma HLS INLINE off
  ap_int<64> acc[NPE_P];
#pragma HLS ARRAY_PARTITION variable = acc complete

fc_out_word:
  for (int ow = 0; ow < d.cout_w; ow++) {
  fc_mac:
    for (int iw = 0; iw < d.cin_w; iw++) {
#pragma HLS PIPELINE II = 1
      npe_vword_t x = V[d.a_first + iw];
      npe_wword_t weights = W[d.w_base + ow * d.cin_w + iw];
    fc_out_ch:
      for (int o = 0; o < NPE_P; o++) {
        ap_int<64> partial = 0;
      fc_in_ch:
        for (int c = 0; c < NPE_P; c++) {
          ap_int<32> xv = x.range(32 * c + 31, 32 * c);
          ap_int<8> w = weights.range(8 * (NPE_P * o + c) + 7, 8 * (NPE_P * o + c));
          partial += xv * w;
        }
        acc[o] = (iw == 0 ? (ap_int<64>)0 : acc[o]) + partial;
      }
    }
    npe_qword_t quant = Q[d.q_base + ow];
  fc_write:
    for (int o = 0; o < NPE_P; o++) {
#pragma HLS PIPELINE II = 1
      if (NPE_P * ow + o < d.out_c) {
        npe_acc_t bias = quant.range(32 * o + 31, 32 * o);
        OUT.write((npe_acc_t)(acc[o] + bias));
      }
    }
  }
}

#endif // NPE_FC_H_
