#ifndef NPE_OUT_H_
#define NPE_OUT_H_

#include "npe_mem.h"

// Feature-map output: streams the first out_c channels of every pixel of A
// to OUT as int32, pixel by pixel in HWC order (a detection heatmap, a
// segmentation map; several heads are several OUT instructions in turn).
//
// One clock cycle per lane: the unit reads the pixel's channel word and
// writes lane l of it, so a pixel takes cout_w * P cycles. Lanes past out_c
// (the padding of the last word) are not written. Only in builds whose
// program has an OUT instruction (NPE_HAS_OUT).
inline void npe_out(const npe_dec_t &d, NPE_BUF_PARAMS) {
#pragma HLS INLINE off
  const int pixels = d.a_h * d.a_w;
  const int total = pixels * d.cout_w * NPE_P;
  int lane = 0, cw = 0;
  int pix_addr = d.a_first; // first word of the current pixel

out_loop:
  for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
    npe_word_t av[NPE_NB], bv[NPE_NB];
    npe_read_ab(d, S0, A, B, C, pix_addr + cw, pix_addr + cw, av, bv);
    const int ch = cw * NPE_P + lane;
    if (ch < d.out_c)
      OUT.write((npe_acc_t)npe_lane(av[0], lane, d.a_signed));

    // Advance: lane, then channel word, then pixel.
    if (lane != NPE_P - 1) {
      lane++;
    } else {
      lane = 0;
      if (cw != d.cout_w - 1) {
        cw++;
      } else {
        cw = 0;
        pix_addr += d.a_step;
      }
    }
  }
}

#endif // NPE_OUT_H_
