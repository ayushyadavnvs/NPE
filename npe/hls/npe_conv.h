#ifndef NPE_CONV_H_
#define NPE_CONV_H_

#include "npe_mem.h"
#include "npe_wstream.h"

// Dense convolution: any kernel size, stride and padding, with an optional
// epilogue: INTERLEAVE, or ADD (residual; only in builds with NPE_HAS_ADD,
// so models without residuals do not pay its multipliers).
//
// Each clock cycle multiplies PI input words (PI * P channels) by PO x PI
// blocks of P x P weights and adds the results to PO x P running sums:
// P * P * PO * PI multipliers. One group of PO output words needs
// kh * kw * cin_g cycles; then the sums are requantized. Loop order: output
// group (highest first), output pixel, kernel row, kernel column, input
// group. With the output group outermost, a group's weights (kh * kw *
// cin_g words) serve every pixel before the next group's are needed, so
// weights can be streamed from DRAM once per inference. Words past the end of the layer's channels (a partial last group)
// are masked to zero on input and not written on output.
//
// Input pixel of output pixel (oy, ox) and kernel position (ky, kx):
//   iy = oy * stride_h - pad_t + ky,   ix = ox * stride_w - pad_l + kx
// If (iy, ix) is outside the input, the cycle uses the padding word instead
// of a memory word: zero, or the layer's constant (stored after the
// layer's weights; only for one-word inputs, i.e. the image).
//
// The image (IN) has fewer channels than a word (3 of 8), so an image layer
// packs IN_PPW pixels of a kernel row into the input word: pixel s of the
// step at lanes s * IN_C ..; a kernel row takes kw_c = ceil(kw / IN_PPW)
// cycles instead of kw. Each packed pixel has its own padding check.
//
//   plain       output words j of the group are written to the output view.
//   interleave  out channel 2i = B channel i, 2i+1 = result channel i. Result
//               word j and B word j together make output words 2j and 2j+1.
//               The output may be the same memory as B (in place): walking
//               groups from the highest down, the words a group writes
//               (>= 2 * PO * g) lie above every B word a lower group reads.
//   add         result = requant(acc) with B word j added in before the
//               rounding. The output may be the B view itself (in place).
//
// Requantization is shared, as in npe_dwconv: the sums (and B words) of a
// finished group move to a pending register in its last input cycle and are
// requantized and written RQC words per cycle (NPE_RQC, a knob) during the
// next group's cycles; the loop runs PO / RQC extra cycles at the end to drain.
// A group shorter than PO / RQC cycles (a 1x1 layer with few input words)
// is stretched with idle cycles, so the requant always keeps up.
//
// Memory ports per cycle: each buffer is read once (NB consecutive words)
// and written once (up to 2 * PO consecutive words), see npe_mem.h.
//
// Streamed weights (d.w_stream, see npe_wstream.h): W is read from the ring.
// Before the first input cycle of a group the loop waits (cycles that change
// nothing but the requant of the pending group) until the whole group is in
// the ring. Every cycle, streamed layer or not, takes in one stream beat if
// there is room, so the next layers' weights arrive during this one.
inline void npe_conv(const npe_dec_t &d, NPE_BUF_PARAMS, NPE_MEM_PARAMS) {
#pragma HLS INLINE off
  npe_acc_t acc[NPE_PO][NPE_P]; // running sums of the current output group
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
  npe_word_t b_word[NPE_PO];    // B words of the current output group
#pragma HLS ARRAY_PARTITION variable = b_word complete
  npe_acc_t pend[NPE_PO][NPE_P]; // sums of the last finished group
#pragma HLS ARRAY_PARTITION variable = pend complete dim = 0
  npe_word_t pend_b[NPE_PO];     // and its B words
#pragma HLS ARRAY_PARTITION variable = pend_b complete
  // No word is read after it has been overwritten (see above).
#pragma HLS DEPENDENCE variable = S0 inter false
#pragma HLS DEPENDENCE variable = A inter false
#pragma HLS DEPENDENCE variable = B inter false
#pragma HLS DEPENDENCE variable = C inter false
  // The ring slot written is never one that is being read (npe_wstream.h).
#pragma HLS DEPENDENCE variable = W inter false
#pragma HLS DEPENDENCE variable = W intra false

  // Padding word: stored after the layer's weights.
  npe_word_t pad_word = 0;
  if (d.pad_const)
    pad_word = W[d.w_base + d.cout_g * d.w_per_out].range(8 * NPE_P - 1, 0);

  const int drain = (NPE_PO + NPE_RQC - 1) / NPE_RQC; // cycles to requant a group
  const int stretch = drain > d.w_per_out ? drain - d.w_per_out : 0; // idle cycles per group
  const int steps = d.out_h * d.out_w * d.cout_g * (d.w_per_out + stretch);
  const int total = steps + drain;
  int hold = 0; // idle cycles left in the current group
  const bool from_in = (d.a_buf == NPE_BUF_IN);
  int ox = 0;              // output pixel column
  int og = d.cout_g - 1;   // output group, descending (outermost)
  int pix = 0;             // output pixel index within the group
  const int n_pix = d.out_h * d.out_w;
  int ky = 0, kx = 0;      // kernel position
  int ig = 0;              // input group within the pixel
  // Input position of kernel tap (0, 0) for the current output pixel, and
  // its word address. Both can lie outside the input (padding).
  int iy0 = -d.pad_t, ix0 = -d.pad_l;
  const int a_origin = d.a_first - d.pad_t * d.a_line_step - d.pad_l * d.a_step;
  int a_row = a_origin;    // address of tap (0, 0) for output pixel (oy, 0)
  int a_pixel = a_origin;  // ... for output pixel (oy, ox)
  int a_line = a_origin;   // address of tap (ky, 0)
  int a_tap = a_origin;    // address of tap (ky, kx)
  int a_word = a_origin;   // address of input group ig of tap (ky, kx)
  int b_pixel = d.b_first; // operand B word address of the current pixel
  int o_pixel = d.o_first; // output word address of the current pixel
  int w_addr = d.w_base + og * d.w_per_out; // next weight word
  // Pending group: its Q word, first output word, real words, and the next
  // of its words to requantize (pend_k >= pend_n: nothing pending).
  int pend_q = d.q_base, pend_o = 0, pend_n = 0, pend_k = 0;
  // Streamed weights: stream word of the current group's first word, and of
  // the word read this cycle.
  const bool streamed = d.w_stream;
  int s_group = WR.pos, s_addr = WR.pos;
  int s_need = WR.pos + d.w_per_out; // the ring must reach this word first
  bool g_first = true; // at the first input cycle of a group (pixel 0)

  int i = 0; // cycles that did work (not waiting for weights)
conv_loop:
  while (i < total) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 100000
    const bool last_ig = (ig == d.cin_g - 1);
    const bool last_kx = (kx == d.kw_c - 1);
    const bool last_ky = (ky == d.kh - 1);
    const bool idle = (hold > 0);   // stretch cycle: no new input
    const bool tap0 = (ky == 0 && kx == 0 && ig == 0) && !idle;
    // Waiting for a streamed group's weights (only before its first cycle).
    const bool wait = streamed && g_first && WR.wr < s_need;
    npe_ws_take(WS, WR, W, streamed ? s_group : WR.pos);
    const int i_now = i;
    if (!wait)
      i++;
    const bool quiet = idle || wait; // no new input this cycle
    const bool first_in = tap0 && !wait;
    const bool last_in = last_ky && last_kx && last_ig;
    // The group ends in its last input cycle, or its last idle cycle.
    const bool done = wait ? false : idle ? (hold == 1) : (last_in && stretch == 0);
    const int iy = iy0 + ky, ix = ix0 + kx;
    const bool inside = (iy >= 0 && iy < d.a_h && ix >= 0 && ix < d.a_w);

    // Read. The A operand and the B operand are in different buffers.
    int a_addr = (inside && !from_in) ? a_word : 0; // no access outside
    int b_addr = b_pixel + og * NPE_PO;
    npe_word_t av[NPE_NB], bv[NPE_NB];
    npe_read_ab(d, S0, A, B, C, a_addr, b_addr, av, bv);
    // Image layer: IN_PPW pixels of the kernel row, each checked for padding.
    npe_pixel_t px[NPE_IN_PPB];
    npe_in_read(IN, from_in ? a_tap : 0, px);
    npe_word_t image_word = 0;
  conv_in_pixel:
    for (int s = 0; s < NPE_IN_PPW; s++) {
      const int kxs = kx * NPE_IN_PPW + s; // kernel column of this pixel
      const int ixs = ix0 + kxs;
      const bool in_s = (iy >= 0 && iy < d.a_h && ixs >= 0 && ixs < d.a_w);
      npe_pixel_t v = in_s ? px[s] : (npe_pixel_t)pad_word.range(8 * NPE_IN_C - 1, 0);
      if (kxs < d.kw)
        image_word.range(8 * NPE_IN_C * (s + 1) - 1, 8 * NPE_IN_C * s) = v;
    }
    npe_word_t x[NPE_PI];
  conv_in_word:
    for (int k = 0; k < NPE_PI; k++) {
      bool real = (ig * NPE_PI + k < d.cin_w);
      if (quiet)
        x[k] = 0;
      else if (from_in)
        x[k] = (k == 0) ? image_word : (npe_word_t)0;
      else if (!inside)
        x[k] = (k == 0) ? pad_word : (npe_word_t)0;
      else
        x[k] = real ? av[k] : (npe_word_t)0;
    }
    if (d.use_b && first_in) {
    conv_b_word:
      for (int po = 0; po < NPE_PO; po++)
        b_word[po] = bv[po];
    }

    const bool step = (i_now < steps); // else: draining the last group

    // Requantize RQC words of the pending group.
    npe_qword_t quant = Q[pend_q];
    npe_word_t result[NPE_RQC], rb[NPE_RQC]; // results and their B words
  conv_rq_word:
    for (int j = 0; j < NPE_RQC; j++) {
      // word pend_k + j of the group; pend_k is a multiple of RQC
      npe_word_t bw = 0;
    conv_rq_b:
      for (int k = j; k < NPE_PO; k += NPE_RQC)
        if (k - j == pend_k)
          bw = pend_b[k];
      rb[j] = bw;
      // Quant block of that word: PO constant slices, one picked (a run-time
      // shift of the whole Q word costs ~50k LUT at PO 4).
      ap_uint<NPE_QL> qs = 0;
    conv_rq_q:
      for (int k = j; k < NPE_PO; k += NPE_RQC)
        if (k - j == pend_k)
          qs = quant.range(k * NPE_QL + NPE_QL - 1, k * NPE_QL);
    conv_rq_lane:
      for (int o = 0; o < NPE_P; o++) {
        npe_acc_t sum = 0, bias = 0;
        ap_int<32> M = 0, Mb = 0;
        ap_uint<6> S = 1;
      conv_rq_sel:
        for (int k = j; k < NPE_PO; k += NPE_RQC) {
          if (k - j == pend_k) {
            sum = pend[k][o];
          }
        }
        bias = qs.range(32 * o + 31, 32 * o);
        M = qs.range(32 * NPE_P + 32 * o + 31, 32 * NPE_P + 32 * o);
        S = qs.range(64 * NPE_P + 8 * o + 5, 64 * NPE_P + 8 * o);
        if (S == 0) // no word pending: keep the rounding shift defined
          S = 1;
#if NPE_HAS_ADD
        Mb = qs.range(72 * NPE_P + 32 * o + 31, 72 * NPE_P + 32 * o);
#endif
        ap_int<48> extra = 0; // ADD epilogue: b * Mb joins before the rounding
        if (NPE_HAS_ADD && d.add)
          extra = npe_lane(bw, o, d.b_signed) * Mb;
        result[j].range(8 * o + 7, 8 * o) = npe_requant(sum + bias, M, S, d.lo, d.hi, extra);
      }
    }

    // Output words: the results, or B and the results interleaved byte by
    // byte (result word j -> output words 2j, 2j + 1).
    npe_word_t ov[NPE_NB];
  conv_out_vec:
    for (int k = 0; k < NPE_NB; k++) {
      npe_word_t w = 0;
      if (d.interleave) {
        if (k < 2 * NPE_RQC) {
        conv_mix:
          for (int byte = 0; byte < NPE_P; byte++) {
            int lane = (NPE_P * (k & 1) + byte) / 2; // channel within word k / 2
            w.range(8 * byte + 7, 8 * byte) =
                (byte & 1) ? result[k / 2].range(8 * lane + 7, 8 * lane)
                           : rb[k / 2].range(8 * lane + 7, 8 * lane);
          }
        }
      } else if (k < NPE_RQC) {
        w = result[k];
      }
      ov[k] = w;
    }
    int n_wr = pend_n - pend_k;
    if (n_wr > NPE_RQC)
      n_wr = NPE_RQC;
    npe_write_o(d, S0, A, B, C, n_wr > 0,
                pend_o + (d.interleave ? 2 * pend_k : pend_k),
                d.interleave ? 2 * n_wr : n_wr, ov);
    if (pend_k < pend_n)
      pend_k += NPE_RQC;

    // Multiply-accumulate: PO x P outputs, PI x P inputs.
    npe_wword_t weights =
        W[streamed ? NPE_W_RING_BASE + (s_addr & (NPE_W_RING - 1)) : w_addr];
  conv_out_word:
    for (int po = 0; po < NPE_PO; po++) {
    conv_out_ch:
      for (int o = 0; o < NPE_P; o++) {
        npe_acc_t partial = 0;
      conv_in_ch:
        for (int k = 0; k < NPE_PI; k++) {
          for (int c = 0; c < NPE_P; c++) {
            int wb = ((po * NPE_PI + k) * NPE_P + o) * NPE_P + c;
            ap_int<8> w = weights.range(8 * wb + 7, 8 * wb);
            partial += npe_lane(x[k], c, d.a_signed) * w;
          }
        }
        npe_acc_t sum = (first_in ? (npe_acc_t)0 : acc[po][o]) + partial;
        acc[po][o] = sum;
        if (step && done)
          pend[po][o] = sum;
      }
      if (step && done)
        pend_b[po] = (d.use_b && first_in) ? bv[po] : b_word[po];
    }
    if (step && done) { // the group is done: it becomes the pending one
      int n_real = d.cout_w - og * NPE_PO;
      pend_n = n_real > NPE_PO ? NPE_PO : n_real;
      pend_k = 0;
      pend_q = d.q_base + og;
      pend_o = o_pixel + (d.interleave ? 2 * NPE_PO * og : NPE_PO * og);
    }
    if (!step || wait)
      continue;
    g_first = false;
    if (!idle && last_in && stretch > 0) { // start the idle cycles
      hold = stretch;
      continue;
    }
    if (idle) {
      hold--;
      if (hold > 0)
        continue;
    }

    // Advance, innermost first: input group, kernel column, kernel row,
    // output pixel, output group (descending).
    w_addr++;
    s_addr++;
    if (!last_ig) {
      ig++;
      a_word += NPE_PI;
    } else {
      ig = 0;
      if (!last_kx) {
        kx++;
        a_tap += d.a_kstep;
      } else {
        kx = 0;
        if (!last_ky) {
          ky++;
          a_line += d.a_line_step;
          a_tap = a_line;
        } else {
          ky = 0;
          w_addr -= d.w_per_out; // the same group's weights, next pixel
          s_addr -= d.w_per_out;
          if (pix != n_pix - 1) {
            pix++;
            b_pixel += d.b_step;
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
          } else if (og != 0) { // next lower group, from the first pixel
            pix = 0;
            og--;
            w_addr = d.w_base + og * d.w_per_out;
            s_group += d.w_per_out; // the stream holds the groups in this order
            s_addr = s_group;
            s_need += d.w_per_out;
            g_first = true;
            b_pixel = d.b_first;
            o_pixel = d.o_first;
            ox = 0;
            ix0 = -d.pad_l;
            iy0 = -d.pad_t;
            a_row = a_origin;
            a_pixel = a_origin;
          }
          a_line = a_pixel;
          a_tap = a_pixel;
        }
      }
      a_word = a_tap;
    }
  }
  if (streamed)
    WR.pos += d.cout_g * d.w_per_out;
}

#endif // NPE_CONV_H_
