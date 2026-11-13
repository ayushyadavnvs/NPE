#ifndef NPE_CONV_H_
#define NPE_CONV_H_

#include "npe_mem.h"

// Dense convolution: any kernel size, stride and padding, with an optional
// epilogue: INTERLEAVE, or ADD (residual; only in builds with NPE_HAS_ADD,
// so models without residuals do not pay its multipliers).
//
// Each clock cycle multiplies PI input words (PI * P channels) by PO x PI
// blocks of P x P weights and adds the results to PO x P running sums:
// P * P * PO * PI multipliers. One group of PO output words needs
// kh * kw * cin_g cycles; then the sums are requantized. Loop order: output
// pixel, output group (highest first), kernel row, kernel column, input
// group. Words past the end of the layer's channels (a partial last group)
// are masked to zero on input and not written on output.
//
// Input pixel of output pixel (oy, ox) and kernel position (ky, kx):
//   iy = oy * stride_h - pad_t + ky,   ix = ox * stride_w - pad_l + kx
// If (iy, ix) is outside the input, the cycle uses the padding word instead
// of a memory word: zero, or the layer's constant (stored after the
// layer's weights; only for one-word inputs, i.e. the image).
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
// Memory ports per cycle: each buffer is read once (NB consecutive words)
// and written once (up to 2 * PO consecutive words), see npe_mem.h.
inline void npe_conv(const npe_dec_t &d, NPE_BUF_PARAMS, NPE_MEM_PARAMS) {
#pragma HLS INLINE off
  npe_acc_t acc[NPE_PO][NPE_P]; // running sums of the current output group
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
  npe_word_t b_word[NPE_PO];    // B words of the current output group
#pragma HLS ARRAY_PARTITION variable = b_word complete
  // No word is read after it has been overwritten (see above).
#pragma HLS DEPENDENCE variable = S0 inter false
#pragma HLS DEPENDENCE variable = A inter false
#pragma HLS DEPENDENCE variable = B inter false
#pragma HLS DEPENDENCE variable = C inter false

  // Padding word: stored after the layer's weights.
  npe_word_t pad_word = 0;
  if (d.pad_const)
    pad_word = W[d.w_base + d.cout_g * d.w_per_out].range(8 * NPE_P - 1, 0);

  const int total = d.out_h * d.out_w * d.cout_g * d.w_per_out;
  const bool from_in = (d.a_buf == NPE_BUF_IN);
  int ox = 0;              // output pixel column
  int og = d.cout_g - 1;   // output group within the pixel, descending
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

conv_loop:
  for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
    const bool last_ig = (ig == d.cin_g - 1);
    const bool last_kx = (kx == d.kw - 1);
    const bool last_ky = (ky == d.kh - 1);
    const bool first_in = (ky == 0 && kx == 0 && ig == 0);
    const bool last_in = last_ky && last_kx && last_ig;
    const int iy = iy0 + ky, ix = ix0 + kx;
    const bool inside = (iy >= 0 && iy < d.a_h && ix >= 0 && ix < d.a_w);

    // Read. The A operand and the B operand are in different buffers.
    int a_addr = (inside && !from_in) ? a_word : 0; // no access outside
    int b_addr = b_pixel + og * NPE_PO;
    npe_word_t av[NPE_NB], bv[NPE_NB];
    npe_read_ab(d, S0, A, B, C, a_addr, b_addr, av, bv);
    npe_word_t image_word = IN[(inside && from_in) ? a_tap : 0]; // zero-extended
    npe_word_t x[NPE_PI];
  conv_in_word:
    for (int k = 0; k < NPE_PI; k++) {
      bool real = (ig * NPE_PI + k < d.cin_w);
      if (!inside)
        x[k] = (k == 0) ? pad_word : (npe_word_t)0;
      else if (from_in)
        x[k] = (k == 0) ? image_word : (npe_word_t)0;
      else
        x[k] = real ? av[k] : (npe_word_t)0;
    }
    if (d.use_b && first_in) {
    conv_b_word:
      for (int po = 0; po < NPE_PO; po++)
        b_word[po] = bv[po];
    }

    // Multiply-accumulate: PO x P outputs, PI x P inputs.
    npe_wword_t weights = W[w_addr];
    npe_qword_t quant = Q[d.q_base + og];
    npe_word_t result[NPE_PO]; // valid in the last input cycle
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
        const int q = po * NPE_QL;
        npe_acc_t bias = quant.range(q + 32 * o + 31, q + 32 * o);
        ap_int<32> M = quant.range(q + 32 * NPE_P + 32 * o + 31, q + 32 * NPE_P + 32 * o);
        ap_uint<6> S = quant.range(q + 64 * NPE_P + 8 * o + 5, q + 64 * NPE_P + 8 * o);
        ap_int<48> extra = 0; // ADD epilogue: b * Mb joins before the rounding
#if NPE_HAS_ADD
        if (d.add) {
          ap_int<32> Mb = quant.range(q + 72 * NPE_P + 32 * o + 31, q + 72 * NPE_P + 32 * o);
          extra = npe_lane(b_word[po], o, d.b_signed) * Mb;
        }
#endif
        result[po].range(8 * o + 7, 8 * o) =
            npe_requant(sum + bias, M, S, d.lo, d.hi, extra);
      }
    }

    // Output words: the results, or B and the results interleaved byte by
    // byte (result word j -> output words 2j, 2j + 1).
    npe_word_t ov[NPE_NB];
  conv_out_vec:
    for (int k = 0; k < NPE_NB; k++) {
      npe_word_t w = 0;
      if (d.interleave) {
        if (k < 2 * NPE_PO) {
        conv_mix:
          for (int byte = 0; byte < NPE_P; byte++) {
            int lane = (NPE_P * (k & 1) + byte) / 2; // channel within word k / 2
            w.range(8 * byte + 7, 8 * byte) =
                (byte & 1) ? result[k / 2].range(8 * lane + 7, 8 * lane)
                           : b_word[k / 2].range(8 * lane + 7, 8 * lane);
          }
        }
      } else if (k < NPE_PO) {
        w = result[k];
      }
      ov[k] = w;
    }
    int n_real = d.cout_w - og * NPE_PO; // real output words in this group
    if (n_real > NPE_PO)
      n_real = NPE_PO;
    int o_addr = o_pixel + (d.interleave ? 2 * NPE_PO * og : NPE_PO * og);
    npe_write_o(d, S0, A, B, C, last_in, o_addr,
                d.interleave ? 2 * n_real : n_real, ov);

    // Advance, innermost first: input group, kernel column, kernel row,
    // output group (descending), output pixel. Addresses only ever add.
    w_addr++;
    if (!last_ig) {
      ig++;
      a_word += NPE_PI;
    } else {
      ig = 0;
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
          if (og != 0) {
            og--;
            w_addr -= 2 * d.w_per_out; // weights of the next lower group
          } else {
            og = d.cout_g - 1;
            w_addr = d.w_base + og * d.w_per_out;
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
          }
          a_line = a_pixel;
          a_tap = a_pixel;
        }
      }
      a_word = a_tap;
    }
  }
}

#endif // NPE_CONV_H_
