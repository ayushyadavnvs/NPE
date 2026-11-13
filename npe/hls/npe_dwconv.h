#ifndef NPE_DWCONV_H_
#define NPE_DWCONV_H_

#include "npe_conv.h" // npe_requant, npe_bank_read / write, npe_read_row

// Depthwise convolution: every channel is convolved with its own kh x kw
// kernel (groups = channels). Any kernel size, stride and zero padding.
//
// Each clock cycle reads one input word (P channels of one input pixel) and
// multiplies it lane by lane with one kernel tap of those P channels: P
// multipliers. One output word needs kh * kw cycles. Loop order: output
// pixel, channel word, kernel row, kernel column. The weight address (WD)
// therefore just counts up within a pixel.
//
// Input pixel of output pixel (oy, ox) and tap (ky, kx), and padding, are as
// in npe_conv. The output word is written once, in the last tap cycle; the
// input and output are in different buffers (checked by the packer).
inline void npe_dwconv(const npe_dec_t &d, NPE_BUF_PARAMS, NPE_MEM_PARAMS) {
#pragma HLS INLINE off
  npe_acc_t acc[NPE_P]; // running sums of the current channel word
#pragma HLS ARRAY_PARTITION variable = acc complete
  // Reads and writes go to different buffers.
#pragma HLS DEPENDENCE variable = S0 inter false
#pragma HLS DEPENDENCE variable = A inter false
#pragma HLS DEPENDENCE variable = B inter false
#pragma HLS DEPENDENCE variable = C inter false

  const int taps = d.kh * d.kw;
  const int total = d.out_h * d.out_w * d.cout_w * taps;
  int ox = 0, oy = 0;      // output pixel
  int cw = 0;              // channel word
  int ky = 0, kx = 0;      // kernel position
  int iy0 = -d.pad_t, ix0 = -d.pad_l; // input position of tap (0, 0)
  const int a_origin = d.a_first - d.pad_t * d.a_line_step - d.pad_l * d.a_step;
  int a_row = a_origin;    // address of tap (0, 0), channel word 0, pixel (oy, 0)
  int a_pixel = a_origin;  // ... pixel (oy, ox)
  int a_line = a_origin;   // address of tap (ky, 0), channel word cw
  int a_tap = a_origin;    // address of tap (ky, kx), channel word cw
  int o_pixel = d.o_first; // output word address of the current pixel
  int w_addr = d.w_base;   // next WD word
  int q_addr = d.q_base;   // Q word of the current channel word

dwconv_loop:
  for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
    const bool last_kx = (kx == d.kw - 1);
    const bool last_ky = (ky == d.kh - 1);
    const bool first_tap = (ky == 0 && kx == 0);
    const bool last_tap = last_ky && last_kx;
    const int iy = iy0 + ky, ix = ix0 + kx;
    const bool inside = (iy >= 0 && iy < d.a_h && ix >= 0 && ix < d.a_w);

    // Read: one input word (zero at a padding position).
    int a_addr = inside ? a_tap : 0;
    npe_word_t ev[4], od[4];
    npe_bank_read(S0, d.a_buf == NPE_BUF_S0 ? a_addr >> 1 : 0, ev[0], od[0]);
    npe_bank_read(A, d.a_buf == NPE_BUF_A ? a_addr >> 1 : 0, ev[1], od[1]);
    npe_bank_read(B, d.a_buf == NPE_BUF_B ? a_addr >> 1 : 0, ev[2], od[2]);
    npe_bank_read(C, d.a_buf == NPE_BUF_C ? a_addr >> 1 : 0, ev[3], od[3]);
    int a_sel = d.a_buf - NPE_BUF_S0;
    npe_word_t in_word = 0;
    if (inside)
      in_word = (a_addr & 1) ? od[a_sel] : ev[a_sel];

    // Multiply-accumulate: lane c of the input times lane c of the tap.
    npe_word_t weights = WD[w_addr];
    npe_qword_t quant = Q[q_addr];
    npe_word_t result; // valid in the last tap cycle
  dwconv_lane:
    for (int c = 0; c < NPE_P; c++) {
      ap_uint<8> raw = in_word.range(8 * c + 7, 8 * c);
      ap_int<9> x = d.a_signed ? (ap_int<9>)(ap_int<8>)raw : (ap_int<9>)raw;
      ap_int<8> w = weights.range(8 * c + 7, 8 * c);
      npe_acc_t sum = (first_tap ? (npe_acc_t)0 : acc[c]) + x * w;
      acc[c] = sum;
      npe_acc_t bias = quant.range(32 * c + 31, 32 * c);
      ap_int<32> M = quant.range(32 * NPE_P + 32 * c + 31, 32 * NPE_P + 32 * c);
      ap_uint<6> S = quant.range(64 * NPE_P + 8 * c + 5, 64 * NPE_P + 8 * c);
      result.range(8 * c + 7, 8 * c) = npe_requant(sum + bias, M, S, d.lo, d.hi);
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

    // Advance, innermost first: kernel column, kernel row, channel word,
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
        if (cw != d.cout_w - 1) {
          cw++;
          q_addr++;
        } else {
          cw = 0;
          q_addr = d.q_base;
          w_addr = d.w_base;
          o_pixel += d.o_step;
          if (ox != d.out_w - 1) {
            ox++;
            ix0 += d.stride_w;
            a_pixel += d.a_col_step;
          } else { // next output row
            ox = 0;
            oy++;
            ix0 = -d.pad_l;
            iy0 += d.stride_h;
            a_row += d.a_row_step;
            a_pixel = a_row;
          }
        }
        a_line = a_pixel + cw; // tap (0, 0) of the new channel word
        a_tap = a_line;
      }
    }
  }
}

#endif // NPE_DWCONV_H_
