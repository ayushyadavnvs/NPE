#ifndef NPE_CONV1X1_H_
#define NPE_CONV1X1_H_

#include "npe_decode.h"

// out = clamp((acc * M + 2^(S-1)) >> S, lo, hi). Same rule as the reference
// (npe/ref/isa_ref.py, _requant). The result is returned as a raw byte: for a
// signed output it is the two's complement int8.
inline ap_uint<8> npe_requant(npe_acc_t acc, ap_int<32> M, ap_uint<6> S, int lo,
                              int hi) {
#pragma HLS INLINE
  ap_int<64> prod = (ap_int<64>)acc * M;
  ap_int<64> rounded = (prod + ((ap_int<64>)1 << (S - 1))) >> S;
  if (rounded < lo)
    rounded = lo;
  if (rounded > hi)
    rounded = hi;
  return (ap_uint<8>)rounded;
}

// Reads row `row` of both banks of a buffer. One read statement per bank.
template <int N>
inline void npe_bank_read(const npe_word_t buf[2][N], int row, npe_word_t &even,
                          npe_word_t &odd) {
#pragma HLS INLINE
  even = buf[0][row];
  odd = buf[1][row];
}

// Row that buffer `buf` reads this cycle: the A operand's if the instruction
// reads A from it, operand B's if it reads B from it, else row 0 (unused).
inline int npe_read_row(const npe_dec_t &d, int buf, int a_addr, int b_addr) {
#pragma HLS INLINE
  if (d.a_buf == buf)
    return a_addr >> 1;
  if (d.interleave && d.b_buf == buf)
    return b_addr >> 1;
  return 0;
}

// Writes up to one word per bank. One write statement per bank.
template <int N>
inline void npe_bank_write(npe_word_t buf[2][N], bool en0, int row0,
                           npe_word_t v0, bool en1, int row1, npe_word_t v1) {
#pragma HLS INLINE
  if (en0)
    buf[0][row0] = v0;
  if (en1)
    buf[1][row1] = v1;
}

// 1x1 convolution, any stride, with the optional INTERLEAVE epilogue.
//
// Each clock cycle multiplies one input word (P channels) by a P x P block of
// weights and adds the P results to P running sums: P * P multipliers. After
// cin_w cycles the sums are complete; they are requantized into one output
// word. Loop order: output pixel, output word (highest first), input word.
//
//   plain       output word j of the pixel is written to the output view.
//   interleave  out channel 2i = B channel i, 2i+1 = result channel i. Result
//               word j and B word j together make output words 2j and 2j+1.
//               The output may be the same memory as B (in place): walking j
//               from the highest down, words 2j and 2j+1 are written only
//               after every B word that still has to be read lies below them.
//
// Memory ports per cycle: each buffer is read at one row (both banks) and
// written at most one word per bank. Words 2j and 2j+1 are neighbours, so
// one is even and one is odd.
inline void npe_conv1x1(const npe_dec_t &d, NPE_BUF_PARAMS, NPE_MEM_PARAMS) {
#pragma HLS INLINE off
  npe_acc_t acc[NPE_P]; // running sums of the current output word
#pragma HLS ARRAY_PARTITION variable = acc complete
  // No word is read after it has been overwritten (see above).
#pragma HLS DEPENDENCE variable = A inter false
#pragma HLS DEPENDENCE variable = B inter false
#pragma HLS DEPENDENCE variable = C inter false

  const int total = d.out_h * d.out_w * d.cout_w * d.cin_w;
  int ox = 0, oy = 0;         // output pixel
  int ow = d.cout_w - 1;      // output word within the pixel, descending
  int iw = 0;                 // input word within the pixel
  int a_row = d.a_first;      // input word address of pixel (oy * sh, 0)
  int a_pixel = d.a_first;    // input word address of the current pixel
  int b_pixel = d.b_first;    // operand B word address of the current pixel
  int o_pixel = d.o_first;    // output word address of the current pixel
  npe_word_t b_word = 0;      // B word of the current output word

conv1x1_loop:
  for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
    const bool first_in = (iw == 0);
    const bool last_in = (iw == d.cin_w - 1);

    // Read. The A operand and the B operand are in different buffers.
    int a_addr = a_pixel + iw;
    int b_addr = b_pixel + ow;
    npe_word_t ev[3], od[3];
    npe_bank_read(A, npe_read_row(d, NPE_BUF_A, a_addr, b_addr), ev[0], od[0]);
    npe_bank_read(B, npe_read_row(d, NPE_BUF_B, a_addr, b_addr), ev[1], od[1]);
    npe_bank_read(C, npe_read_row(d, NPE_BUF_C, a_addr, b_addr), ev[2], od[2]);
    int a_sel = d.a_buf - NPE_BUF_A, b_sel = d.b_buf - NPE_BUF_A;
    npe_word_t in_word = (a_addr & 1) ? od[a_sel] : ev[a_sel];
    if (d.interleave && first_in)
      b_word = (b_addr & 1) ? od[b_sel] : ev[b_sel];

    // Multiply-accumulate: P output channels x P input channels.
    npe_wword_t weights = W[d.w_base + ow * d.cin_w + iw];
    npe_qword_t quant = Q[d.q_base + ow];
    npe_word_t result; // valid in the last input cycle
  conv1x1_out_ch:
    for (int o = 0; o < NPE_P; o++) {
      npe_acc_t partial = 0;
    conv1x1_in_ch:
      for (int c = 0; c < NPE_P; c++) {
        ap_uint<8> raw = in_word.range(8 * c + 7, 8 * c);
        ap_int<9> x = d.a_signed ? (ap_int<9>)(ap_int<8>)raw : (ap_int<9>)raw;
        ap_int<8> w = weights.range(8 * (NPE_P * o + c) + 7, 8 * (NPE_P * o + c));
        partial += x * w;
      }
      npe_acc_t sum = (first_in ? (npe_acc_t)0 : acc[o]) + partial;
      acc[o] = sum;
      npe_acc_t bias = quant.range(32 * o + 31, 32 * o);
      ap_int<32> M = quant.range(32 * NPE_P + 32 * o + 31, 32 * NPE_P + 32 * o);
      ap_uint<6> S = quant.range(64 * NPE_P + 8 * o + 5, 64 * NPE_P + 8 * o);
      result.range(8 * o + 7, 8 * o) = npe_requant(sum + bias, M, S, d.lo, d.hi);
    }

    // Interleave B and the result, byte by byte, into two words.
    npe_word_t mix[2];
  conv1x1_mix:
    for (int k = 0; k < 2 * NPE_P; k++) {
      int lane = k / 2; // channel i of B (k even) or of the result (k odd)
      mix[k / NPE_P].range(8 * (k % NPE_P) + 7, 8 * (k % NPE_P)) =
          (k & 1) ? result.range(8 * lane + 7, 8 * lane)
                  : b_word.range(8 * lane + 7, 8 * lane);
    }

    // Write: one word (plain) or two neighbouring words (interleave).
    int o_addr = o_pixel + (d.interleave ? 2 * ow : ow);
    npe_word_t v_first = d.interleave ? mix[0] : result;
    bool first_odd = o_addr & 1;
    // bank 0 takes whichever of the two words has an even address
    bool en0 = last_in && (!first_odd || d.interleave);
    bool en1 = last_in && (first_odd || d.interleave);
    int row0 = first_odd ? (o_addr + 1) >> 1 : o_addr >> 1;
    int row1 = o_addr >> 1;
    npe_word_t v0 = first_odd ? mix[1] : v_first;
    npe_word_t v1 = first_odd ? v_first : mix[1];
    npe_bank_write(A, en0 && d.o_buf == NPE_BUF_A, row0, v0,
                   en1 && d.o_buf == NPE_BUF_A, row1, v1);
    npe_bank_write(B, en0 && d.o_buf == NPE_BUF_B, row0, v0,
                   en1 && d.o_buf == NPE_BUF_B, row1, v1);
    npe_bank_write(C, en0 && d.o_buf == NPE_BUF_C, row0, v0,
                   en1 && d.o_buf == NPE_BUF_C, row1, v1);

    // Advance: input word, then output word (descending), then pixel.
    if (last_in) {
      iw = 0;
      if (ow == 0) {
        ow = d.cout_w - 1;
        b_pixel += d.b_step;
        o_pixel += d.o_step;
        if (ox == d.out_w - 1) { // next output row
          ox = 0;
          oy++;
          a_row += d.stride_h * d.a_w * d.a_step;
          a_pixel = a_row;
        } else {
          ox++;
          a_pixel += d.stride_w * d.a_step;
        }
      } else {
        ow--;
      }
    } else {
      iw++;
    }
  }
}

#endif // NPE_CONV1X1_H_
