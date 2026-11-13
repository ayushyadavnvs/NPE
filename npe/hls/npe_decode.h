#ifndef NPE_DECODE_H_
#define NPE_DECODE_H_

#include "npe_types.h"

// An instruction after decoding: everything the datapath needs, in the units
// the hardware works in (words of P channels instead of channels).
struct npe_dec_t {
  // what to run
  bool run_conv1x1;  // the only unit so far
  bool interleave;   // epilogue INTERLEAVE
  // loop bounds
  int out_h, out_w;  // output pixels
  int cin_w, cout_w; // input / output words per pixel that this layer uses
  // input A: word address of pixel (y, x) = a_first + (y * a_w + x) * a_step
  int a_buf, a_first, a_step, a_w;
  int stride_h, stride_w;
  bool a_signed;     // input bytes are int8 (else uint8)
  // operand B (interleave only), same pixel grid as the output
  int b_buf, b_first, b_step;
  // output
  int o_buf, o_first, o_step;
  // requant clamp range: includes the activation
  int lo, hi;
  // parameters
  int w_base, q_base;
};

// Channel counts and offsets are multiples of P (checked by the packer), so
// dividing by P is exact.
inline npe_dec_t npe_decode(const npe_instr_t &I) {
#pragma HLS INLINE
  npe_dec_t d;
  d.run_conv1x1 = (I.op == NPE_OP_CONV && I.kh == 1 && I.kw == 1 &&
                   I.groups == 1 && I.a_bits == 8 && !I.raw_out);
  d.interleave = (I.epi == NPE_EPI_INTERLEAVE);
  d.out_h = I.o_h;
  d.out_w = I.o_w;
  d.cin_w = I.a_c / NPE_P;
  d.cout_w = I.o_c / NPE_P;

  // A pixel stride of 0 in the instruction means "the tensor's own channels".
  int a_pix = I.a_pix ? I.a_pix : I.a_c;
  int out_c = d.interleave ? 2 * I.o_c : I.o_c; // channels written per pixel
  int o_pix = I.o_pix ? I.o_pix : out_c;
  int b_pix = I.b_pix ? I.b_pix : I.o_c;
  d.a_buf = I.a_buf;
  d.a_first = (I.a_base + I.a_off) / NPE_P;
  d.a_step = a_pix / NPE_P;
  d.a_w = I.a_w;
  d.stride_h = I.sh;
  d.stride_w = I.sw;
  d.a_signed = I.a_signed;
  d.b_buf = I.b_buf;
  d.b_first = (I.b_base + I.b_off) / NPE_P;
  d.b_step = b_pix / NPE_P;
  d.o_buf = I.o_buf;
  d.o_first = (I.o_base + I.o_off) / NPE_P;
  d.o_step = o_pix / NPE_P;

  // Output range of the type, then ReLU raises the lower bound to 0.
  int bits = I.o_bits;
  d.lo = I.o_signed ? -(1 << (bits - 1)) : 0;
  d.hi = I.o_signed ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
  if (I.act == NPE_ACT_RELU && d.lo < 0)
    d.lo = 0;
  d.w_base = I.w_base;
  d.q_base = I.q_base;
  return d;
}

#endif // NPE_DECODE_H_
