#ifndef NPE_DECODE_H_
#define NPE_DECODE_H_

#include "npe_types.h"

// An instruction after decoding: everything the datapath needs, in the units
// the hardware works in (words of P channels instead of channels).
struct npe_dec_t {
  // what to run
  bool run_conv;     // dense convolution (groups 1)
  bool run_dw;       // depthwise convolution (groups = channels)
  bool run_pool;     // max pool
  bool run_gap;      // global average pool (sum) into V
  bool run_fc;       // fully connected: V in, int32 out to OUT
  bool interleave;   // epilogue INTERLEAVE
  // loop bounds
  int out_h, out_w;  // output pixels
  int cin_w, cout_w; // input / output words per pixel that this layer uses
  int out_c;         // output channels (fc: how many results go to OUT)
  int kh, kw;        // kernel size
  // input A: word address of pixel (y, x) = a_first + (y * a_w + x) * a_step
  int a_buf, a_first, a_step, a_h, a_w;
  int stride_h, stride_w, pad_t, pad_l;
  bool pad_const;    // padding value comes from the W memory (else 0)
  bool a_signed;     // input bytes are int8 (else uint8)
  // address steps, multiplied out once here so the unit only adds
  int a_row_step;    // one output row down:   stride_h * a_w * a_step
  int a_col_step;    // one output pixel right: stride_w * a_step
  int a_line_step;   // one kernel row down:   a_w * a_step
  int w_per_out;     // weight words per output word: kh * kw * cin_w
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
  const bool conv8 = (I.op == NPE_OP_CONV && I.a_bits == 8 && !I.raw_out);
  d.run_conv = conv8 && I.groups == 1;
  d.run_dw = conv8 && I.groups > 1 && I.groups == I.a_c && I.a_c == I.o_c;
  d.run_pool = (I.op == NPE_OP_POOL_MAX);
  d.run_gap = (I.op == NPE_OP_GAP);
  d.run_fc = (I.op == NPE_OP_CONV && I.a_buf == NPE_BUF_V && I.raw_out);
  d.interleave = (I.epi == NPE_EPI_INTERLEAVE);
  d.out_h = I.o_h;
  d.out_w = I.o_w;
  // The image buffer holds one pixel per word, whatever its channel count.
  const bool from_in = (I.a_buf == NPE_BUF_IN);
  d.cin_w = from_in ? 1 : I.a_c / NPE_P;
  d.kh = I.kh;
  d.kw = I.kw;
  d.cout_w = (I.o_c + NPE_P - 1) / NPE_P; // fc: classes rounded up to words
  d.out_c = I.o_c;

  // A pixel stride of 0 in the instruction means "the tensor's own channels".
  int a_pix = I.a_pix ? I.a_pix : I.a_c;
  int out_c = d.interleave ? 2 * I.o_c : I.o_c; // channels written per pixel
  int o_pix = I.o_pix ? I.o_pix : out_c;
  int b_pix = I.b_pix ? I.b_pix : I.o_c;
  d.a_buf = I.a_buf;
  d.a_first = from_in ? 0 : (I.a_base + I.a_off) / NPE_P;
  d.a_step = from_in ? 1 : a_pix / NPE_P;
  d.a_h = I.a_h;
  d.a_w = I.a_w;
  d.stride_h = I.sh;
  d.stride_w = I.sw;
  d.pad_t = I.pad_t;
  d.pad_l = I.pad_l;
  d.pad_const = (I.pad_mode == NPE_PAD_CONST);
  d.a_row_step = I.sh * I.a_w * d.a_step;
  d.a_col_step = I.sw * d.a_step;
  d.a_line_step = I.a_w * d.a_step;
  d.w_per_out = I.kh * I.kw * d.cin_w;
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
