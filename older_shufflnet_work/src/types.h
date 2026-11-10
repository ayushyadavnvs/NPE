#ifndef BD_TYPES_H_
#define BD_TYPES_H_

// Wide weight words (up to 1152 bits) in the unit engine.
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 2048
#endif
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// INT8 inference: per-channel symmetric INT8 weights, unsigned INT8
// activations after ReLU, 32-bit accumulators. BatchNorm is folded into the
// weights and bias offline, so no BN multipliers exist in hardware.
typedef ap_uint<8> act_t;  // post-ReLU activation
typedef ap_int<8> wgt_t;   // convolution weight
typedef ap_int<32> acc_t;  // MAC accumulator
typedef ap_int<32> bias_t; // folded bias, in accumulator scale

// Requantization: out = clamp((acc * mult) >> shift, 0, 255), with mult and
// shift exported per output channel by the quantization flow.
typedef ap_int<32> rq_mult_t;
typedef ap_uint<6> rq_shift_t;

// Video-style AXI4-Stream input: one RGB pixel per beat in bits [23:0]
// (R in [7:0], G in [15:8], B in [23:16]); TUSER marks the first pixel of a
// frame and TLAST the last pixel of each line.
#define AXIS_WIDTH 32
typedef ap_axiu<AXIS_WIDTH, 1, 1, 1> axis_pixel_t;

struct rgb_t {
  act_t ch[3];
};

typedef ap_int<32> logit_t;

#endif // BD_TYPES_H_
