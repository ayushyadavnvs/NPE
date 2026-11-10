#ifndef BD_LAYERS_REQUANT_H_
#define BD_LAYERS_REQUANT_H_

#include "../types.h"

// Scales a 32-bit accumulator back to an unsigned INT8 activation with fused
// ReLU. Rounds half up; the software reference must use the same rule so that
// C simulation stays bit-exact against the model team's golden vectors.
// The quantization flow must export shift >= 1.
inline act_t requant_relu(acc_t acc, rq_mult_t mult, rq_shift_t shift) {
#pragma HLS INLINE
  ap_int<64> prod = (ap_int<64>)acc * mult;
  ap_int<64> rounded = (prod + ((ap_int<64>)1 << (shift - 1))) >> shift;
  if (rounded < 0)
    return 0;
  if (rounded > 255)
    return 255;
  return (act_t)rounded;
}

// Same scaling without ReLU, for the depthwise convolutions, which
// ShuffleNetV2 follows with BatchNorm only.
inline ap_int<8> requant_linear(acc_t acc, rq_mult_t mult, rq_shift_t shift) {
#pragma HLS INLINE
  ap_int<64> prod = (ap_int<64>)acc * mult;
  ap_int<64> rounded = (prod + ((ap_int<64>)1 << (shift - 1))) >> shift;
  if (rounded < -128)
    return -128;
  if (rounded > 127)
    return 127;
  return (ap_int<8>)rounded;
}

#endif // BD_LAYERS_REQUANT_H_
