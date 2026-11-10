#ifndef BD_ACCEL_H_
#define BD_ACCEL_H_

#include "model_config.h"
#include "types.h"

// Full bird-vs-drone accelerator.
//   load_weights = 1: read the UNIT_LOAD_WORDS 64-bit words of
//                     data/unit_weights.bin from weights_in into the on-chip
//                     weight memories (once after power-up or on new weights).
//   load_weights = 0: classify one IMG_SIZE x IMG_SIZE RGB frame from
//                     pixels_in. Logits are raw INT32 accumulator values; the
//                     host takes the argmax (class 0 = other, class 1 = bird,
//                     class 2 = drone, the clean_v2 label order).
//                     frame_count counts classified frames.
// conv1 weights are compiled in (src/weights/stem_weights.h).
void bird_drone_accel(hls::stream<axis_pixel_t> &pixels_in,
                      hls::stream<ap_uint<64> > &weights_in, bool load_weights,
                      logit_t &logit_other, logit_t &logit_bird,
                      logit_t &logit_drone, ap_uint<32> &frame_count);

#endif // BD_ACCEL_H_
