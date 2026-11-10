#ifndef BD_LAYERS_IO_H_
#define BD_LAYERS_IO_H_

#include "../model_config.h"
#include "../types.h"

// Reads one IMG_SIZE x IMG_SIZE frame from the video stream and emits one RGB
// pixel per cycle. Normalization (mean/std) is folded into conv1 offline, so
// raw 0..255 values pass through unchanged.
template <int SIZE>
void read_frame(hls::stream<axis_pixel_t> &in, hls::stream<rgb_t> &out) {
read_frame_loop:
  for (int i = 0; i < SIZE * SIZE; i++) {
#pragma HLS PIPELINE II = 1
    axis_pixel_t beat = in.read();
    rgb_t px;
    px.ch[0] = beat.data.range(7, 0);
    px.ch[1] = beat.data.range(15, 8);
    px.ch[2] = beat.data.range(23, 16);
    out.write(px);
  }
}

// Publishes the three class logits (in class order: other, bird, drone) and a
// frame counter to AXI-Lite registers.
// frame_no is counted by the caller (the top function resets it on a weight
// load), so the counter is not hidden in a static here.
inline void write_result(hls::stream<logit_t> &logits, ap_uint<32> frame_no,
                         logit_t &logit_other, logit_t &logit_bird,
                         logit_t &logit_drone, ap_uint<32> &frame_count) {
#pragma HLS INLINE off
  logit_other = logits.read();
  logit_bird = logits.read();
  logit_drone = logits.read();
  frame_count = frame_no;
}

#endif // BD_LAYERS_IO_H_
