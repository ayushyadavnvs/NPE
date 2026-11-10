#ifndef BD_LAYERS_PLACEHOLDER_H_
#define BD_LAYERS_PLACEHOLDER_H_

#include "../model_config.h"
#include "../types.h"

// PLACEHOLDER for phase 0 only. It stands where the stem, shuffle-unit engine
// and conv5 will go, so the project builds end to end (csim, csynth, cosim,
// export) on the KV260 part before any real layer exists. It computes the
// per-channel mean of the frame and a tiny 3->2 linear head. Delete it once
// the real pipeline from PLAN.md phases 2-4 is wired in.

static const wgt_t PLACEHOLDER_W[NUM_CLASSES][IMG_CHANNELS] = {{1, -1, 0},
                                                               {-1, 1, 0}};
static const bias_t PLACEHOLDER_B[NUM_CLASSES] = {0, 0};

template <int SIZE>
void placeholder_channel_mean(hls::stream<rgb_t> &in,
                              hls::stream<act_t> &out) {
  acc_t sum[IMG_CHANNELS] = {0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = sum complete
  for (int i = 0; i < SIZE * SIZE; i++) {
#pragma HLS PIPELINE II = 1
    rgb_t px = in.read();
    for (int c = 0; c < IMG_CHANNELS; c++) {
#pragma HLS UNROLL
      sum[c] += px.ch[c];
    }
  }
  for (int c = 0; c < IMG_CHANNELS; c++) {
#pragma HLS PIPELINE II = 1
    out.write((act_t)(sum[c] / (SIZE * SIZE)));
  }
}

inline void placeholder_head(hls::stream<act_t> &in,
                             hls::stream<logit_t> &out) {
#pragma HLS INLINE off
  act_t feat[IMG_CHANNELS];
  for (int c = 0; c < IMG_CHANNELS; c++)
    feat[c] = in.read();
  for (int k = 0; k < NUM_CLASSES; k++) {
    acc_t acc = PLACEHOLDER_B[k];
    for (int c = 0; c < IMG_CHANNELS; c++)
      acc += PLACEHOLDER_W[k][c] * feat[c];
    out.write(acc);
  }
}

#endif // BD_LAYERS_PLACEHOLDER_H_
