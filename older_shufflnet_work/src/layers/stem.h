#ifndef BD_LAYERS_STEM_H_
#define BD_LAYERS_STEM_H_

#include "../model_config.h"
#include "../types.h"
#include "../weights/stem_weights.h"
#include "requant.h"

// Stem stream (PLAN.md phase 2):
//   rgb pixels 224x224 -> conv1 3x3 s2 p1 3->24 + ReLU -> 112x112x24
//                      -> maxpool 3x3 s2 p1 -> 56x56x24
// Everything streams through two line buffers per 3x3 stage; nothing is
// stored as a full feature map.

typedef ap_uint<24> rgb_pix_t;                // R [7:0], G [15:8], B [23:16]
typedef ap_uint<8 * STEM_COUT> stem_pix_t;    // channel c in [8c+7:8c]
typedef ap_uint<24 * 9> stem_win_t;           // tap t = ky*3+kx in [24t+23:24t]

// Shared 3x3 stride-2 window step with padding 1. Call once per input pixel
// (y, x), in raster order. Only top and left padding is ever needed because
// the input size is even (the last window ends on the last row/column).
// Returns true when (y, x) completes the window of output pixel
// ((y - 1) / 2, (x - 1) / 2), i.e. on odd rows and odd columns.
//   line_buf  the two previous input rows (row y-2 in [0], row y-1 in [1])
//   window    the current 3x3 window, [row][col], col 2 is the newest
template <int WIDTH, typename T>
inline bool s2_window_step(T line_buf[2][WIDTH], T window[3][3], T px, int y,
                           int x, T pad) {
#pragma HLS INLINE
  T top = (y < 2) ? pad : line_buf[0][x];
  T mid = line_buf[1][x];
  line_buf[0][x] = mid;
  line_buf[1][x] = px;
win_shift:
  for (int r = 0; r < 3; r++) {
    window[r][0] = window[r][1];
    window[r][1] = window[r][2];
  }
  window[0][2] = top;
  window[1][2] = mid;
  window[2][2] = px;
  if (x == 0) {
    // Column -1 of the next window is padding.
  win_left_pad:
    for (int r = 0; r < 3; r++)
      window[r][1] = pad;
  }
  return (y & 1) && (x & 1);
}

// Builds the conv1 windows from the RGB pixel stream at one pixel per cycle.
// Emits STEM_SIZE * STEM_SIZE windows of 27 bytes.
inline void stem_conv1_window(hls::stream<rgb_t> &in,
                              hls::stream<stem_win_t> &out) {
  rgb_pix_t line_buf[2][IMG_SIZE];
#pragma HLS ARRAY_PARTITION variable = line_buf dim = 1 complete
#pragma HLS DEPENDENCE variable = line_buf inter false
  rgb_pix_t window[3][3];
#pragma HLS ARRAY_PARTITION variable = window complete dim = 0
  // Padding pixel: the folded input mean, so it becomes 0 after normalization.
  const rgb_pix_t pad = (rgb_pix_t)((ap_uint<24>)STEM_PAD[0] |
                                    ((ap_uint<24>)STEM_PAD[1] << 8) |
                                    ((ap_uint<24>)STEM_PAD[2] << 16));
  int x = 0, y = 0;
conv1_window_loop:
  for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
#pragma HLS PIPELINE II = 1
    rgb_t rgb = in.read();
    rgb_pix_t px = (rgb_pix_t)(((ap_uint<24>)rgb.ch[2] << 16) |
                               ((ap_uint<24>)rgb.ch[1] << 8) | rgb.ch[0]);
    if (s2_window_step<IMG_SIZE, rgb_pix_t>(line_buf, window, px, y, x, pad)) {
      stem_win_t window_bits; // 9 taps x 24 bits, packed for the engine
    conv1_pack_ky:
      for (int ky = 0; ky < 3; ky++)
      conv1_pack_kx:
        for (int kx = 0; kx < 3; kx++)
          window_bits.range(24 * (ky * 3 + kx) + 23, 24 * (ky * 3 + kx)) =
              window[ky][kx];
      out.write(window_bits);
    }
    if (++x == IMG_SIZE) {
      x = 0;
      y++;
    }
  }
}

// conv1 engine: 27 taps x STEM_PAR output channels per cycle, so
// STEM_GROUPS (6) cycles per output pixel. Requantizes exactly as
// requant_relu: clamp((acc * M + 2^(S-1)) >> S, 0, 255), acc includes bias.
inline void stem_conv1_engine(hls::stream<stem_win_t> &in,
                              hls::stream<stem_pix_t> &out) {
#pragma HLS ARRAY_PARTITION variable = STEM_W dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = STEM_BIAS dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = STEM_M dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = STEM_S dim = 2 complete
  stem_win_t window_bits = 0; // current 3x3x3 window, reused for all groups
  stem_pix_t out_px = 0;      // 24 output channels being assembled
  int group = 0;              // which STEM_PAR output channels this cycle
conv1_engine_loop:
  for (int i = 0; i < STEM_SIZE * STEM_SIZE * STEM_GROUPS; i++) {
#pragma HLS PIPELINE II = 1
    if (group == 0)
      window_bits = in.read();
    ap_uint<8 * STEM_PAR> group_out; // STEM_PAR requantized channels
  conv1_out_ch:
    for (int j = 0; j < STEM_PAR; j++) {
      acc_t acc = STEM_BIAS[group][j];
    conv1_tap:
      for (int tap = 0; tap < STEM_TAPS; tap++) {
        ap_uint<8> in_val = window_bits.range(8 * tap + 7, 8 * tap);
        acc += (ap_int<17>)(in_val * STEM_W[group][j * STEM_TAPS + tap]);
      }
      group_out.range(8 * j + 7, 8 * j) =
          requant_relu(acc, STEM_M[group][j], STEM_S[group][j]);
    }
    // Groups arrive in channel order; shift them in from the top so channel 0
    // ends up in bits [7:0] after the last group.
    out_px = (out_px >> (8 * STEM_PAR)) |
             ((stem_pix_t)group_out << (8 * (STEM_COUT - STEM_PAR)));
    if (group == STEM_GROUPS - 1) {
      out.write(out_px);
      group = 0;
    } else {
      group++;
    }
  }
}

// Streaming maxpool 3x3 s2 p1 on the 112x112x24 conv1 output, one input
// pixel per cycle (II=1). Inputs are post-ReLU, so pad 0 equals -inf.
inline void stem_maxpool(hls::stream<stem_pix_t> &in,
                         hls::stream<stem_pix_t> &out) {
  stem_pix_t line_buf[2][STEM_SIZE];
#pragma HLS ARRAY_PARTITION variable = line_buf dim = 1 complete
#pragma HLS DEPENDENCE variable = line_buf inter false
  stem_pix_t window[3][3];
#pragma HLS ARRAY_PARTITION variable = window complete dim = 0
  int x = 0, y = 0;
maxpool_loop:
  for (int i = 0; i < STEM_SIZE * STEM_SIZE; i++) {
#pragma HLS PIPELINE II = 1
    stem_pix_t px = in.read();
    if (s2_window_step<STEM_SIZE, stem_pix_t>(line_buf, window, px, y, x, 0)) {
      stem_pix_t pooled;
    maxpool_channel:
      for (int ch = 0; ch < STEM_COUT; ch++) {
        ap_uint<8> max_val = 0;
      maxpool_ky:
        for (int ky = 0; ky < 3; ky++)
        maxpool_kx:
          for (int kx = 0; kx < 3; kx++) {
            ap_uint<8> val = window[ky][kx].range(8 * ch + 7, 8 * ch);
            if (val > max_val)
              max_val = val;
          }
        pooled.range(8 * ch + 7, 8 * ch) = max_val;
      }
      out.write(pooled);
    }
    if (++x == STEM_SIZE) {
      x = 0;
      y++;
    }
  }
}

#endif // BD_LAYERS_STEM_H_
