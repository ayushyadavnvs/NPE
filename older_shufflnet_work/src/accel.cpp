#include "accel.h"
#include "layers/io.h"
#include "layers/requant.h"
#include "layers/stem.h"
#include "layers/units.h"

// Network engine process: takes the 56x56x24 maxpool map, runs the 16 units,
// conv5 and the head from the layer program, and emits the NUM_CLASSES logits
// in class order.
static void net_engine(hls::stream<stem_pix_t> &pool_in, UNIT_W_PARAMS,
                       hls::stream<logit_t> &logits_out) {
  static uword_t X[UNIT_BUF_X_WORDS];
  static uword_t T1[UNIT_BUF_T1_WORDS];
  static uword_t T2[UNIT_BUF_T2_WORDS];
  static uword_t B1[UNIT_BUF_B1_WORDS];
  static uword_t B2[UNIT_BUF_B2_WORDS];
#pragma HLS BIND_STORAGE variable = X type = ram_t2p impl = uram
  // Merge-on-write in unit_pw stores four X words per output pair: two
  // banks (even / odd words) of true dual-port memory take them at II = 1.
#pragma HLS ARRAY_PARTITION variable = X type = cyclic factor = 2
#pragma HLS BIND_STORAGE variable = T1 type = ram_t2p impl = uram
  // The pointwise engine writes two words per cycle into T1 or B1.
#pragma HLS BIND_STORAGE variable = B1 type = ram_t2p impl = bram
#pragma HLS BIND_STORAGE variable = B2 type = ram_t2p impl = bram

  // Each 24-channel maxpool pixel (192 bits) becomes 3 words of 8 channels.
  stem_pix_t px = 0;
load_pool_map:
  for (int i = 0; i < POOL_SIZE * POOL_SIZE * 3; i++) {
#pragma HLS PIPELINE II = 1
    int part = i % 3; // which 64-bit third of the pixel
    if (part == 0)
      px = pool_in.read();
    X[i] = px.range(64 * part + 63, 64 * part);
  }
  logit_t logits[NUM_CLASSES]; // [0] other, [1] bird, [2] drone
  unit_run(NET_NUM_LAYERS, X, T1, T2, B1, B2, UNIT_W_ARGS, logits);
write_logits:
  for (int k = 0; k < NUM_CLASSES; k++)
    logits_out.write(logits[k]);
}

// One frame: stem stream and network engine in a dataflow pipeline. The
// engine's input loop takes the maxpool map while the stem produces it, so
// stem and engine overlap within a frame, but each call returns only when the
// frame is done (ap_ctrl_hs; the dataflow region is not the top), so frames do
// not overlap: interval ~ stem (75k) + engine compute. A small pool FIFO is
// therefore enough (depth 64, like the conv1 window FIFO).
static void accel_frame(hls::stream<axis_pixel_t> &pixels_in, UNIT_W_PARAMS,
                        ap_uint<32> frame_no, logit_t &logit_other,
                        logit_t &logit_bird, logit_t &logit_drone,
                        ap_uint<32> &frame_count) {
#pragma HLS DATAFLOW
  hls::stream<rgb_t> pixel_stream("pixel_stream");
  hls::stream<stem_win_t> window_stream("window_stream");
  hls::stream<stem_pix_t> conv1_stream("conv1_stream");
  hls::stream<stem_pix_t> pool_stream("pool_stream");
  hls::stream<logit_t> logit_stream("logit_stream");
#pragma HLS STREAM variable = pixel_stream depth = 2
  // conv1 windows come in bursts on odd rows; see stem_top.cpp.
#pragma HLS STREAM variable = window_stream depth = 64
#pragma HLS STREAM variable = conv1_stream depth = 2
#pragma HLS STREAM variable = pool_stream depth = 64
#pragma HLS STREAM variable = logit_stream depth = 4 // >= NUM_CLASSES

  read_frame<IMG_SIZE>(pixels_in, pixel_stream);
  stem_conv1_window(pixel_stream, window_stream);
  stem_conv1_engine(window_stream, conv1_stream);
  stem_maxpool(conv1_stream, pool_stream);
  net_engine(pool_stream, UNIT_W_ARGS, logit_stream);
  write_result(logit_stream, frame_no, logit_other, logit_bird, logit_drone,
               frame_count);
}

void bird_drone_accel(hls::stream<axis_pixel_t> &pixels_in,
                      hls::stream<ap_uint<64> > &weights_in, bool load_weights,
                      logit_t &logit_other, logit_t &logit_bird,
                      logit_t &logit_drone, ap_uint<32> &frame_count) {
#pragma HLS INTERFACE mode = axis port = pixels_in
#pragma HLS INTERFACE mode = axis port = weights_in
#pragma HLS INTERFACE mode = s_axilite port = load_weights bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = logit_other bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = logit_bird bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = logit_drone bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = frame_count bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = return bundle = CTRL

  static pw_wword_t PW_W[UNIT_PW_WORDS];
  static dw_wword_t DW_W[UNIT_DW_WORDS];
  static pw_qword_t PW_Q[UNIT_QP_WORDS];
  static dw_qword_t DW_Q[UNIT_QD_WORDS];
  static fc_wword_t FC_W[UNIT_FC_WORDS];
  static fc_bword_t FC_B;
#pragma HLS BIND_STORAGE variable = PW_W type = ram_2p impl = uram
  static ap_uint<32> frames = 0; // frames classified since the last weight load

  // Every call writes all four result registers (the host and the cosim
  // testbench wait for their valid bits). A weight load clears them and
  // restarts the frame count.
  if (load_weights) {
    unit_load_weights(weights_in, UNIT_W_ARGS);
    frames = 0;
    logit_other = 0;
    logit_bird = 0;
    logit_drone = 0;
    frame_count = 0;
  } else {
    frames++;
    accel_frame(pixels_in, UNIT_W_ARGS, frames, logit_other, logit_bird,
                logit_drone, frame_count);
  }
}
