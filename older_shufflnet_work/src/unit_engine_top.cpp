#include "unit_engine_top.h"

void unit_engine_top(hls::stream<ap_uint<64> > &weights_in,
                     hls::stream<stem_pix_t> &feat_in,
                     hls::stream<uword_t> &feat_out, bool load_weights,
                     int n_layers) {
#pragma HLS INTERFACE mode = axis port = weights_in
#pragma HLS INTERFACE mode = axis port = feat_in
#pragma HLS INTERFACE mode = axis port = feat_out
#pragma HLS INTERFACE mode = s_axilite port = load_weights bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = n_layers bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = return bundle = CTRL

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
  static pw_wword_t PW_W[UNIT_PW_WORDS];
  static dw_wword_t DW_W[UNIT_DW_WORDS];
  static pw_qword_t PW_Q[UNIT_QP_WORDS];
  static dw_qword_t DW_Q[UNIT_QD_WORDS];
  static fc_wword_t FC_W[UNIT_FC_WORDS];
  static fc_bword_t FC_B;
#pragma HLS BIND_STORAGE variable = PW_W type = ram_2p impl = uram

  if (load_weights) {
    unit_load_weights(weights_in, UNIT_W_ARGS);
    return;
  }

  // Maxpool map in: 3 words per 24-channel pixel.
  stem_pix_t px = 0;
load_pool_map:
  for (int i = 0; i < POOL_SIZE * POOL_SIZE * 3; i++) {
#pragma HLS PIPELINE II = 1
    int part = i % 3; // which 64-bit third of the pixel
    if (part == 0)
      px = feat_in.read();
    X[i] = px.range(64 * part + 63, 64 * part);
  }

  // Up to conv5; the head's logits are checked through bird_drone_accel.
  if (n_layers < 1 || n_layers > NET_NUM_LAYERS - 1)
    n_layers = UNIT_NUM_LAYERS;
  logit_t logits[NUM_CLASSES];
  unit_run(n_layers, X, T1, T2, B1, B2, UNIT_W_ARGS, logits);

  // Stream out the last layer's output buffer.
  unit_layer_t last = UNIT_PROGRAM[n_layers - 1];
  int out_size = (last.kind == UL_DW && last.stride == 2) ? last.size / 2
                                                          : last.size;
  int out_words = out_size * out_size * last.cout_g;
stream_out:
  for (int i = 0; i < out_words; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 588 max = 9408 avg = 1176
    uword_t word;
    switch (last.dst) {
    case BUF_X: word = X[i]; break;
    case BUF_T1: word = T1[i]; break;
    case BUF_T2: word = T2[i]; break;
    case BUF_B1: word = B1[i]; break;
    default: word = B2[i]; break;
    }
    feat_out.write(word);
  }
}
