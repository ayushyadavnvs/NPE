#include "npe_top.h"

// BIND_STORAGE with the technology from a macro of npe_config.h.
#define NPE_PRAGMA(x) _Pragma(#x)
#define NPE_BIND(var, kind, tech)                                              \
  NPE_PRAGMA(HLS BIND_STORAGE variable = var type = kind impl = tech)

// Packed instruction word -> instruction struct (field i = bits 16i..16i+15).
static npe_instr_t npe_unpack(const npe_pword_t &p) {
#pragma HLS INLINE
  npe_instr_t I;
#define NPE_UNPACK_FIELD(name, i) I.name = p.range(16 * (i) + 15, 16 * (i));
  NPE_INSTR_FIELD_LIST(NPE_UNPACK_FIELD)
#undef NPE_UNPACK_FIELD
  return I;
}

// Fills `count` words of `mem` from 64-bit beats, low beat first. WBITS is
// the word width; a word takes ceil(WBITS / 64) beats.
template <int WBITS, int N>
static void npe_load(hls::stream<npe_load_beat_t> &in, ap_uint<WBITS> mem[N],
                     int count) {
#pragma HLS INLINE off
  const int BEATS = (WBITS + 63) / 64;
  ap_uint<64 * BEATS> word = 0;
load_loop:
  for (int i = 0; i < count * BEATS; i++) {
#pragma HLS PIPELINE II = 1
    word = (word >> 64) | ((ap_uint<64 * BEATS>)in.read().data << (64 * BEATS - 64));
    if (i % BEATS == BEATS - 1)
      mem[i / BEATS] = word.range(WBITS - 1, 0);
  }
}

void npe_top(hls::stream<npe_load_beat_t> &load_in,
             hls::stream<npe_img_beat_t> &img_in,
             hls::stream<npe_out_beat_t> &out, int mode, int count, int n_out) {
#pragma HLS INTERFACE mode = axis port = load_in
#pragma HLS INTERFACE mode = axis port = img_in
#pragma HLS INTERFACE mode = axis port = out
#pragma HLS INTERFACE mode = s_axilite port = mode bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = count bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = n_out bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = return bundle = CTRL

  // On-chip state, kept between calls: program, parameters, feature buffers.
  static npe_pword_t PROG[NPE_PROG_WORDS];
  static npe_wword_t W[NPE_W_WORDS];
  static npe_dwword_t WD[NPE_WD_WORDS];
  static npe_qword_t Q[NPE_Q_WORDS];
  static npe_qdword_t QD[NPE_QD_WORDS];
  static npe_pixel_t IN[NPE_IN_PPB][NPE_IN_ROWS];
#pragma HLS ARRAY_PARTITION variable = IN dim = 1 complete
  static npe_word_t S0[NPE_NB][NPE_ROWS(NPE_BUF_S0_WORDS)];
  static npe_word_t A[NPE_NB][NPE_ROWS(NPE_BUF_A_WORDS)];
  static npe_word_t B[NPE_NB][NPE_ROWS(NPE_BUF_B_WORDS)];
  static npe_word_t C[NPE_NB][NPE_ROWS(NPE_BUF_C_WORDS)];
  static npe_vword_t V[NPE_V_WORDS];
  // NB banks per feature buffer, each a true dual-port RAM (see npe_types.h).
#pragma HLS ARRAY_PARTITION variable = S0 dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = A dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = B dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = C dim = 1 complete
  // Memory technology (UltraRAM, block RAM or LUTRAM) of each memory, chosen
  // per build by npe/compiler/memplan.py so the build fits its board.
  NPE_BIND(S0, ram_t2p, NPE_IMPL_S0)
  NPE_BIND(A, ram_t2p, NPE_IMPL_A)
  NPE_BIND(B, ram_t2p, NPE_IMPL_B)
  NPE_BIND(C, ram_t2p, NPE_IMPL_C)
  NPE_BIND(IN, ram_2p, NPE_IMPL_IN)
  NPE_BIND(W, ram_2p, NPE_IMPL_W)
  NPE_BIND(WD, ram_2p, NPE_IMPL_WD)
  NPE_BIND(Q, ram_2p, NPE_IMPL_Q)
  NPE_BIND(QD, ram_2p, NPE_IMPL_QD)
  // Results of a run, streamed out after the last instruction.
  hls::stream<npe_acc_t> results;
#pragma HLS STREAM variable = results depth = NPE_OUT_MAX

  switch (mode) {
  case NPE_LOAD_PROG:
    npe_load<16 * NPE_INSTR_FIELDS, NPE_PROG_WORDS>(load_in, PROG, count);
    return;
  case NPE_LOAD_W:
    npe_load<npe_wword_t::width, NPE_W_WORDS>(load_in, W, count);
    return;
  case NPE_LOAD_WD:
    npe_load<npe_dwword_t::width, NPE_WD_WORDS>(load_in, WD, count);
    return;
  case NPE_LOAD_Q:
    npe_load<npe_qword_t::width, NPE_Q_WORDS>(load_in, Q, count);
    return;
  case NPE_LOAD_QD:
    npe_load<npe_qdword_t::width, NPE_QD_WORDS>(load_in, QD, count);
    return;
  default:
    break;
  }

  // NPE_RUN: image in, program, results out.
read_image:
  for (int i = 0; i < (NPE_IN_PIXELS + NPE_IN_PPB - 1) / NPE_IN_PPB; i++) {
#pragma HLS PIPELINE II = 1
    npe_img_beat_t beat = img_in.read();
    for (int b = 0; b < NPE_IN_PPB; b++)
      IN[b][i] = beat.data.range(8 * NPE_IN_C * (b + 1) - 1, 8 * NPE_IN_C * b);
  }
run_program:
  for (int pc = 0; pc < count; pc++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64 avg = 59
    npe_instr_t instr = npe_unpack(PROG[pc]);
    npe_step(instr, IN, S0, A, B, C, V, results, W, WD, Q, QD);
  }
write_results:
  for (int k = 0; k < n_out; k++) {
#pragma HLS PIPELINE II = 1
    npe_out_beat_t beat;
    beat.data = (ap_uint<32>)results.read();
    beat.keep = -1;
    beat.strb = -1;
    beat.user = 0;
    beat.id = 0;
    beat.dest = 0;
    beat.last = (k == n_out - 1);
    out.write(beat);
  }
}
