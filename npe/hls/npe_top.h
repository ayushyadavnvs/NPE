#ifndef NPE_TOP_H_
#define NPE_TOP_H_

#include <ap_axi_sdata.h>

#include "npe_engine.h"

// Top level of the engine. All data moves over AXI-Stream, so any master
// (DMA, camera pipeline, the PS) can drive it; control is AXI-Lite.
//
//   load_in  64-bit beats that fill one memory (mode NPE_LOAD_*). Each memory
//            word is sent low bits first, padded to a whole number of beats.
//   img_in   the input image for mode NPE_RUN: one pixel per beat, channel k
//            in bits [8k + 7 : 8k]. Exactly NPE_IN_PIXELS beats; TUSER and
//            TLAST are ignored, so any framing works.
//   out      the int32 results (logits) of mode NPE_RUN, TLAST on the last.
//
//   mode     what this call does (below); count: words to load, or number of
//            instructions to run; n_out: results to send in NPE_RUN.
enum { NPE_RUN = 0, NPE_LOAD_PROG = 1, NPE_LOAD_W = 2, NPE_LOAD_WD = 3, NPE_LOAD_Q = 4 };

#define NPE_IN_BEAT_BITS (8 * ((NPE_IN_C + 3) / 4) * 4) // whole bytes, >= 32
typedef ap_axiu<64, 1, 1, 1> npe_load_beat_t;
typedef ap_axiu<NPE_IN_BEAT_BITS, 1, 1, 1> npe_img_beat_t;
typedef ap_axiu<32, 1, 1, 1> npe_out_beat_t;

typedef ap_uint<16 * NPE_INSTR_FIELDS> npe_pword_t; // one packed instruction

void npe_top(hls::stream<npe_load_beat_t> &load_in,
             hls::stream<npe_img_beat_t> &img_in,
             hls::stream<npe_out_beat_t> &out, int mode, int count, int n_out);

#endif // NPE_TOP_H_
