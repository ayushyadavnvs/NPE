#ifndef NPE_TOP_H_
#define NPE_TOP_H_

// Wide parameter words (npe_types.h): set before anything includes ap_int.h.
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 8192
#endif
#include <ap_axi_sdata.h>

#include "npe_engine.h"

// Top level of the engine. All data moves over AXI-Stream, so any master
// (DMA, camera pipeline, the PS) can drive it; control is AXI-Lite.
//
//   load_in  64-bit beats that fill one memory (mode NPE_LOAD_*). Each memory
//            word is sent low bits first, padded to a whole number of beats.
//   img_in   the input image for mode NPE_RUN: NPE_IN_PPB pixels per beat,
//            pixel j of the beat in bits [8 * IN_C * j ...], channel k of it
//            at + 8k. Exactly ceil(NPE_IN_PIXELS / PPB) beats (the last one
//            padded); TUSER and TLAST are ignored, so any framing works.
//   out      the int32 results (logits) of mode NPE_RUN, TLAST on the last.
//   w_in     weights of the streamed layers (npe_wstream.h), NPE_WS_BITS
//            per beat, in use order; each NPE_RUN takes exactly the words of
//            one inference. Fed by npe_wfetch (an AXI master reading DRAM) or
//            any other AXI-Stream source. Unused in builds without streamed
//            layers (NPE_W_RING 0).
//
//   mode     what this call does (below); count: words to load, or number of
//            instructions to run; n_out: results to send in NPE_RUN.
enum { NPE_RUN = 0, NPE_LOAD_PROG = 1, NPE_LOAD_W = 2, NPE_LOAD_WD = 3, NPE_LOAD_Q = 4,
       NPE_LOAD_QD = 5 };

// NPE_IN_BEAT_BITS (npe_config.h): PPB pixels rounded up to a power of two
// >= 32 bits, a width any AXI DMA supports.
typedef ap_axiu<64, 1, 1, 1> npe_load_beat_t;
typedef ap_axiu<NPE_IN_BEAT_BITS, 1, 1, 1> npe_img_beat_t;
typedef ap_axiu<32, 1, 1, 1> npe_out_beat_t;

typedef ap_uint<16 * NPE_INSTR_FIELDS> npe_pword_t; // one packed instruction

void npe_top(hls::stream<npe_load_beat_t> &load_in,
             hls::stream<npe_img_beat_t> &img_in,
             hls::stream<npe_out_beat_t> &out, hls::stream<npe_ws_t> &w_in,
             int mode, int count, int n_out);

#endif // NPE_TOP_H_
