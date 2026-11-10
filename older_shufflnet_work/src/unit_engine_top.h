#ifndef BD_UNIT_ENGINE_TOP_H_
#define BD_UNIT_ENGINE_TOP_H_

#include "layers/stem.h"
#include "layers/units.h"

// Phase 3 test top for the shuffle-unit engine.
//   load_weights = 1: read UNIT_LOAD_WORDS words of data/unit_weights.bin from
//                     weights_in into the on-chip weight memories, then return.
//   load_weights = 0: read one 56x56 maxpool map (stem_top's output format)
//                     from feat_in, run program layers [0, n_layers) and
//                     stream out the buffer written by the last of them (HWC
//                     8-channel words). n_layers = UNIT_NUM_LAYERS gives
//                     s4u3.out, 7x7x192 = 1176 words; UNIT_NUM_LAYERS + 1
//                     gives conv5, 7x7x1024. Anything else runs the units.
void unit_engine_top(hls::stream<ap_uint<64> > &weights_in,
                     hls::stream<stem_pix_t> &feat_in,
                     hls::stream<uword_t> &feat_out, bool load_weights,
                     int n_layers);

#endif // BD_UNIT_ENGINE_TOP_H_
