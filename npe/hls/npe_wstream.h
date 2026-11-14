#ifndef NPE_WSTREAM_H_
#define NPE_WSTREAM_H_

#include "npe_decode.h"

// Weight streaming. Layers too big to keep on chip (w_stream = 1) read their
// weights from a ring of NPE_W_RING words at the end of W. The words come
// from DRAM on the WS stream (npe_wfetch.cpp reads them with long AXI
// bursts) in exactly the order the engine uses them: streamed layers in
// program order, each layer's output groups from the highest down, each
// group's w_per_out words in address order. Stream word s goes to ring slot
// s % NPE_W_RING.
//
// While it computes, the engine takes in at most one beat per cycle
// (npe_ws_take) as long as the ring has room: a word may be written only
// while every word still needed is less than NPE_W_RING words behind it.
// `keep` is the first stream word still needed: the start of the group a
// streamed conv is working on, or else the start of the next streamed layer.
// A streamed conv waits at the start of each group until all of the group's
// words are in; then it reuses them for every pixel. So the next group's
// words arrive while the current group computes, and a group must fit in
// the ring (checked by the packer).
inline void npe_ws_take(hls::stream<npe_ws_t> &WS, npe_wring_t &R,
                        npe_wword_t W[NPE_W_WORDS], int keep) {
#pragma HLS INLINE
  if (NPE_W_RING == 0)
    return;
  const int BEATS = npe_wword_t::width / NPE_WS_BITS;
  if (R.wr - keep < NPE_W_RING && !WS.empty()) {
    npe_ws_t b = WS.read();
    npe_wword_t part = (R.part >> NPE_WS_BITS) |
                       ((npe_wword_t)b << (npe_wword_t::width - NPE_WS_BITS));
    if (R.nbeat == BEATS - 1) {
      W[NPE_W_RING_BASE + (R.wr & (NPE_W_RING - 1))] = part;
      R.wr++;
      R.nbeat = 0;
    } else {
      R.nbeat++;
    }
    R.part = part;
  }
}

// Start of a run: nothing taken yet.
inline void npe_ws_reset(npe_wring_t &R) {
  R.wr = 0;
  R.pos = 0;
  R.nbeat = 0;
  R.part = 0;
}

#endif // NPE_WSTREAM_H_
