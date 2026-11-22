#ifndef NPE_WFETCH_H_
#define NPE_WFETCH_H_

#include <ap_int.h>
#include <hls_stream.h>

// Weight fetcher: reads n_beats 128-bit beats from DRAM at `wmem` (the
// streamed weights file written by npe/compiler/build.py) and streams them to
// the engine's w_in port, one inference per call. A separate IP, so the
// engine keeps plain AXI-Stream ports: any master may feed w_in instead.
// The host writes the buffer address and n_beats once and sets the
// auto-restart bit; then it reruns by itself, frame after frame, held back
// only by the engine taking the stream (back-pressure).
typedef ap_uint<128> npe_fetch_beat_t;

void npe_wfetch(const npe_fetch_beat_t *wmem, int n_beats,
                hls::stream<npe_fetch_beat_t> &w_out);

#endif // NPE_WFETCH_H_
