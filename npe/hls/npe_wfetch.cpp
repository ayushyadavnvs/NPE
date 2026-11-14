#include "npe_wfetch.h"

void npe_wfetch(const npe_fetch_beat_t *wmem, int n_beats,
                hls::stream<npe_fetch_beat_t> &w_out) {
  // 4 KB bursts (256 beats, the AXI limit), many in flight to hide the DDR
  // latency.
#pragma HLS INTERFACE mode = m_axi port = wmem offset = slave bundle = gmem \
    max_read_burst_length = 256 num_read_outstanding = 16 depth = 65536
#pragma HLS INTERFACE mode = axis port = w_out
#pragma HLS INTERFACE mode = s_axilite port = wmem bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = n_beats bundle = CTRL
#pragma HLS INTERFACE mode = s_axilite port = return bundle = CTRL
fetch_loop:
  for (int i = 0; i < n_beats; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 200000
    w_out.write(wmem[i]);
  }
}
