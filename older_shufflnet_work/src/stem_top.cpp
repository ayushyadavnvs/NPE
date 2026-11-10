#include "stem_top.h"
#include "layers/io.h"

void stem_top(hls::stream<axis_pixel_t> &pixels_in,
              hls::stream<stem_pix_t> &pool_out) {
#pragma HLS INTERFACE mode = axis port = pixels_in
#pragma HLS INTERFACE mode = axis port = pool_out
#pragma HLS INTERFACE mode = ap_ctrl_hs port = return
#pragma HLS DATAFLOW

  hls::stream<rgb_t> pixel_stream("pixel_stream");
  hls::stream<stem_win_t> window_stream("window_stream");
  hls::stream<stem_pix_t> conv1_stream("conv1_stream");
#pragma HLS STREAM variable = pixel_stream depth = 2
  // Windows arrive in bursts on odd rows only; the FIFO lets the conv1 engine
  // keep working through the even rows (otherwise ~98k instead of ~75k cycles).
#pragma HLS STREAM variable = window_stream depth = 64
#pragma HLS STREAM variable = conv1_stream depth = 2

  read_frame<IMG_SIZE>(pixels_in, pixel_stream);
  stem_conv1_window(pixel_stream, window_stream);
  stem_conv1_engine(window_stream, conv1_stream);
  stem_maxpool(conv1_stream, pool_out);
}
