#ifndef BD_STEM_TOP_H_
#define BD_STEM_TOP_H_

#include "layers/stem.h"

// Phase 2 top: one 224x224 RGB frame in, 56x56 maxpool pixels out (24
// channels packed per beat, channel c in bits [8c+7:8c]), raster order.
void stem_top(hls::stream<axis_pixel_t> &pixels_in,
              hls::stream<stem_pix_t> &pool_out);

#endif // BD_STEM_TOP_H_
