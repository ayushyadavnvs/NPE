#pragma once
#include <ap_int.h>
template <int D, int U, int TI, int TD> struct ap_axiu {
  ap_uint<D> data; ap_uint<(D + 7) / 8> keep, strb; ap_uint<U> user; ap_uint<1> last; ap_uint<TI> id; ap_uint<TD> dest;
};
