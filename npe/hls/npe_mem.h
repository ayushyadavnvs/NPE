#ifndef NPE_MEM_H_
#define NPE_MEM_H_

#include "npe_decode.h"

// Vector access to the banked feature buffers. Word w is in bank w % NB at
// row w / NB, so the NB words starting at any address are in NB different
// banks: each bank is read (or written) once, at its own row, and the words
// are rotated into order. One read and one write statement per bank, which
// is what keeps every unit at II = 1 on dual-port RAMs.

// v[k] = word addr + k, k < NB.
template <int N>
inline void npe_vread(const npe_word_t buf[NPE_NB][N], int addr,
                      npe_word_t v[NPE_NB]) {
#pragma HLS INLINE
  npe_word_t bank[NPE_NB];
#pragma HLS ARRAY_PARTITION variable = bank complete
vread_bank:
  for (int b = 0; b < NPE_NB; b++) {
#pragma HLS UNROLL
    int k = (b - addr) & (NPE_NB - 1); // which word of the vector bank b holds
    bank[b] = buf[b][(addr + k) >> NPE_LOG2NB];
  }
vread_rotate:
  for (int k = 0; k < NPE_NB; k++) {
#pragma HLS UNROLL
    v[k] = bank[(addr + k) & (NPE_NB - 1)];
  }
}

// Writes words addr .. addr + n - 1 (n <= NB) from v[0 .. n-1] if en.
template <int N>
inline void npe_vwrite(npe_word_t buf[NPE_NB][N], bool en, int addr, int n,
                       const npe_word_t v[NPE_NB]) {
#pragma HLS INLINE
vwrite_bank:
  for (int b = 0; b < NPE_NB; b++) {
#pragma HLS UNROLL
    int k = (b - addr) & (NPE_NB - 1);
    if (en && k < n)
      buf[b][(addr + k) >> NPE_LOG2NB] = v[k];
  }
}

// Address a buffer is read at: a_addr if it holds operand A, b_addr if it
// holds operand B, else 0 (an unused buffer must not be read out of range).
inline int npe_sel_addr(const npe_dec_t &d, int buf, int a_addr, int b_addr) {
#pragma HLS INLINE
  return d.a_buf == buf ? a_addr : (d.use_b && d.b_buf == buf) ? b_addr : 0;
}

// Reads the A operand vector (at a_addr) and the B operand vector (at
// b_addr) in one cycle. A and B are in different buffers (the packer checks
// that), so each buffer is read once.
inline void npe_read_ab(const npe_dec_t &d, const npe_word_t S0[NPE_NB][NPE_ROWS(NPE_BUF_S0_WORDS)],
                        const npe_word_t A[NPE_NB][NPE_ROWS(NPE_BUF_A_WORDS)],
                        const npe_word_t B[NPE_NB][NPE_ROWS(NPE_BUF_B_WORDS)],
                        const npe_word_t C[NPE_NB][NPE_ROWS(NPE_BUF_C_WORDS)],
                        int a_addr, int b_addr, npe_word_t av[NPE_NB],
                        npe_word_t bv[NPE_NB]) {
#pragma HLS INLINE
  npe_word_t v[4][NPE_NB];
#pragma HLS ARRAY_PARTITION variable = v complete dim = 0
  npe_vread(S0, npe_sel_addr(d, NPE_BUF_S0, a_addr, b_addr), v[0]);
  npe_vread(A, npe_sel_addr(d, NPE_BUF_A, a_addr, b_addr), v[1]);
  npe_vread(B, npe_sel_addr(d, NPE_BUF_B, a_addr, b_addr), v[2]);
  npe_vread(C, npe_sel_addr(d, NPE_BUF_C, a_addr, b_addr), v[3]);
  int a_sel = (d.a_buf - NPE_BUF_S0) & 3, b_sel = (d.b_buf - NPE_BUF_S0) & 3;
read_ab_sel:
  for (int k = 0; k < NPE_NB; k++) {
#pragma HLS UNROLL
    av[k] = v[a_sel][k];
    bv[k] = v[b_sel][k];
  }
}

// Writes n words to the output buffer o_buf.
inline void npe_write_o(const npe_dec_t &d, npe_word_t S0[NPE_NB][NPE_ROWS(NPE_BUF_S0_WORDS)],
                        npe_word_t A[NPE_NB][NPE_ROWS(NPE_BUF_A_WORDS)],
                        npe_word_t B[NPE_NB][NPE_ROWS(NPE_BUF_B_WORDS)],
                        npe_word_t C[NPE_NB][NPE_ROWS(NPE_BUF_C_WORDS)], bool en,
                        int addr, int n, const npe_word_t v[NPE_NB]) {
#pragma HLS INLINE
  npe_vwrite(S0, en && d.o_buf == NPE_BUF_S0, addr, n, v);
  npe_vwrite(A, en && d.o_buf == NPE_BUF_A, addr, n, v);
  npe_vwrite(B, en && d.o_buf == NPE_BUF_B, addr, n, v);
  npe_vwrite(C, en && d.o_buf == NPE_BUF_C, addr, n, v);
}

// out = clamp((acc * M + extra + 2^(S-1)) >> S, lo, hi). Same rule as the
// reference (npe/ref/isa_ref.py). extra is b * Mb for the ADD epilogue, else
// 0. The result is returned as a raw byte: for a signed output it is the
// two's complement int8.
inline ap_uint<8> npe_requant(npe_acc_t acc, ap_int<32> M, ap_uint<6> S, int lo,
                              int hi, ap_int<48> extra = 0) {
#pragma HLS INLINE
  ap_int<64> prod = (ap_int<64>)acc * M + extra;
  ap_int<64> rounded = (prod + ((ap_int<64>)1 << (S - 1))) >> S;
  if (rounded < lo)
    rounded = lo;
  if (rounded > hi)
    rounded = hi;
  return (ap_uint<8>)rounded;
}

// Lane `lane` of an input byte vector as a 9-bit signed value.
inline ap_int<9> npe_lane(const npe_word_t &w, int lane, bool is_signed) {
#pragma HLS INLINE
  ap_uint<8> raw = w.range(8 * lane + 7, 8 * lane);
  return is_signed ? (ap_int<9>)(ap_int<8>)raw : (ap_int<9>)raw;
}

#endif // NPE_MEM_H_
