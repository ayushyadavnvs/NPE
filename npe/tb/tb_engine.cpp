// Runs instructions on the HLS engine and compares every feature buffer with
// the state the reference interpreter reached. The case file is written by
// npe/tests/test_engine.py:
//   u32 n_cases
//   W image (NPE_W_WORDS words), Q image (NPE_Q_WORDS words)
//   per case: the instruction (NPE_INSTR_FIELDS x u16), buffers A, B, C before
//             it, buffers A, B, C after it (one byte per channel)
#include <cstdio>
#include <cstring>
#include <vector>

#include "npe_engine.h"

static npe_word_t A[2][NPE_HALF(NPE_BUF_A_WORDS)];
static npe_word_t B[2][NPE_HALF(NPE_BUF_B_WORDS)];
static npe_word_t C[2][NPE_HALF(NPE_BUF_C_WORDS)];
static npe_wword_t W[NPE_W_WORDS];
static npe_qword_t Q[NPE_Q_WORDS];

static std::vector<uint8_t> read_bytes(FILE *f, size_t n) {
  std::vector<uint8_t> v(n);
  if (fread(v.data(), 1, n, f) != n) {
    printf("case file too short\n");
    exit(2);
  }
  return v;
}

// Word w of a buffer holds bytes [P * w, P * w + P) of its byte image.
template <int N> static void load(npe_word_t buf[2][N], FILE *f) {
  std::vector<uint8_t> img = read_bytes(f, 2 * N * NPE_P);
  for (int w = 0; w < 2 * N; w++)
    for (int k = 0; k < NPE_P; k++)
      buf[w & 1][w >> 1].range(8 * k + 7, 8 * k) = img[NPE_P * w + k];
}

template <int N> static int diff(npe_word_t buf[2][N], FILE *f) {
  std::vector<uint8_t> img = read_bytes(f, 2 * N * NPE_P);
  int bad = 0;
  for (int w = 0; w < 2 * N; w++)
    for (int k = 0; k < NPE_P; k++)
      bad += ((uint8_t)buf[w & 1][w >> 1].range(8 * k + 7, 8 * k) !=
              img[NPE_P * w + k]);
  return bad;
}

template <int BITS, int N> static void load_mem(ap_uint<BITS> (&mem)[N], FILE *f) {
  std::vector<uint8_t> img = read_bytes(f, (size_t)N * BITS / 8);
  for (int w = 0; w < N; w++)
    for (int k = 0; k < BITS / 8; k++)
      mem[w].range(8 * k + 7, 8 * k) = img[(size_t)w * BITS / 8 + k];
}

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    printf("cannot open %s\n", argv[1]);
    return 2;
  }
  uint32_t n_cases;
  fread(&n_cases, 4, 1, f);
  load_mem(W, f);
  load_mem(Q, f);
  int failed = 0;
  for (uint32_t n = 0; n < n_cases; n++) {
    std::vector<uint8_t> raw = read_bytes(f, 2 * NPE_INSTR_FIELDS);
    npe_instr_t instr;
    memcpy(&instr, raw.data(), sizeof instr);
    load(A, f);
    load(B, f);
    load(C, f);
    bool ran = npe_step(instr, NPE_BUF_ARGS, NPE_MEM_ARGS);
    int bad = diff(A, f) + diff(B, f) + diff(C, f);
    if (!ran || bad) {
      failed++;
      printf("case %u: %s, %d wrong bytes\n", n, ran ? "ran" : "NOT RUN", bad);
    }
  }
  printf("%u cases, %d failed\n", n_cases, failed);
  return failed ? 1 : 0;
}
