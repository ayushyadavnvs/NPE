// Runs instructions on the HLS engine and compares every feature buffer with
// the state the reference interpreter reached. The case file is written by
// npe/tests/test_engine.py:
//   u32 n_cases
//   W image (NPE_W_WORDS words), WD image (NPE_WD_WORDS words),
//   Q image (NPE_Q_WORDS words), QD image (NPE_QD_WORDS words)
//   IN image (NPE_IN_PIXELS x NPE_IN_C bytes)
//   per case: the instruction (NPE_INSTR_FIELDS x u16), buffers S0, A, B, C
//             (one byte per channel) and V (4 bytes per channel) before it,
//             the same after it, then u32 n_out and the n_out int32 values
//             the instruction must stream to OUT
#include <cstdio>
#include <cstring>
#include <vector>

#include "npe_engine.h"

static npe_pixel_t IN[NPE_IN_PPB][NPE_IN_ROWS];

// Pixel p of the image (IN_C bytes each) -> bank p % PPB, row p / PPB.
static void load_in(FILE *f);
static npe_word_t S0[NPE_NB][NPE_ROWS(NPE_BUF_S0_WORDS)];
static npe_word_t A[NPE_NB][NPE_ROWS(NPE_BUF_A_WORDS)];
static npe_word_t B[NPE_NB][NPE_ROWS(NPE_BUF_B_WORDS)];
static npe_word_t C[NPE_NB][NPE_ROWS(NPE_BUF_C_WORDS)];
static npe_wword_t W[NPE_W_WORDS];
static hls::stream<npe_ws_t> WS; // weight stream: unused, no layer streams here
static npe_wring_t WR;
static npe_vword_t V[NPE_V_WORDS];
static npe_dwword_t WD[NPE_WD_WORDS];
static npe_qword_t Q[NPE_Q_WORDS];
static npe_qdword_t QD[NPE_QD_WORDS];

static std::vector<uint8_t> read_bytes(FILE *f, size_t n) {
  std::vector<uint8_t> v(n);
  if (fread(v.data(), 1, n, f) != n) {
    printf("case file too short\n");
    exit(2);
  }
  return v;
}

// Word w of a buffer (bank w % NB, row w / NB) holds bytes [P * w, P * w + P)
// of its byte image.
template <int N> static void load(npe_word_t buf[NPE_NB][N], FILE *f) {
  std::vector<uint8_t> img = read_bytes(f, NPE_NB * N * NPE_P);
  for (int w = 0; w < NPE_NB * N; w++)
    for (int k = 0; k < NPE_P; k++)
      buf[w % NPE_NB][w / NPE_NB].range(8 * k + 7, 8 * k) = img[NPE_P * w + k];
}

template <int N> static int diff(npe_word_t buf[NPE_NB][N], FILE *f) {
  std::vector<uint8_t> img = read_bytes(f, NPE_NB * N * NPE_P);
  int bad = 0;
  for (int w = 0; w < NPE_NB * N; w++)
    for (int k = 0; k < NPE_P; k++)
      bad += ((uint8_t)buf[w % NPE_NB][w / NPE_NB].range(8 * k + 7, 8 * k) !=
              img[NPE_P * w + k]);
  return bad;
}

static int diff_v(FILE *f) {
  std::vector<uint8_t> img = read_bytes(f, NPE_V_WORDS * NPE_P * 4);
  int bad = 0;
  for (int w = 0; w < NPE_V_WORDS; w++)
    for (int k = 0; k < NPE_P * 4; k++)
      bad += ((uint8_t)V[w].range(8 * k + 7, 8 * k) != img[NPE_P * 4 * w + k]);
  return bad;
}

template <int BITS, int N> static void load_mem(ap_uint<BITS> (&mem)[N], FILE *f) {
  std::vector<uint8_t> img = read_bytes(f, (size_t)N * BITS / 8);
  for (int w = 0; w < N; w++)
    for (int k = 0; k < BITS / 8; k++)
      mem[w].range(8 * k + 7, 8 * k) = img[(size_t)w * BITS / 8 + k];
}

static void load_in(FILE *f) {
  std::vector<uint8_t> img = read_bytes(f, (size_t)NPE_IN_PIXELS * NPE_IN_C);
  for (int p = 0; p < NPE_IN_PIXELS; p++)
    for (int c = 0; c < NPE_IN_C; c++)
      IN[p % NPE_IN_PPB][p / NPE_IN_PPB].range(8 * c + 7, 8 * c) = img[NPE_IN_C * p + c];
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
  load_mem(WD, f);
  load_mem(Q, f);
  load_mem(QD, f);
  load_in(f);
  int failed = 0;
  for (uint32_t n = 0; n < n_cases; n++) {
    std::vector<uint8_t> raw = read_bytes(f, 2 * NPE_INSTR_FIELDS);
    npe_instr_t instr;
    memcpy(&instr, raw.data(), sizeof instr);
    load(S0, f);
    load(A, f);
    load(B, f);
    load(C, f);
    load_mem(V, f);
    hls::stream<npe_acc_t> OUT;
    bool ran = npe_step(instr, NPE_BUF_ARGS, NPE_MEM_ARGS);
    int bad = diff(S0, f) + diff(A, f) + diff(B, f) + diff(C, f) + diff_v(f);
    uint32_t n_out;
    fread(&n_out, 4, 1, f);
    std::vector<uint8_t> want = read_bytes(f, 4 * n_out);
    for (uint32_t k = 0; k < n_out; k++) {
      int32_t w;
      memcpy(&w, &want[4 * k], 4);
      if (OUT.empty() || (int32_t)OUT.read() != w)
        bad++;
    }
    bad += OUT.size(); // nothing extra may be streamed
    if (!ran || bad) {
      failed++;
      printf("case %u: %s, %d wrong bytes\n", n, ran ? "ran" : "NOT RUN", bad);
    }
  }
  printf("%u cases, %d failed\n", n_cases, failed);
  return failed ? 1 : 0;
}
