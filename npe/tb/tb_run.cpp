// Runs a whole program on the HLS engine, image after image, and compares the
// int32 results streamed to OUT with the expected ones. Nothing is reset
// between images. The run file is written by npe/tests/test_engine_e2e.py:
//   u32 n_instr, u32 n_images, u32 n_out
//   program (n_instr x NPE_INSTR_FIELDS x u16)
//   W, WD, Q, QD images
//   per image: IN (NPE_IN_PIXELS x NPE_IN_C bytes), expected OUT (n_out x i32)
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
static npe_vword_t V[NPE_V_WORDS];
static npe_wword_t W[NPE_W_WORDS];
static hls::stream<npe_ws_t> WS; // weight stream: unused, no layer streams here
static npe_wring_t WR;
static npe_dwword_t WD[NPE_WD_WORDS];
static npe_qword_t Q[NPE_Q_WORDS];
static npe_qdword_t QD[NPE_QD_WORDS];

static std::vector<uint8_t> read_bytes(FILE *f, size_t n) {
  std::vector<uint8_t> v(n);
  if (fread(v.data(), 1, n, f) != n) {
    printf("run file too short\n");
    exit(2);
  }
  return v;
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
  uint32_t hdr[3];
  fread(hdr, 4, 3, f);
  uint32_t n_instr = hdr[0], n_images = hdr[1], n_out = hdr[2];
  std::vector<npe_instr_t> prog(n_instr);
  for (uint32_t i = 0; i < n_instr; i++) {
    std::vector<uint8_t> raw = read_bytes(f, 2 * NPE_INSTR_FIELDS);
    memcpy(&prog[i], raw.data(), sizeof(npe_instr_t));
  }
  load_mem(W, f);
  load_mem(WD, f);
  load_mem(Q, f);
  load_mem(QD, f);
  int failed = 0;
  for (uint32_t n = 0; n < n_images; n++) {
    load_in(f);
    hls::stream<npe_acc_t> OUT;
    bool all_ran = true;
    for (uint32_t i = 0; i < n_instr; i++)
      all_ran &= npe_step(prog[i], NPE_BUF_ARGS, NPE_MEM_ARGS);
    std::vector<uint8_t> want = read_bytes(f, 4 * n_out);
    bool ok = all_ran && OUT.size() == n_out;
    printf("image %u:", n);
    for (uint32_t k = 0; k < n_out && !OUT.empty(); k++) {
      int32_t w, got = (int32_t)OUT.read();
      memcpy(&w, &want[4 * k], 4);
      ok &= (got == w);
      printf(" %d", got);
    }
    printf(" %s\n", ok ? "PASS" : "FAIL");
    failed += !ok;
  }
  printf("%u images, %d failed\n", n_images, failed);
  return failed ? 1 : 0;
}
