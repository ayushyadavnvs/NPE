// Head-only test for the 3-class fc (plain g++ or Vitis csim): fills T1 with
// conv5-shaped data, loads FC_W / FC_B through unit_load_weights-compatible
// packing, runs unit_head and compares against a plain C++ reference.
// Cases include the INT32 worst case (all inputs 255, weights +-127).
// Usage: tb_head
#include "../src/layers/units.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static uword_t X[UNIT_BUF_X_WORDS], T1[UNIT_BUF_T1_WORDS],
    T2[UNIT_BUF_T2_WORDS], B1[UNIT_BUF_B1_WORDS], B2[UNIT_BUF_B2_WORDS];
static fc_wword_t FC_W[UNIT_FC_WORDS];
static fc_bword_t FC_B;

int main() {
  const unit_layer_t head = UNIT_PROGRAM[NET_NUM_LAYERS - 1];
  const int groups = head.cin_g, pixels = head.size * head.size;
  const int cin = groups * UW;
  std::mt19937 rng(1234);
  int errors = 0;
  for (int c = 0; c < 6; c++) {
    std::vector<int> act(pixels * cin), w(NUM_CLASSES * cin);
    int32_t bias[NUM_CLASSES];
    for (size_t i = 0; i < act.size(); i++)
      act[i] = c == 0 ? 255 : c == 1 ? 0 : (int)(rng() % 256);
    for (int k = 0; k < NUM_CLASSES; k++) {
      for (int ch = 0; ch < cin; ch++) {
        int r = (int)(rng() % 255) - 127;
        // case 0: class 0 all +127, class 1 all -127, class 2 alternating
        w[k * cin + ch] = c == 0 ? (k == 0 ? 127 : k == 1 ? -127
                                                 : (ch & 1 ? 127 : -127))
                                 : r;
      }
      bias[k] = c == 0 ? (k == 0 ? (1 << 21) : -(1 << 21)) : (int32_t)(rng() % (1 << 26)) - (1 << 25);
    }
    // Pack exactly like tools/gen_weights.py + unit_load_weights.
    hls::stream<ap_uint<64> > s;
    for (int g = 0; g < groups; g++)
      for (int k = 0; k < NUM_CLASSES; k++) {
        ap_uint<64> beat = 0;
        for (int ch = 0; ch < UW; ch++)
          beat.range(8 * ch + 7, 8 * ch) = (ap_uint<8>)(int8_t)w[k * cin + g * UW + ch];
        s.write(beat);
      }
    for (int b = 0; b < FC_B_BEATS; b++) {
      ap_uint<64> beat = 0;
      for (int j = 0; j < 2; j++)
        if (2 * b + j < NUM_CLASSES)
          beat.range(32 * j + 31, 32 * j) = (ap_uint<32>)(uint32_t)bias[2 * b + j];
      s.write(beat);
    }
    fc_wword_t word = 0;
    for (int i = 0; i < groups * NUM_CLASSES; i++) {
      word = (word >> 64) | ((fc_wword_t)s.read() << (fc_wword_t::width - 64));
      if (i % NUM_CLASSES == NUM_CLASSES - 1)
        FC_W[i / NUM_CLASSES] = word;
    }
    fc_bword_t bw = 0;
    for (int i = 0; i < FC_B_BEATS; i++)
      bw = (bw >> 64) | ((fc_bword_t)s.read() << (fc_bword_t::width - 64));
    FC_B = bw;

    for (int p = 0; p < pixels; p++)
      for (int g = 0; g < groups; g++) {
        uword_t word8 = 0;
        for (int ch = 0; ch < UW; ch++)
          word8.range(8 * ch + 7, 8 * ch) = act[p * cin + g * UW + ch];
        T1[p * groups + g] = word8;
      }
    logit_t lg[NUM_CLASSES];
    unit_head(head, X, T1, T2, B1, B2, FC_W, FC_B, lg);
    for (int k = 0; k < NUM_CLASSES; k++) {
      int64_t ref = bias[k];
      for (int ch = 0; ch < cin; ch++) {
        int64_t gap = 0;
        for (int p = 0; p < pixels; p++)
          gap += act[p * cin + ch];
        ref += gap * w[k * cin + ch];
      }
      bool ok = ref == (int64_t)lg[k].to_int();
      if (!ok || c == 0)
        printf("case %d class %d hw=%d ref=%lld %s\n", c, k, lg[k].to_int(),
               (long long)ref, ok ? "ok" : "MISMATCH");
      errors += !ok;
    }
  }
  printf("%s: head test, %d mismatches\n", errors ? "TEST FAILED" : "TEST PASSED",
         errors);
  return errors ? 1 : 0;
}
