// Unit engine testbench. Loads data/unit_weights.bin, then for every image
// folder under the golden directory:
//  1. runs the program layer by layer on golden maxpool.bin with the engine
//     functions from src/layers/units.h and checks every tensor (b1.dw, b1.pw,
//     pw1, dw, pw2, s{st}u{u}.out, conv5) byte for byte;
//  2. runs the synthesizable unit_engine_top up to conv5 and checks conv5.
// Usage: tb_units <golden_dir> <unit_weights.bin> [quick]
//   quick: step 2 only (for cosim).
#include "../src/unit_engine_top.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

static bool load(const std::string &path, std::vector<uint8_t> &buf) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f)
    return false;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  buf.resize(n);
  bool ok = fread(buf.data(), 1, n, f) == (size_t)n;
  fclose(f);
  return ok;
}

// Compares the first gold.size() bytes of a word buffer with a golden tensor.
static long diff(const uword_t *buf, const std::vector<uint8_t> &gold,
                 const char *name) {
  long bad = 0;
  for (size_t i = 0; i < gold.size(); i++) {
    int hw = (int)buf[i / UW].range(8 * (i % UW) + 7, 8 * (i % UW));
    if (hw != gold[i] && bad++ < 3)
      printf("    %s byte %zu: hw=%d golden=%d\n", name, i, hw, gold[i]);
  }
  return bad;
}

static int out_words(const unit_layer_t &L) {
  int osz = (L.kind == UL_DW && L.stride == 2) ? L.size / 2 : L.size;
  return osz * osz * L.cout_g;
}

// Testbench copies of the engine memories for the layer-by-layer run.
static uword_t X[UNIT_BUF_X_WORDS], T1[UNIT_BUF_T1_WORDS], T2[UNIT_BUF_T2_WORDS],
    B1[UNIT_BUF_B1_WORDS], B2[UNIT_BUF_B2_WORDS];
static pw_wword_t PW_W[UNIT_PW_WORDS];
static dw_wword_t DW_W[UNIT_DW_WORDS];
static pw_qword_t PW_Q[UNIT_QP_WORDS];
static dw_qword_t DW_Q[UNIT_QD_WORDS];
static fc_wword_t FC_W[UNIT_FC_WORDS];
static fc_bword_t FC_B;

static const uword_t *dst_buf(int id) {
  switch (id) {
  case BUF_X: return X;
  case BUF_T1: return T1;
  case BUF_T2: return T2;
  case BUF_B1: return B1;
  default: return B2;
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("usage: %s <golden_dir> <unit_weights.bin> [quick]\n", argv[0]);
    return 1;
  }
  std::string root = argv[1];
  bool quick = argc > 3 && strcmp(argv[3], "quick") == 0;

  std::vector<uint8_t> image;
  if (!load(argv[2], image) || image.size() != (size_t)UNIT_LOAD_WORDS * 8) {
    printf("bad weight image %s (want %d words)\n", argv[2], UNIT_LOAD_WORDS);
    return 1;
  }
  {
    hls::stream<ap_uint<64> > w1("w1"), w2("w2");
    hls::stream<stem_pix_t> fin("fin");
    hls::stream<uword_t> fout("fout");
    for (int i = 0; i < UNIT_LOAD_WORDS; i++) {
      uint64_t v;
      memcpy(&v, &image[8 * i], 8);
      w1.write((ap_uint<64>)v);
      w2.write((ap_uint<64>)v);
    }
    unit_engine_top(w1, fin, fout, true, 0);
    unit_load_weights(w2, PW_W, DW_W, PW_Q, DW_Q, FC_W, FC_B);
  }

  std::vector<std::string> images;
  DIR *d = opendir(root.c_str());
  if (!d) {
    printf("cannot open %s\n", root.c_str());
    return 1;
  }
  while (dirent *e = readdir(d)) {
    std::string name = e->d_name;
    struct stat st;
    if (name[0] != '.' && stat((root + "/" + name).c_str(), &st) == 0 &&
        S_ISDIR(st.st_mode))
      images.push_back(name);
  }
  closedir(d);
  std::sort(images.begin(), images.end());
  if (images.empty()) {
    printf("no image folders in %s\n", root.c_str());
    return 1;
  }

  const int conv5 = UNIT_NUM_LAYERS; // program index of conv5
  int failed = 0;
  for (const std::string &img : images) {
    std::string dir = root + "/" + img + "/";
    std::vector<uint8_t> pool, gold;
    if (!load(dir + "maxpool.bin", pool) ||
        pool.size() != (size_t)POOL_SIZE * POOL_SIZE * STEM_COUT) {
      printf("%s: missing maxpool.bin\n", img.c_str());
      failed++;
      continue;
    }
    bool ok = true;
    int units_ok = 0;

    // 1. Layer by layer, every tensor.
    if (!quick) {
      for (size_t i = 0; i < pool.size(); i++)
        X[i / UW].range(8 * (i % UW) + 7, 8 * (i % UW)) = pool[i];
      const char *first_bad = 0;
      for (int l = 0; l <= conv5; l++) {
        const unit_layer_t &L = UNIT_PROGRAM[l];
        logit_t lg[NUM_CLASSES];
        unit_layer(L, X, T1, T2, B1, B2, PW_W, DW_W, PW_Q, DW_Q, FC_W, FC_B, lg);
        if (!load(dir + UNIT_LAYER_NAMES[l] + ".bin", gold) ||
            gold.size() != (size_t)out_words(L) * UW) {
          printf("    %s: missing or wrong-size golden file\n", UNIT_LAYER_NAMES[l]);
          ok = false;
          break;
        }
        if (diff(dst_buf(L.dst), gold, UNIT_LAYER_NAMES[l]) != 0) {
          first_bad = UNIT_LAYER_NAMES[l];
          ok = false;
          break;
        }
        if (L.kind == UL_MERGE)
          units_ok++;
      }
      if (first_bad)
        printf("    first differing tensor: %s\n", first_bad);
    }

    // 2. Synthesizable top up to conv5.
    hls::stream<ap_uint<64> > win("win");
    hls::stream<stem_pix_t> fin("fin");
    hls::stream<uword_t> fout("fout");
    for (int p = 0; p < POOL_SIZE * POOL_SIZE; p++) {
      stem_pix_t px = 0;
      for (int c = 0; c < STEM_COUT; c++)
        px.range(8 * c + 7, 8 * c) = pool[p * STEM_COUT + c];
      fin.write(px);
    }
    unit_engine_top(win, fin, fout, false, conv5 + 1);
    std::vector<uword_t> top_out;
    while (!fout.empty())
      top_out.push_back(fout.read());
    bool top_ok = load(dir + "conv5.bin", gold) &&
                  top_out.size() * UW == gold.size() &&
                  diff(top_out.data(), gold, "top conv5") == 0;
    ok &= top_ok;

    if (quick)
      printf("%-18s top conv5 %s\n", img.c_str(), top_ok ? "ok" : "FAIL");
    else
      printf("%-18s %d of 16 units + conv5 bit-exact layer by layer, top conv5 "
             "%s%s\n", img.c_str(), units_ok, top_ok ? "ok" : "FAIL",
             ok ? "" : "  FAIL");
    if (!ok)
      failed++;
  }
  printf("%s: %d of %zu images bit-exact\n",
         failed ? "TEST FAILED" : "TEST PASSED", (int)images.size() - failed,
         images.size());
  return failed ? 1 : 0;
}
