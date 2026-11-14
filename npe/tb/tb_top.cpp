// Drives npe_top only through its ports, the way a host would: load the
// program and parameters from load.bin, then for each image stream the
// pixels in and the results out, and compare them with the golden logits.
//   tb_top <load.bin> <list file: one golden image directory per line> [starve %]
// Streamed weights: weights_dram.bin next to load.bin is the "DRAM" that
// npe_wfetch reads before every run. With a starve percentage the weight
// stream looks empty that often, so the engine has to wait for weights.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "npe_top.h"
#include "npe_wfetch.h"

static bool read_file(const std::string &path, std::vector<uint8_t> &v) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f)
    return false;
  fseek(f, 0, SEEK_END);
  v.resize(ftell(f));
  fseek(f, 0, SEEK_SET);
  bool ok = fread(v.data(), 1, v.size(), f) == v.size();
  fclose(f);
  return ok;
}

int main(int argc, char **argv) {
  std::vector<uint8_t> load;
  FILE *list = argc >= 3 ? fopen(argv[2], "r") : NULL;
  if (!list || !read_file(argv[1], load)) {
    printf("usage: tb_top load.bin image_list\n");
    return 2;
  }
  uint32_t hdr[6]; // words of prog, W, WD, Q, QD; results per run
  memcpy(hdr, load.data(), sizeof hdr);
  const int n_out = hdr[5];
  const int bytes[5] = {2 * NPE_INSTR_FIELDS, npe_wword_t::width / 8,
                        npe_dwword_t::width / 8, npe_qword_t::width / 8,
                        npe_qdword_t::width / 8};
  const int modes[5] = {NPE_LOAD_PROG, NPE_LOAD_W, NPE_LOAD_WD, NPE_LOAD_Q,
                        NPE_LOAD_QD};
  size_t pos = sizeof hdr;
  hls::stream<npe_load_beat_t> load_in;
  hls::stream<npe_img_beat_t> img_in;
  hls::stream<npe_out_beat_t> out;
  hls::stream<npe_ws_t> w_in;
  if (argc >= 4)
    w_in.starve = atoi(argv[3]);
  std::string lb(argv[1]);
  std::string dram_path = lb.substr(0, lb.find_last_of('/') + 1) + "weights_dram.bin";
  std::vector<uint8_t> dram_bytes;
  read_file(dram_path, dram_bytes); // missing or empty: nothing streams
  std::vector<npe_fetch_beat_t> dram(dram_bytes.size() / 16);
  for (size_t b = 0; b < dram.size(); b++)
    for (int k = 0; k < 16; k++)
      dram[b].range(8 * k + 7, 8 * k) = dram_bytes[16 * b + k];
  for (int m = 0; m < 5; m++) {
    int n_beats = hdr[m] * ((bytes[m] + 7) / 8);
    if (pos + 8 * (size_t)n_beats > load.size()) {
      printf("load.bin too short\n");
      return 2;
    }
    for (int b = 0; b < n_beats; b++, pos += 8) {
      npe_load_beat_t beat;
      uint64_t v;
      memcpy(&v, &load[pos], 8);
      beat.data = v;
      beat.last = (b == n_beats - 1);
      load_in.write(beat);
    }
    npe_top(load_in, img_in, out, w_in, modes[m], hdr[m], 0);
    if (!load_in.empty()) {
      printf("load mode %d left %d beats unread\n", modes[m], (int)load_in.size());
      return 2;
    }
  }
  if (pos != load.size()) {
    printf("load.bin has %d extra bytes\n", (int)(load.size() - pos));
    return 2;
  }

  int failed = 0, n = 0;
  char dir[1024];
  for (; fscanf(list, "%1023s", dir) == 1; n++) {
    std::vector<uint8_t> img, gold;
    if (!read_file(std::string(dir) + "/input.bin", img) ||
        !read_file(std::string(dir) + "/logits.bin", gold)) {
      printf("cannot read %s\n", dir);
      return 2;
    }
    for (int p = 0; p < NPE_IN_PIXELS; p += NPE_IN_PPB) {
      npe_img_beat_t beat;
      beat.data = 0;
      for (int j = 0; j < NPE_IN_PPB && p + j < NPE_IN_PIXELS; j++)
        for (int c = 0; c < NPE_IN_C; c++)
          beat.data.range(8 * (NPE_IN_C * j + c) + 7, 8 * (NPE_IN_C * j + c)) =
              img[NPE_IN_C * (p + j) + c];
      beat.user = (p == 0);
      beat.last = 0;
      img_in.write(beat);
    }
    if (!dram.empty())
      npe_wfetch(dram.data(), (int)dram.size(), w_in);
    int starve = w_in.starve;
    npe_top(load_in, img_in, out, w_in, NPE_RUN, hdr[0], n_out);
    w_in.starve = 0;
    bool ok = (int)out.size() == n_out;
    if (!w_in.empty()) {
      printf("run left %d weight beats unread\n", (int)w_in.size());
      ok = false;
    }
    w_in.starve = starve;
    printf("%s:", strrchr(dir, '/') ? strrchr(dir, '/') + 1 : dir);
    for (int k = 0; k < n_out && !out.empty(); k++) {
      npe_out_beat_t beat = out.read();
      int32_t got = (int32_t)(uint32_t)beat.data, want;
      memcpy(&want, &gold[4 * k], 4);
      ok &= (got == want) && (beat.last == (k == n_out - 1));
      printf(" %d", got);
    }
    printf(" %s\n", ok ? "PASS" : "FAIL");
    failed += !ok;
  }
  printf("%d images, %d failed\n", n, failed);
  return (failed || n == 0) ? 1 : 0;
}
