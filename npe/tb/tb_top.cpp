// Drives npe_top only through its ports, the way a host would: load the
// program and parameters from load.bin, then for each image stream the
// pixels in and the results out, and compare them with the golden logits.
//   tb_top <load.bin> <list file: one golden image directory per line>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "npe_top.h"

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
  FILE *list = argc == 3 ? fopen(argv[2], "r") : NULL;
  if (!list || !read_file(argv[1], load)) {
    printf("usage: tb_top load.bin image_list\n");
    return 2;
  }
  uint32_t hdr[5];
  memcpy(hdr, load.data(), sizeof hdr);
  const int n_out = hdr[4];
  const int bytes[4] = {2 * NPE_INSTR_FIELDS, NPE_P * NPE_P, NPE_P, 9 * NPE_P};
  const int modes[4] = {NPE_LOAD_PROG, NPE_LOAD_W, NPE_LOAD_WD, NPE_LOAD_Q};
  size_t pos = sizeof hdr;
  hls::stream<npe_load_beat_t> load_in;
  hls::stream<npe_img_beat_t> img_in;
  hls::stream<npe_out_beat_t> out;
  for (int m = 0; m < 4; m++) {
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
    npe_top(load_in, img_in, out, modes[m], hdr[m], 0);
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
    for (int p = 0; p < NPE_IN_PIXELS; p++) {
      npe_img_beat_t beat;
      beat.data = 0;
      for (int c = 0; c < NPE_IN_C; c++)
        beat.data.range(8 * c + 7, 8 * c) = img[NPE_IN_C * p + c];
      beat.user = (p == 0);
      beat.last = 0;
      img_in.write(beat);
    }
    npe_top(load_in, img_in, out, NPE_RUN, hdr[0], n_out);
    bool ok = (int)out.size() == n_out;
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
