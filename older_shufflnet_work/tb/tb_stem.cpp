// Stem testbench: for every image folder under the golden directory, checks
// conv1 (112x112x24) and maxpool (56x56x24) byte for byte against
// conv1.bin and maxpool.bin. conv1 is checked by running the stem's first
// stages directly, maxpool through stem_top.
// Usage: tb_stem <golden_dir>   (e.g. .../data/golden)
#include "../src/layers/io.h"
#include "../src/stem_top.h"

#include <algorithm>
#include <cstdio>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

static bool load(const std::string &path, std::vector<uint8_t> &buf,
                 size_t size) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f)
    return false;
  buf.resize(size);
  size_t n = fread(buf.data(), 1, size, f);
  bool extra = fgetc(f) != EOF;
  fclose(f);
  return n == size && !extra;
}

static void push_frame(const std::vector<uint8_t> &rgb,
                       hls::stream<axis_pixel_t> &s) {
  for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    axis_pixel_t beat;
    beat.data = (ap_uint<32>)rgb[3 * i] | ((ap_uint<32>)rgb[3 * i + 1] << 8) |
                ((ap_uint<32>)rgb[3 * i + 2] << 16);
    beat.keep = -1;
    beat.strb = -1;
    beat.user = (i == 0);
    beat.last = ((i % IMG_SIZE) == IMG_SIZE - 1);
    beat.id = 0;
    beat.dest = 0;
    s.write(beat);
  }
}

// Compares a stream of packed pixels with an HWC uint8 tensor.
static long compare(const char *what, hls::stream<stem_pix_t> &s,
                    const std::vector<uint8_t> &ref, int size) {
  long bad = 0;
  for (int p = 0; p < size * size; p++) {
    if (s.empty()) {
      printf("  %s: stream ended after %d of %d pixels\n", what, p,
             size * size);
      return bad + (long)(size * size - p) * STEM_COUT;
    }
    stem_pix_t v = s.read();
    for (int c = 0; c < STEM_COUT; c++) {
      int hw = (int)v.range(8 * c + 7, 8 * c);
      int gold = ref[(size_t)p * STEM_COUT + c];
      if (hw != gold && bad++ < 5)
        printf("  %s mismatch y=%d x=%d c=%d hw=%d golden=%d\n", what,
               p / size, p % size, c, hw, gold);
    }
  }
  if (!s.empty()) {
    printf("  %s: extra data in stream\n", what);
    bad++;
  }
  return bad;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("usage: %s <golden_dir>\n", argv[0]);
    return 1;
  }
  std::string root = argv[1];
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

  int failed = 0;
  for (const std::string &img : images) {
    std::string dir = root + "/" + img + "/";
    std::vector<uint8_t> input, conv1, pool;
    if (!load(dir + "input.bin", input, IMG_SIZE * IMG_SIZE * 3) ||
        !load(dir + "conv1.bin", conv1, STEM_SIZE * STEM_SIZE * STEM_COUT) ||
        !load(dir + "maxpool.bin", pool, POOL_SIZE * POOL_SIZE * STEM_COUT)) {
      printf("%s: missing or wrong-size golden files\n", img.c_str());
      failed++;
      continue;
    }

    // conv1 through the stem's first stages.
    hls::stream<axis_pixel_t> in1("in1");
    hls::stream<rgb_t> px("px");
    hls::stream<stem_win_t> win("win");
    hls::stream<stem_pix_t> c1("c1");
    push_frame(input, in1);
    read_frame<IMG_SIZE>(in1, px);
    stem_conv1_window(px, win);
    stem_conv1_engine(win, c1);
    long bad_c1 = compare("conv1", c1, conv1, STEM_SIZE);

    // maxpool through the synthesizable top.
    hls::stream<axis_pixel_t> in2("in2");
    hls::stream<stem_pix_t> out("out");
    push_frame(input, in2);
    stem_top(in2, out);
    long bad_mp = compare("maxpool", out, pool, POOL_SIZE);

    bool ok = bad_c1 == 0 && bad_mp == 0;
    printf("%-18s conv1 %s (%ld bad)  maxpool %s (%ld bad)\n", img.c_str(),
           bad_c1 ? "FAIL" : "ok", bad_c1, bad_mp ? "FAIL" : "ok", bad_mp);
    if (!ok)
      failed++;
  }
  if (images.empty()) {
    printf("no image folders in %s\n", root.c_str());
    return 1;
  }
  printf("%s: %d of %zu images bit-exact\n", failed ? "TEST FAILED" : "TEST PASSED",
         (int)images.size() - failed, images.size());
  return failed ? 1 : 0;
}
