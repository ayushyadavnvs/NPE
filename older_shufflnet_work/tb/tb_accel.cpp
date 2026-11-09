// Full-accelerator testbench. Loads data/unit_weights.bin through the weight
// port, then classifies input.bin of every image folder under the golden
// directory and checks all NUM_CLASSES INT32 logits (other, bird, drone)
// against logits.bin exactly.
// Usage: tb_accel <golden_dir> <unit_weights.bin> [max_images]
#include "../src/accel.h"
#include "../src/layers/units.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

static const int N = IMG_SIZE * IMG_SIZE;

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

static void push_frame(const std::vector<uint8_t> &rgb,
                       hls::stream<axis_pixel_t> &s) {
  for (int i = 0; i < N; i++) {
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

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("usage: %s <golden_dir> <unit_weights.bin> [max_images]\n", argv[0]);
    return 1;
  }
  std::string root = argv[1];
  int max_images = argc > 3 ? atoi(argv[3]) : 0;

  std::vector<uint8_t> image;
  if (!load(argv[2], image) || image.size() != (size_t)UNIT_LOAD_WORDS * 8) {
    printf("bad weight image %s (want %d words)\n", argv[2], UNIT_LOAD_WORDS);
    return 1;
  }
  logit_t other = 0, bird = 0, drone = 0;
  ap_uint<32> count = 0;
  {
    hls::stream<axis_pixel_t> in("tb_in");
    hls::stream<ap_uint<64> > w("tb_w");
    for (int i = 0; i < UNIT_LOAD_WORDS; i++) {
      uint64_t v;
      memcpy(&v, &image[8 * i], 8);
      w.write((ap_uint<64>)v);
    }
    bird_drone_accel(in, w, true, other, bird, drone, count);
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
  if (max_images > 0 && (int)images.size() > max_images)
    images.resize(max_images);
  if (images.empty()) {
    printf("no image folders in %s\n", root.c_str());
    return 1;
  }

  int errors = 0;
  for (size_t f = 0; f < images.size(); f++) {
    std::string dir = root + "/" + images[f] + "/";
    std::vector<uint8_t> rgb, lg;
    if (!load(dir + "input.bin", rgb) || rgb.size() != (size_t)3 * N ||
        !load(dir + "logits.bin", lg) || lg.size() != 4 * NUM_CLASSES) {
      printf("%s: missing or wrong-size input.bin / logits.bin\n",
             images[f].c_str());
      errors++;
      continue;
    }
    int32_t ref[NUM_CLASSES];
    memcpy(ref, lg.data(), sizeof(ref));
    hls::stream<axis_pixel_t> in("tb_in");
    hls::stream<ap_uint<64> > w("tb_w");
    push_frame(rgb, in);
    bird_drone_accel(in, w, false, other, bird, drone, count);
    bool ok = other == ref[CLASS_OTHER] && bird == ref[CLASS_BIRD] &&
              drone == ref[CLASS_DRONE] && count == f + 1;
    printf("%-18s logits hw=(%d,%d,%d) golden=(%d,%d,%d) count=%u %s\n",
           images[f].c_str(), other.to_int(), bird.to_int(), drone.to_int(),
           ref[0], ref[1], ref[2], count.to_uint(), ok ? "PASS" : "FAIL");
    errors += !ok;
  }
  printf("%s: %d of %zu images bit-exact\n", errors ? "TEST FAILED" : "TEST PASSED",
         (int)images.size() - errors, images.size());
  return errors ? 1 : 0;
}
