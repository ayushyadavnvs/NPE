// Command-line driver for the INT8 golden model.
//   golden_model <weights.bin> <image.bin> [dump_dir]
// image.bin: 224*224*3 bytes, HWC RGB (make one with ../img2bin.py).
// With dump_dir, every intermediate tensor is written as <dump_dir>/<name>.bin.
#include "golden_model.h"

#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s weights.bin image.bin [dump_dir]\n", argv[0]);
    return 2;
  }
  golden::Model model;
  std::string err;
  if (!model.load(argv[1], &err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }

  const size_t n = (size_t)golden::IMG_SIZE * golden::IMG_SIZE * 3;
  std::vector<uint8_t> img(n);
  FILE *f = fopen(argv[2], "rb");
  if (!f || fread(img.data(), 1, n, f) != n) {
    fprintf(stderr, "cannot read %zu bytes from %s\n", n, argv[2]);
    return 1;
  }
  fclose(f);

  golden::TensorSink sink;
  bool dump_ok = true;
  const std::string dir = argc > 3 ? argv[3] : "";
  if (argc > 3) {
    sink = [&](const std::string &name, const golden::Tensor &t) {
      dump_ok &= golden::write_tensor(dir + "/" + name + ".bin", t);
    };
  }

  int32_t logits[golden::NUM_CLASSES];
  bool ok = model.run(img.data(), logits, sink);
  static const char *const names[golden::NUM_CLASSES] = {"other", "bird",
                                                          "drone"};
  int best = 0; // argmax, first index wins ties (as numpy.argmax)
  printf("logits_raw");
  for (int k = 0; k < golden::NUM_CLASSES; k++) {
    printf(" %d", logits[k]);
    if (logits[k] > logits[best])
      best = k;
  }
  printf("\nlogits_float");
  for (int k = 0; k < golden::NUM_CLASSES; k++)
    printf(" %.6f", logits[k] * model.logit_scale());
  printf("\nclass %d (%s)\n", best, names[best]);
  if (!ok)
    fprintf(stderr, "ERROR: an accumulator left the INT32 range\n");
  if (!dump_ok)
    fprintf(stderr, "ERROR: could not write tensor dumps to %s\n", argv[3]);
  return (ok && dump_ok) ? 0 : 1;
}
