// Bit-exact INT8 golden model of ShuffleNetV2 x0.5 (3 classes) for the
// bird-vs-drone accelerator. Plain C++11, no HLS types, so it can be linked
// into the Vitis HLS testbench and compared against the hardware tensor by
// tensor. The arithmetic contract is documented in ../quantize.py and
// PLAN.md section 4; any change there must be mirrored here.
#ifndef BD_GOLDEN_MODEL_H_
#define BD_GOLDEN_MODEL_H_

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace golden {

const int IMG_SIZE = 224;
const int NUM_CLASSES = 3; // 0 other, 1 bird, 2 drone

enum DType { U8, I8, I32 };

// Feature map in HWC order (channel fastest), the order the hardware streams.
struct Tensor {
  int h = 0, w = 0, c = 0;
  DType type = U8;
  std::vector<int32_t> d;
  Tensor() {}
  Tensor(int h_, int w_, int c_, DType t)
      : h(h_), w(w_), c(c_), type(t), d((size_t)h_ * w_ * c_, 0) {}
  int32_t &at(int y, int x, int ch) { return d[((size_t)y * w + x) * c + ch]; }
  int32_t at(int y, int x, int ch) const {
    return d[((size_t)y * w + x) * c + ch];
  }
};

enum Kind { STD = 0, DW = 1, PW = 2, FC = 3 };

struct Layer {
  std::string name;
  Kind kind;
  int k, stride, cin, cout;
  bool relu;
  std::vector<int8_t> w; // [cout][cin_per_group][k][k]
  std::vector<int32_t> bias, mult;
  std::vector<uint8_t> shift;
};

// Called with every intermediate tensor, named as in ../ref_int8.py:
// input, conv1, maxpool, s2u0.pw1, ..., s2u0.out, ..., conv5, gap, logits.
typedef std::function<void(const std::string &, const Tensor &)> TensorSink;

class Model {
public:
  // Loads weights.bin. Returns false and fills *err on failure.
  bool load(const std::string &path, std::string *err);

  // img: IMG_SIZE*IMG_SIZE*3 bytes, HWC, RGB. Writes raw INT32 logits
  // (class 0 = other, class 1 = bird, class 2 = drone).
  // Returns false if any accumulator left the INT32 range.
  bool run(const uint8_t *img, int32_t logits[NUM_CLASSES],
           const TensorSink &sink = TensorSink()) const;

  float logit_scale() const { return logit_scale_; }
  const std::vector<Layer> &layers() const { return layers_; }

private:
  Tensor conv(const std::string &name, const Tensor &in) const;
  const Layer &layer(const std::string &name) const;

  std::vector<Layer> layers_;
  std::map<std::string, size_t> index_;
  uint8_t pad_value_[3] = {0, 0, 0};
  float logit_scale_ = 0;
  mutable bool overflow_ = false;
};

// out = clamp((acc * mult + 2^(shift-1)) >> shift) with a floor shift.
int32_t requant(int64_t acc, int32_t mult, int shift, bool relu);

// Writes a tensor as raw bytes: U8/I8 one byte per value, I32 little endian.
bool write_tensor(const std::string &path, const Tensor &t);

} // namespace golden

#endif // BD_GOLDEN_MODEL_H_
