#include "golden_model.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace golden {

namespace {

const int UNITS_PER_STAGE[3] = {4, 8, 4};

int64_t floor_shift(int64_t v, int s) {
  // Arithmetic right shift that is floor() for negative values on any
  // compiler, matching ap_int >> and NumPy >>.
  return v >= 0 ? (v >> s) : -(((-v) - 1) >> s) - 1;
}

template <typename T> bool read_pod(FILE *f, T *v, size_t n = 1) {
  return fread(v, sizeof(T), n, f) == n;
}

Tensor maxpool3x3s2(const Tensor &in) {
  // Inputs are post-ReLU (>= 0), so padding with 0 equals padding with -inf.
  Tensor out((in.h - 1) / 2 + 1, (in.w - 1) / 2 + 1, in.c, U8);
  for (int y = 0; y < out.h; y++)
    for (int x = 0; x < out.w; x++)
      for (int ch = 0; ch < in.c; ch++) {
        int32_t m = 0;
        for (int ky = 0; ky < 3; ky++)
          for (int kx = 0; kx < 3; kx++) {
            int iy = 2 * y + ky - 1, ix = 2 * x + kx - 1;
            if (iy >= 0 && iy < in.h && ix >= 0 && ix < in.w &&
                in.at(iy, ix, ch) > m)
              m = in.at(iy, ix, ch);
          }
        out.at(y, x, ch) = m;
      }
  return out;
}

// torchvision channel_shuffle(groups=2) of concat(a, b):
// out[2i] = a[i], out[2i+1] = b[i].
Tensor concat_shuffle(const Tensor &a, const Tensor &b) {
  Tensor out(a.h, a.w, a.c + b.c, U8);
  for (int y = 0; y < a.h; y++)
    for (int x = 0; x < a.w; x++)
      for (int i = 0; i < a.c; i++) {
        out.at(y, x, 2 * i) = a.at(y, x, i);
        out.at(y, x, 2 * i + 1) = b.at(y, x, i);
      }
  return out;
}

Tensor channel_slice(const Tensor &in, int first, int count) {
  Tensor out(in.h, in.w, count, in.type);
  for (int y = 0; y < in.h; y++)
    for (int x = 0; x < in.w; x++)
      for (int i = 0; i < count; i++)
        out.at(y, x, i) = in.at(y, x, first + i);
  return out;
}

} // namespace

int32_t requant(int64_t acc, int32_t mult, int shift, bool relu) {
  int64_t v = floor_shift(acc * (int64_t)mult + ((int64_t)1 << (shift - 1)),
                          shift);
  int64_t lo = relu ? 0 : -128, hi = relu ? 255 : 127;
  if (v < lo)
    v = lo;
  if (v > hi)
    v = hi;
  return (int32_t)v;
}

bool Model::load(const std::string &path, std::string *err) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) {
    *err = "cannot open " + path;
    return false;
  }
  char magic[4];
  uint32_t version = 0, n = 0;
  uint8_t reserved;
  bool ok = read_pod(f, magic, 4) && memcmp(magic, "BDQW", 4) == 0 &&
            read_pod(f, &version) && version == 1 && read_pod(f, &n) &&
            read_pod(f, pad_value_, 3) && read_pod(f, &reserved) &&
            read_pod(f, &logit_scale_);
  layers_.clear();
  index_.clear();
  for (uint32_t i = 0; ok && i < n; i++) {
    char name[33] = {0};
    uint8_t kind, k, stride, relu;
    uint16_t cin, cout;
    ok = read_pod(f, name, 32) && read_pod(f, &kind) && read_pod(f, &k) &&
         read_pod(f, &stride) && read_pod(f, &relu) && read_pod(f, &cin) &&
         read_pod(f, &cout);
    if (!ok)
      break;
    Layer L;
    L.name = name;
    L.kind = (Kind)kind;
    L.k = k;
    L.stride = stride;
    L.relu = relu != 0;
    L.cin = cin;
    L.cout = cout;
    int cin_g = (L.kind == DW) ? 1 : cin;
    L.w.resize((size_t)cout * cin_g * k * k);
    L.bias.resize(cout);
    L.mult.resize(cout);
    L.shift.resize(cout);
    ok = read_pod(f, L.w.data(), L.w.size()) &&
         read_pod(f, L.bias.data(), cout) && read_pod(f, L.mult.data(), cout) &&
         read_pod(f, L.shift.data(), cout);
    index_[L.name] = layers_.size();
    layers_.push_back(L);
  }
  if (ok && fgetc(f) != EOF) {
    ok = false;
  }
  fclose(f);
  if (!ok)
    *err = "malformed weights file " + path;
  return ok;
}

const Layer &Model::layer(const std::string &name) const {
  auto it = index_.find(name);
  if (it == index_.end())
    throw std::runtime_error("layer missing from weights file: " + name);
  return layers_[it->second];
}

// Standard, depthwise or pointwise convolution with padding k/2, followed by
// requantization. Only conv1 pads with a non-zero value (the folded mean).
Tensor Model::conv(const std::string &name, const Tensor &in) const {
  const Layer &L = layer(name);
  const int p = L.k / 2;
  const int oh = (in.h + 2 * p - L.k) / L.stride + 1;
  const int ow = (in.w + 2 * p - L.k) / L.stride + 1;
  const bool is_dw = L.kind == DW;
  const int cin_g = is_dw ? 1 : L.cin;
  Tensor out(oh, ow, L.cout, L.relu ? U8 : I8);
  for (int y = 0; y < oh; y++)
    for (int x = 0; x < ow; x++)
      for (int o = 0; o < L.cout; o++) {
        int64_t acc = L.bias[o];
        for (int ky = 0; ky < L.k; ky++)
          for (int kx = 0; kx < L.k; kx++) {
            int iy = y * L.stride + ky - p, ix = x * L.stride + kx - p;
            bool inside = iy >= 0 && iy < in.h && ix >= 0 && ix < in.w;
            for (int ci = 0; ci < cin_g; ci++) {
              int ch = is_dw ? o : ci;
              int32_t v = inside ? in.at(iy, ix, ch)
                                 : (name == "conv1" ? pad_value_[ch] : 0);
              int32_t wt = L.w[(((size_t)o * cin_g + ci) * L.k + ky) * L.k + kx];
              acc += (int64_t)wt * v;
            }
          }
        if (acc > INT32_MAX || acc < INT32_MIN)
          overflow_ = true;
        out.at(y, x, o) = requant(acc, L.mult[o], L.shift[o], L.relu);
      }
  return out;
}

bool Model::run(const uint8_t *img, int32_t logits[NUM_CLASSES],
                const TensorSink &sink) const {
  overflow_ = false;
  auto emit = [&](const std::string &n, const Tensor &t) {
    if (sink)
      sink(n, t);
  };

  Tensor x(IMG_SIZE, IMG_SIZE, 3, U8);
  for (size_t i = 0; i < x.d.size(); i++)
    x.d[i] = img[i];
  emit("input", x);

  x = conv("conv1", x);
  emit("conv1", x);
  x = maxpool3x3s2(x);
  emit("maxpool", x);

  for (int s = 0; s < 3; s++) {
    for (int u = 0; u < UNITS_PER_STAGE[s]; u++) {
      std::string pre = "s" + std::to_string(s + 2) + "u" + std::to_string(u);
      auto run_layer = [&](const std::string &part, const Tensor &t) {
        Tensor r = conv(pre + "." + part, t);
        emit(pre + "." + part, r);
        return r;
      };
      Tensor b1, b2;
      if (u == 0) {
        b1 = run_layer("b1.pw", run_layer("b1.dw", x));
        b2 = run_layer("pw2", run_layer("dw", run_layer("pw1", x)));
      } else {
        int half = x.c / 2;
        b1 = channel_slice(x, 0, half);
        b2 = run_layer("pw2",
                       run_layer("dw", run_layer("pw1", channel_slice(x, half,
                                                                      half))));
      }
      x = concat_shuffle(b1, b2);
      emit(pre + ".out", x);
    }
  }

  x = conv("conv5", x);
  emit("conv5", x);

  Tensor gap(1, 1, x.c, I32);
  for (int ch = 0; ch < x.c; ch++) {
    int64_t s = 0;
    for (int y = 0; y < x.h; y++)
      for (int xx = 0; xx < x.w; xx++)
        s += x.at(y, xx, ch);
    gap.d[ch] = (int32_t)s;
  }
  emit("gap", gap);

  const Layer &fc = layer("fc");
  Tensor out(1, 1, fc.cout, I32);
  for (int k = 0; k < fc.cout; k++) {
    int64_t acc = fc.bias[k];
    for (int ch = 0; ch < fc.cin; ch++)
      acc += (int64_t)fc.w[(size_t)k * fc.cin + ch] * gap.d[ch];
    if (acc > INT32_MAX || acc < INT32_MIN)
      overflow_ = true;
    out.d[k] = (int32_t)acc;
    if (k < NUM_CLASSES)
      logits[k] = out.d[k];
  }
  emit("logits", out);
  return !overflow_;
}

bool write_tensor(const std::string &path, const Tensor &t) {
  FILE *f = fopen(path.c_str(), "wb");
  if (!f)
    return false;
  bool ok = true;
  for (int32_t v : t.d) {
    if (t.type == I32) {
      uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16),
                      (uint8_t)(v >> 24)};
      ok &= fwrite(b, 1, 4, f) == 4;
    } else {
      uint8_t b = (uint8_t)(v & 0xff);
      ok &= fwrite(&b, 1, 1, f) == 1;
    }
  }
  fclose(f);
  return ok;
}

} // namespace golden
