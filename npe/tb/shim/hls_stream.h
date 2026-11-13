#pragma once
#include <deque>
#include <cstdio>
#include <cstdlib>
namespace hls {
template <typename T> class stream {
  std::deque<T> q;
public:
  stream() {}
  stream(const char *) {}
  void write(const T &v) { q.push_back(v); }
  T read() { if (q.empty()) { fprintf(stderr, "read from empty stream\n"); abort(); } T v = q.front(); q.pop_front(); return v; }
  void read(T &v) { v = read(); }
  bool empty() const { return q.empty(); }
  bool full() const { return false; }
  size_t size() const { return q.size(); }
  stream &operator<<(const T &v) { write(v); return *this; }
  stream &operator>>(T &v) { v = read(); return *this; }
};
}
