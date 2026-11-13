"""NumPy reference interpreter of the NPE instruction set (npe/compiler/isa.py).

It is the definition of what each instruction means: the HLS engine and the
compiler are both checked against it. Buffers are flat int64 element arrays;
precision only sets the clamp range.
"""
import numpy as np

from npe.compiler import isa


def _range(bits, signed):
    return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)


def _requant(acc, M, S, lo, hi):
    acc = acc.astype(np.int64)
    assert np.abs(acc).max(initial=0) < 2 ** 31, "accumulator exceeds int32"
    y = (acc * M + (np.int64(1) << (S - 1))) >> S
    return np.clip(y, lo, hi)


class Machine:
    def __init__(self, params, sizes):
        """params: list of dicts (w, bias, M, S, optional pad) indexed by w_base.
        sizes: {buf id: element capacity}."""
        self.params = params
        self.buf = {b: np.zeros(n, np.int64) for b, n in sizes.items()}

    # tensor views -------------------------------------------------------
    @staticmethod
    def _idx(base, h, w, c, pix, off):
        y, x, ch = np.meshgrid(np.arange(h), np.arange(w), np.arange(c), indexing="ij")
        return base + (y * w + x) * pix + off + ch

    def read(self, buf, base, h, w, c, pix, off):
        return self.buf[buf][self._idx(base, h, w, c, pix or c, off)]

    def write(self, buf, base, t, pix, off):
        h, w, c = t.shape
        self.buf[buf][self._idx(base, h, w, c, pix or c, off)] = t

    # ops ----------------------------------------------------------------
    def _windows(self, a, I, padv):
        H, W, C = a.shape
        xp = np.empty((H + I.pad_t + I.pad_b, W + I.pad_l + I.pad_r, C), np.int64)
        xp[...] = padv
        xp[I.pad_t:I.pad_t + H, I.pad_l:I.pad_l + W] = a
        Ho = (H + I.pad_t + I.pad_b - I.kh) // I.sh + 1
        Wo = (W + I.pad_l + I.pad_r - I.kw) // I.sw + 1
        assert (Ho, Wo) == (I.o_h, I.o_w), (Ho, Wo, I)
        win = np.empty((Ho, Wo, I.kh, I.kw, C), np.int64)
        for i in range(I.kh):
            for j in range(I.kw):
                win[:, :, i, j] = xp[i:i + I.sh * Ho:I.sh, j:j + I.sw * Wo:I.sw]
        return win

    def _conv(self, a, I):
        P = self.params[I.w_base]
        padv = P["pad"] if I.pad_mode == isa.PAD_CONST else 0
        win = self._windows(a, I, padv)                 # Ho Wo kh kw C
        w = P["w"].astype(np.int64)                     # Cout Cin/g kh kw
        cout, cin_g = w.shape[:2]
        G = I.groups
        assert cout == I.o_c and cin_g * G == I.a_c and cout % G == 0
        Ho, Wo = win.shape[:2]
        win = win.reshape(Ho, Wo, I.kh, I.kw, G, cin_g)
        wg = w.reshape(G, cout // G, cin_g, I.kh, I.kw)
        # per group: matmul over (kh, kw, cin_g)
        acc = np.empty((Ho, Wo, cout), np.int64)
        og = cout // G
        for g in range(G):
            patch = win[:, :, :, :, g, :].reshape(Ho, Wo, -1)            # kh kw cin_g
            wm = wg[g].transpose(2, 3, 1, 0).reshape(-1, og)             # (kh kw cin_g) x og
            acc[:, :, g * og:(g + 1) * og] = patch @ wm
        acc = acc + P["bias"]
        if I.raw_out:
            assert np.abs(acc).max() < 2 ** 31, "raw output exceeds int32"
            return acc
        lo, hi = self._clamp(I)
        if I.epi == isa.EPI_ADD:   # residual: b * Mb joins before the rounding
            assert np.abs(acc).max(initial=0) < 2 ** 31, "accumulator exceeds int32"
            b = self.read(I.b_buf, I.b_base, I.o_h, I.o_w, I.o_c, I.b_pix, I.b_off)
            y = (acc * P["M"] + b * P["Mb"] + (np.int64(1) << (P["S"] - 1))) >> P["S"]
            return np.clip(y, lo, hi)
        return _requant(acc, P["M"], P["S"], lo, hi)

    @staticmethod
    def _clamp(I):
        """Output range: the type's, narrowed by the activation."""
        lo, hi = _range(I.o_bits, I.o_signed)
        if I.act == isa.ACT_RELU:
            lo = max(lo, 0)
        elif I.act == isa.ACT_RELU6:
            lo, hi = max(lo, 0), min(hi, I.act_max)
        return lo, hi

    def _pool_max(self, a, I):
        lo, _ = _range(I.a_bits, I.a_signed)
        win = self._windows(a, I, lo)   # padding never wins: it is the type minimum
        return win.max(axis=(2, 3))

    def step(self, I):
        a = self.read(I.a_buf, I.a_base, I.a_h, I.a_w, I.a_c, I.a_pix, I.a_off)
        if I.op == isa.OP_CONV:
            r = self._conv(a, I)
        elif I.op == isa.OP_POOL_MAX:
            r = self._pool_max(a, I)
        elif I.op == isa.OP_GAP:
            r = a.sum(axis=(0, 1), keepdims=True)
        else:
            raise NotImplementedError(f"op {I.op}")
        if I.epi == isa.EPI_INTERLEAVE:
            b = self.read(I.b_buf, I.b_base, I.o_h, I.o_w, I.o_c, I.b_pix, I.b_off)
            out = np.empty(r.shape[:2] + (2 * r.shape[2],), np.int64)
            out[..., 0::2] = b
            out[..., 1::2] = r
            r = out
        elif I.epi not in (isa.EPI_NONE, isa.EPI_ADD):
            raise NotImplementedError(f"epilogue {I.epi}")
        self.write(I.o_buf, I.o_base, r, I.o_pix, I.o_off)
        return r

    def run(self, prog, inp, trace=None):
        self.write(isa.BUF_IN, 0, inp.astype(np.int64), 0, 0)
        last = None
        for n, I in enumerate(prog):
            last = self.step(I)
            if trace is not None:
                trace.append(last)
        return last
