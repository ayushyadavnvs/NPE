"""ONNX front end: a quantized ONNX model (QDQ format) to an NPE program.

  python3 -m npe.frontend.onnx_import model.onnx [--list]

Input: the standard QDQ form that ONNX Runtime, Brevitas, the TFLite
converter (npe/frontend/tflite2onnx.py) and the NPE quantizer
(npe/frontend/ptq.py) write: QuantizeLinear / DequantizeLinear around float
ops, int8 weights (zero point 0, per tensor or per output channel), int32
biases.

Output: the same (prog, params, names, peak) as the hand-written lowerings,
plus the input and output quantization (Program).

Phase 1 walks the graph and builds layers on logical tensors:
  Conv [+ Add] [+ Relu / Clip]   -> dense conv, or depthwise (groups == C)
  MaxPool                        -> max pool (output scale must equal input's)
  GlobalAveragePool, ReduceMean over H, W, AveragePool over the whole map
                                 -> GAP: int32 sum into V, 1/(H*W) folded
                                    into the next layer's scale
  Gemm / MatMul [+ Add]          -> fc on V, raw int32 logits
  Split / Slice on channels      -> views (no instruction)
  Concat                         -> producers write into one tensor
  Concat + Reshape/Transpose/Reshape channel shuffle (2 groups)
                                 -> the INTERLEAVE epilogue
  Add of a conv output (single use) and another tensor -> ADD epilogue
  Flatten / Reshape / Squeeze to a vector, Softmax at the end: dropped
The conv output that feeds a fused Add is not requantized on its own: the
engine adds at full precision.

Phase 2 lays tensors out in the engine's words of P = 8 channels: a tensor
has a physical width (a multiple of 8) and a map from logical to physical
channel, so channel counts that are not multiples of 8 (ShuffleNet x1.0:
58, 116) still work. Channels the map does not use hold don't-care values;
every consumer reads them with zero weights. Then it assigns each tensor a
buffer (S0, A, B, C) and base by liveness, in place where an epilogue
allows it.

Activations: zero point 0 in the engine. uint8 with zero point 128 and int8
with zero point -128 are the same numbers shifted, so they map exactly. The
graph input may have any zero point (per channel too): it becomes constant
padding plus a bias correction, exact. Any other zero point is widened to a
symmetric int8 range that covers it (reported in Program.notes).

Requant: M, S per output channel from the float scales, M in [2^30, 2^31).
A model may carry exact integers instead in metadata "npe.requant" (JSON,
node name -> {"M": [...], "S": [...]}, logical channels), as the NPE's own
exporters do.
"""
import argparse
import dataclasses
import json
import math

import numpy as np
import onnx
from onnx import numpy_helper

from npe.compiler import isa
from npe.compiler.isa import Instr

P = 8
HW_BUFS = [isa.BUF_S0, isa.BUF_A, isa.BUF_B, isa.BUF_C]


class Unsupported(Exception):
    pass


def rup(n, p=P):
    return -(-n // p) * p


def multiplier(m):
    """m >= 0 -> (M, S), m ~= M * 2^-S, M in [2^30, 2^31) (M = 0 for m = 0)."""
    if m <= 0:
        return 0, 1
    S = 30 - int(math.floor(math.log2(m)))
    if S > 63:
        return int(round(m * 2.0 ** 63)), 63
    M = int(round(m * 2.0 ** S))
    if M == 2 ** 31:
        M, S = M // 2, S - 1
    if S < 1:
        raise Unsupported(f"requant multiplier {m} too large")
    return M, S


def multiplier2(ma, mb):
    """Shared shift for the ADD epilogue: (M, Mb, S), both below 2^31."""
    top = max(ma, mb)
    if top <= 0:
        return 0, 0, 1
    S = min(63, 30 - int(math.floor(math.log2(top))))
    while True:
        M, Mb = int(round(ma * 2.0 ** S)), int(round(mb * 2.0 ** S))
        if max(M, Mb) < 2 ** 31:
            return M, Mb, S
        S -= 1


# ---------------------------------------------------------------- IR -----
@dataclasses.dataclass(eq=False)
class T:
    """A quantized activation tensor (logical channels), or a view of one."""
    name: str
    c: int
    h: int
    w: int
    scale: float = 1.0
    signed: bool = False
    bits: int = 8
    zp: object = 0              # graph input only: scalar or per-channel array
    producer: object = None     # Layer
    root: object = None         # view: the tensor it is a view of
    idx: object = None          # view: logical channels of the root
    # phase 2: format and storage
    width: int = 0              # physical channels of this view
    perm: object = None         # logical -> physical channel within the view
    lo: int = 0                 # view: physical offset of the window in root
    grp: object = None          # Storage (roots)
    off: int = 0                # roots: channel offset inside the pixel
    pix: int = 0                # roots: pixel stride
    onnx_q: tuple = None        # (QuantizeLinear output, scale, zero point), for checks
    onnx_idx: object = None     # logical channel j = channel onnx_idx[j] of that output

    def base_root(self):
        return self.root or self


@dataclasses.dataclass(eq=False)
class Layer:
    kind: str                   # conv, dw, pool, gap, fc, concat, ishuffle
    name: str
    x: object = None            # input tensor (view)
    out: object = None
    attrs: dict = dataclasses.field(default_factory=dict)
    w: object = None            # int weights (logical), float scales
    w_scale: object = None
    b: object = None            # bias, float (real units)
    act: int = isa.ACT_NONE
    act_hi: float = None        # Clip upper bound (real)
    add: object = None          # ADD epilogue operand
    parts: list = None          # concat parts
    epi: int = isa.EPI_NONE
    ib: object = None           # INTERLEAVE: B operand view
    t: int = -1                 # instruction index


@dataclasses.dataclass(eq=False)
class Storage:
    size: int                   # elements
    buf: int = -1
    base: int = 0
    first: int = 1 << 30
    last: int = -1


@dataclasses.dataclass
class Program:
    prog: list
    params: list
    names: list
    peak: dict
    in_shape: tuple             # H, W, C of the engine input
    in_signed: bool
    in_offset: int              # engine input = ONNX quantized input + in_offset
    in_scale: object
    in_zp: object
    out_scale: np.ndarray       # real logit = raw int32 * out_scale
    notes: list


class Pending:
    """A float value not yet requantized: conv accumulator (+ add, + act)."""
    def __init__(self, layer):
        self.layer = layer


class GapVal:
    def __init__(self, t):
        self.t = t


class Shuffle:
    def __init__(self, t, stage, groups):
        self.t, self.stage, self.groups = t, stage, groups


# ------------------------------------------------------------ phase 1 ----
class Importer:
    def __init__(self, model):
        model = onnx.shape_inference.infer_shapes(model)
        self.g = g = model.graph
        self.const = {t.name: numpy_helper.to_array(t) for t in g.initializer}
        for n in g.node:
            if n.op_type == "Constant":
                a = n.attribute[0]
                self.const[n.output[0]] = (numpy_helper.to_array(a.t) if a.name == "value"
                                           else np.array(onnx.helper.get_attribute_value(a)))
        self.shape = {}
        for vi in list(g.value_info) + list(g.input) + list(g.output):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            self.shape[vi.name] = dims
        meta = {p.key: p.value for p in model.metadata_props}
        self.requant = json.loads(meta.get("npe.requant", "{}"))
        self.users = {}
        for n in g.node:
            for i in n.input:
                self.users.setdefault(i, []).append(n)
        self.val = {}
        self.layers = []
        self.notes = []
        self.widened = {}
        self.inp = None
        self.out_layer = None

    # helpers --------------------------------------------------------------
    @staticmethod
    def attrs(n):
        return {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}

    def qparams(self, n):
        s = self.const[n.input[1]].astype(np.float64)
        z = self.const[n.input[2]] if len(n.input) > 2 and n.input[2] else np.zeros((), np.uint8)
        return s, z

    def engine_type(self, s, z, name):
        """(scale, signed) of an engine tensor for ONNX quantization (s, z)."""
        s, zi = float(np.asarray(s).reshape(-1)[0]), np.asarray(z)
        if np.asarray(s).size != 1 or zi.size != 1:
            raise Unsupported(f"{name}: per-channel activation quantization")
        z0, dt = int(zi), zi.dtype
        u = dt == np.uint8
        if (u and z0 == 0) or (not u and z0 == -128):
            return s, False
        if (u and z0 == 128) or (not u and z0 == 0):
            return s, True
        lo, hi = (0, 255) if u else (-128, 127)
        s2 = s * max(hi - z0, z0 - lo) / 127.0
        self.widened[name] = f"zero point {z0} widened to symmetric int8 (scale x{s2 / s:.3f})"
        return s2, True

    def hwc(self, name):
        d = self.shape.get(name)
        if not d or len(d) != 4:
            raise Unsupported(f"{name}: need a known NCHW shape, got {d}")
        return d[2], d[3], d[1]

    def wconst(self, name):
        """Weight or bias input: (integer array or None, scale array, float array)."""
        v = self.val.get(name)
        if isinstance(v, tuple):
            return v
        if name in self.const:
            f = self.const[name].astype(np.float64)
            return None, None, f
        raise Unsupported(f"{name}: weights must be constants")

    def get(self, name):
        if name not in self.val:
            raise Unsupported(f"{name}: produced by an unsupported op")
        return self.val[name]

    def qt(self, name):
        v = self.get(name)
        if isinstance(v, GapVal):
            return v.t
        if not isinstance(v, T):
            raise Unsupported(f"{name}: expected a quantized tensor, got {type(v).__name__}")
        return v

    def view(self, t, idx, name):
        r = t.base_root()
        ridx = np.asarray(idx) if t.root is None else t.idx[np.asarray(idx)]
        return T(name, len(ridx), t.h, t.w, t.scale, t.signed, t.bits, t.zp, root=r, idx=ridx)

    # walk -----------------------------------------------------------------
    def run(self):
        g = self.g
        inputs = [i for i in g.input if i.name not in self.const]
        if len(inputs) != 1:
            raise Unsupported("need exactly one graph input")
        self.in_name = inputs[0].name
        for n in g.node:
            op = n.op_type
            fn = getattr(self, "op_" + op, None)
            if fn is None:
                raise Unsupported(f"node {n.name or n.output[0]}: op {op} not supported")
            fn(n)
        outs = [o.name for o in g.output]
        if self.out_layer is None:
            raise Unsupported("the graph must end in a fully connected layer")
        live = {L.out.name for L in self.layers if L.out is not None}
        self.notes += [f"{k}: {v}" for k, v in self.widened.items() if k in live]
        return self

    def op_Constant(self, n):
        pass

    def op_QuantizeLinear(self, n):
        x, (s, z) = n.input[0], self.qparams(n)
        if x == self.in_name:     # the graph input
            h, w, c = self.hwc(x)
            sa, za = np.asarray(s).reshape(-1), np.asarray(z).reshape(-1)
            if not np.allclose(sa, sa[0], rtol=1e-6):
                raise Unsupported("input: per-channel scales differ")
            u = za.dtype == np.uint8
            zz = za.astype(np.int64)
            if not u and np.all(zz == -128):
                signed, zz, off = False, np.zeros_like(zz), 128
            elif u and np.all(zz == 128):
                signed, zz, off = True, np.zeros_like(zz), -128
            else:
                signed, off = not u, 0
            t = T("input", c, h, w, float(sa[0]), signed, zp=zz if np.any(zz) else 0)
            t.zp = np.broadcast_to(zz, (c,)).copy() if np.any(zz) else 0
            self.inp = t
            self.in_info = (s, z, off)
            self.val[n.output[0]] = t
            return
        v = self.get(x)
        if isinstance(v, Pending):
            L = v.layer
            sc, signed = self.engine_type(s, z, n.output[0])
            h, w, c = self.hwc(n.output[0]) if L.kind != "fc" else (1, 1, L.w.shape[0])
            t = T(n.output[0], c, h, w, sc, signed, producer=L)
            t.onnx_q = (n.output[0], float(np.asarray(s).reshape(-1)[0]), int(np.asarray(z).reshape(-1)[0]))
            L.out = t
            if L.kind != "fc":
                self.layers.append(L)
            self.val[n.output[0]] = t
        elif isinstance(v, (T, GapVal)):
            t = v.t if isinstance(v, GapVal) else v
            if t.bits == 32:               # GAP sum: stays int32 (more precise)
                self.val[n.output[0]] = v
                return
            sc, signed = self.engine_type(s, z, n.output[0])
            if abs(sc - t.scale) > 1e-6 * t.scale or signed != t.signed:
                raise Unsupported(f"{n.output[0]}: requantization without a compute op "
                                  f"({t.scale} -> {sc})")
            if t.root is None:
                t.onnx_idx = None
                t.onnx_q = (n.output[0], float(np.asarray(s).reshape(-1)[0]), int(np.asarray(z).reshape(-1)[0]))
            self.val[n.output[0]] = t
        else:
            raise Unsupported(f"{n.output[0]}: quantize of {type(v).__name__}")

    def op_DequantizeLinear(self, n):
        x = n.input[0]
        if x in self.const:
            q = self.const[x].astype(np.int64)
            s, z = self.qparams(n)
            z = np.asarray(z).astype(np.int64)
            ax = self.attrs(n).get("axis", 1)
            if np.any(z != 0):
                raise Unsupported(f"{x}: weights need zero point 0")
            sh = [1] * q.ndim
            if s.size > 1:
                sh[ax] = -1
            self.val[n.output[0]] = (q, s, q * s.reshape(sh))
            return
        self.val[n.output[0]] = self.get(x)

    def _conv_like(self, n, x, wq, ws, wf, b, attrs, kind):
        kh, kw = wf.shape[2], wf.shape[3]
        h, w, c = x.h, x.w, x.c
        ho, wo = self.hwc(n.output[0])[:2]
        st = attrs.get("strides", [1, 1])
        if any(d != 1 for d in attrs.get("dilations", [1, 1])):
            raise Unsupported(f"{n.name}: dilation")
        pads = attrs.get("pads")
        ap = attrs.get("auto_pad", b"NOTSET")
        ap = ap.decode() if isinstance(ap, bytes) else ap
        if pads is None:
            pads = [0, 0, 0, 0]
            if ap in ("SAME_UPPER", "SAME_LOWER"):
                tot = [max((o - 1) * s + k - i, 0) for o, s, k, i in
                       ((ho, st[0], kh, h), (wo, st[1], kw, w))]
                lo = [t // 2 if ap == "SAME_UPPER" else t - t // 2 for t in tot]
                pads = [lo[0], lo[1], tot[0] - lo[0], tot[1] - lo[1]]
        if wq is None:   # float weights: quantize per output channel
            ws = np.maximum(np.abs(wf).reshape(wf.shape[0], -1).max(axis=1), 1e-12) / 127.0
            wq = np.clip(np.round(wf / ws[:, None, None, None]), -127, 127).astype(np.int64)
            self.notes.append(f"{n.name}: float weights quantized per channel")
        ws = np.broadcast_to(np.asarray(ws, np.float64).reshape(-1), (wf.shape[0],)).copy()
        L = Layer(kind, n.name or n.output[0], x=x, w=wq, w_scale=ws, b=b,
                  attrs=dict(kh=kh, kw=kw, sh=st[0], sw=st[1], pad_t=pads[0], pad_l=pads[1],
                             pad_b=pads[2], pad_r=pads[3]))
        return L

    def op_Conv(self, n):
        x = self.qt(n.input[0])
        wq, ws, wf = self.wconst(n.input[1])
        b = self.wconst(n.input[2])[2] if len(n.input) > 2 and n.input[2] else None
        a = self.attrs(n)
        g = a.get("group", 1)
        if g == 1:
            kind = "conv"
        elif g == x.c == wf.shape[0]:
            kind = "dw"
        else:
            raise Unsupported(f"{n.name}: grouped conv (groups {g}) is not dense or depthwise")
        self.val[n.output[0]] = Pending(self._conv_like(n, x, wq, ws, wf, b, a, kind))

    def _fc(self, n, x, wq, ws, wf, b):
        """wf: [out, in] float."""
        if x.bits != 32:   # an 8-bit 1x1 tensor: GAP over 1x1 moves it to V
            if x.h * x.w != 1:
                raise Unsupported(f"{n.name}: fully connected on a {x.h}x{x.w} map")
            x = self._gap(n.name + ".to_v", x)
        if wq is None:
            ws = np.full(wf.shape[0], max(np.abs(wf).max(), 1e-12) / 127.0)
            wq = np.clip(np.round(wf / ws[:, None]), -127, 127).astype(np.int64)
        ws = np.broadcast_to(np.asarray(ws, np.float64).reshape(-1), (wf.shape[0],)).copy()
        L = Layer("fc", n.name or n.output[0], x=x, w=wq.reshape(wq.shape[0], -1, 1, 1),
                  w_scale=ws, b=b)
        return L

    def op_Gemm(self, n):
        a = self.attrs(n)
        if a.get("transA", 0) or a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0:
            raise Unsupported(f"{n.name}: Gemm attributes {a}")
        x = self.qt(n.input[0])
        wq, ws, wf = self.wconst(n.input[1])
        if not a.get("transB", 0):
            wf = wf.T
            wq = None if wq is None else wq.T
        b = self.wconst(n.input[2])[2] if len(n.input) > 2 and n.input[2] else None
        L = self._fc(n, x, wq, ws, wf, b)
        self.val[n.output[0]] = Pending(L)
        self.out_layer = L
        self.layers.append(L)
        L.out = T(n.output[0], wf.shape[0], 1, 1, bits=32, signed=True, producer=L)

    def op_MatMul(self, n):
        x = self.qt(n.input[0])
        wq, ws, wf = self.wconst(n.input[1])
        L = self._fc(n, x, None if wq is None else wq.T, ws, wf.T, None)
        self.val[n.output[0]] = Pending(L)
        self.out_layer = L
        self.layers.append(L)
        L.out = T(n.output[0], wf.shape[1], 1, 1, bits=32, signed=True, producer=L)

    def op_Add(self, n):
        va, vb = (self.val.get(i, self.const.get(i)) for i in n.input)
        if isinstance(vb, Pending) or (isinstance(vb, tuple) or isinstance(vb, np.ndarray)) and \
                not isinstance(va, (Pending, tuple, np.ndarray)):
            va, vb = vb, va
        if isinstance(va, Pending) and va.layer.kind == "fc":       # fc bias
            bf = vb[2] if isinstance(vb, tuple) else np.asarray(vb, np.float64)
            L = va.layer
            L.b = bf.reshape(-1) + (0 if L.b is None else L.b)
            self.val[n.output[0]] = va
            return
        # residual: fuse into the conv producing one side
        def fusable(v):
            if isinstance(v, Pending):
                L = v.layer
                return L if L.kind == "conv" and L.add is None and L.act == isa.ACT_NONE else None
            if isinstance(v, T) and v.root is None and v.producer is not None:
                L = v.producer
                if (L.kind == "conv" and L.add is None and L.act == isa.ACT_NONE
                        and L.epi == isa.EPI_NONE and len(self.users.get(self._qname(v), [])) == 1
                        and self._consumers_of(v) == 1):
                    return L
            return None
        cand = [(fusable(v), o) for v, o in ((va, vb), (vb, va))]
        cand = [(L, o) for L, o in cand if L is not None and isinstance(o, T)]
        if not cand:
            raise Unsupported(f"{n.name}: Add needs a single-use conv output on one side")
        # the later conv, so the other operand exists when it runs
        L, other = max(cand, key=lambda c: self.layers.index(c[0]) if c[0] in self.layers else 1 << 30)
        if L in self.layers:
            self.layers.remove(L)      # re-added when the sum is requantized
            L.out = None
        L.add = other
        self.val[n.output[0]] = Pending(L)

    def _qname(self, t):
        return t.name

    def _consumers_of(self, t):
        """Number of ONNX consumers of a quantized tensor (via its DQ nodes)."""
        cnt = 0
        for dq in self.users.get(t.name, []):
            if dq.op_type == "DequantizeLinear":
                cnt += len(self.users.get(dq.output[0], []))
            else:
                cnt += 1
        return cnt

    def op_Relu(self, n):
        v = self.get(n.input[0])
        if not isinstance(v, Pending):
            raise Unsupported(f"{n.name}: Relu must follow a conv or add")
        v.layer.act = isa.ACT_RELU
        self.val[n.output[0]] = v

    def op_Clip(self, n):
        v = self.get(n.input[0])
        a = self.attrs(n)
        lo = a.get("min", None)
        hi = a.get("max", None)
        if len(n.input) > 1 and n.input[1]:
            lo = float(np.asarray(self.const[n.input[1]]).reshape(-1)[0])
        if len(n.input) > 2 and n.input[2]:
            hi = float(np.asarray(self.const[n.input[2]]).reshape(-1)[0])
        if not isinstance(v, Pending) or lo is None or lo != 0.0:
            raise Unsupported(f"{n.name}: Clip must be [0, max] after a conv or add")
        v.layer.act = isa.ACT_RELU6 if hi is not None else isa.ACT_RELU
        v.layer.act_hi = hi
        self.val[n.output[0]] = v

    def op_MaxPool(self, n):
        x = self.qt(n.input[0])
        a = self.attrs(n)
        if a.get("ceil_mode", 0) or any(d != 1 for d in a.get("dilations", [1, 1])):
            raise Unsupported(f"{n.name}: MaxPool ceil_mode / dilation")
        k = a["kernel_shape"]
        st = a.get("strides", [1, 1])
        pads = a.get("pads", [0, 0, 0, 0])
        L = Layer("pool", n.name or n.output[0], x=x,
                  attrs=dict(kh=k[0], kw=k[1], sh=st[0], sw=st[1], pad_t=pads[0],
                             pad_l=pads[1], pad_b=pads[2], pad_r=pads[3]))
        h, w, c = self.hwc(n.output[0])
        t = T(n.output[0], c, h, w, x.scale, x.signed, producer=L)
        L.out = t
        self.layers.append(L)
        self.val[n.output[0]] = t

    def _gap(self, name, x):
        L = Layer("gap", name, x=x)
        t = T(name, x.c, 1, 1, x.scale / (x.h * x.w), True, 32, producer=L)
        L.out = t
        self.layers.append(L)
        return t

    def op_GlobalAveragePool(self, n):
        self.val[n.output[0]] = GapVal(self._gap(n.name or n.output[0], self.qt(n.input[0])))

    def op_ReduceMean(self, n):
        a = self.attrs(n)
        axes = a.get("axes") or (list(self.const[n.input[1]]) if len(n.input) > 1 else None)
        if sorted(int(v) % 4 for v in axes) != [2, 3]:
            raise Unsupported(f"{n.name}: ReduceMean over {axes}")
        self.val[n.output[0]] = GapVal(self._gap(n.name or n.output[0], self.qt(n.input[0])))

    def op_AveragePool(self, n):
        x = self.qt(n.input[0])
        a = self.attrs(n)
        if list(a["kernel_shape"]) != [x.h, x.w] or any(a.get("pads", [0] * 4)):
            raise Unsupported(f"{n.name}: AveragePool that is not global")
        self.val[n.output[0]] = GapVal(self._gap(n.name or n.output[0], x))

    def _passthrough(self, n):
        self.val[n.output[0]] = self.get(n.input[0])

    def op_Flatten(self, n):
        v = self.get(n.input[0])
        t = v.t if isinstance(v, GapVal) else v
        if isinstance(t, T) and t.h * t.w != 1 and t.bits != 32:
            raise Unsupported(f"{n.name}: Flatten of a {t.h}x{t.w} map")
        self.val[n.output[0]] = v

    op_Squeeze = op_Flatten
    op_Unsqueeze = _passthrough
    op_Identity = _passthrough

    def op_Softmax(self, n):
        self._passthrough(n)

    def op_Shape(self, n):
        self.val[n.output[0]] = "shape"

    def op_Gather(self, n):
        self.val[n.output[0]] = "shape"

    def op_Reshape(self, n):
        v = self.get(n.input[0])
        shp = self.shape.get(n.output[0]) or list(self.const.get(n.input[1], []))
        if isinstance(v, Shuffle) and v.stage == 2 and len(shp) == 4:
            self.val[n.output[0]] = self._shuffle(v.t, v.groups, n.output[0])
            return
        t = v.t if isinstance(v, GapVal) else v
        if isinstance(t, T) and len(shp) == 5 and shp[1] * shp[2] == t.c:
            self.val[n.output[0]] = Shuffle(t, 1, shp[1])
            return
        if isinstance(t, T) and (t.h * t.w == 1 or t.bits == 32) and len(shp) in (2, 4):
            self.val[n.output[0]] = v
            return
        raise Unsupported(f"{n.name}: Reshape to {shp}")

    def op_Transpose(self, n):
        v = self.get(n.input[0])
        if isinstance(v, Shuffle) and v.stage == 1 and list(self.attrs(n)["perm"]) == [0, 2, 1, 3, 4]:
            self.val[n.output[0]] = Shuffle(v.t, 2, v.groups)
            return
        raise Unsupported(f"{n.name}: Transpose")

    def _shuffle(self, t, g, name):
        """Channel shuffle: out channel j*g + i = in channel i*(C/g) + j."""
        L = t.producer if t.root is None else None
        if g == 2 and L is not None and L.kind == "concat" and len(L.parts) == 2 \
                and L.parts[0].c == L.parts[1].c and self._consumers_of(t) <= 1:
            c = t.c
            if t.onnx_q is not None:   # checks: its ONNX tensor is in concat order
                t.onnx_idx = np.arange(c).reshape(g, c // g).T.reshape(-1)
            L.kind = "ishuffle"      # concat + shuffle: the INTERLEAVE epilogue
            return t
        c = t.c
        idx = np.arange(c).reshape(g, c // g).T.reshape(-1)
        return self.view(t, idx, name)

    def op_Split(self, n):
        x = self.qt(n.input[0])
        a = self.attrs(n)
        if a.get("axis", 0) != 1:
            raise Unsupported(f"{n.name}: Split on axis {a.get('axis')}")
        sp = a.get("split")
        if sp is None and len(n.input) > 1 and n.input[1]:
            sp = list(self.const[n.input[1]])
        if sp is None:
            sp = [x.c // len(n.output)] * len(n.output)
        o = 0
        for name, k in zip(n.output, sp):
            self.val[name] = self.view(x, np.arange(o, o + k), name)
            o += k

    def op_Slice(self, n):
        x = self.qt(n.input[0])
        st, en = int(self.const[n.input[1]][0]), int(self.const[n.input[2]][0])
        axes = list(self.const[n.input[3]]) if len(n.input) > 3 and n.input[3] else [0]
        steps = list(self.const[n.input[4]]) if len(n.input) > 4 and n.input[4] else [1]
        if len(axes) != 1 or int(axes[0]) % 4 != 1 or steps[0] != 1:
            raise Unsupported(f"{n.name}: Slice that is not a channel range")
        en = min(en, x.c)
        self.val[n.output[0]] = self.view(x, np.arange(st, en), n.output[0])

    def op_Concat(self, n):
        if self.attrs(n).get("axis") != 1:
            if all(self.val.get(i) == "shape" or i in self.const for i in n.input):
                self.val[n.output[0]] = "shape"
                return
            raise Unsupported(f"{n.name}: Concat on axis {self.attrs(n).get('axis')}")
        parts = [self.qt(i) for i in n.input]
        s0 = parts[0].scale
        for p in parts:
            if abs(p.scale - s0) > 1e-6 * s0 or p.signed != parts[0].signed:
                raise Unsupported(f"{n.name}: Concat inputs with different scales "
                                  f"({[p.scale for p in parts]}); quantize them with one scale")
        h, w, c = self.hwc(n.output[0])
        L = Layer("concat", n.name or n.output[0], parts=parts)
        t = T(n.output[0], c, h, w, s0, parts[0].signed, producer=L)
        L.out = t
        self.layers.append(L)
        self.val[n.output[0]] = t
        # concat output is quantized with the same scale: the Q after it is a no-op

    def op_BatchNormalization(self, n):
        raise Unsupported(f"{n.name}: fold BatchNormalization before quantizing (ptq.py does)")


# ------------------------------------------------------------ phase 2 ----
class Lowering:
    def __init__(self, imp):
        self.imp = imp
        self.layers = imp.layers
        self.prog, self.params, self.names = [], [], []

    def window(self, v, need=None):
        """Format of a view: physical window of the root that holds its channels."""
        r = v.root
        p = r.perm[v.idx]
        lo = (int(p.min()) // P) * P
        hi = rup(int(p.max()) + 1)
        if need is not None and hi - lo != need:
            if lo + need <= r.width and hi - lo <= need:
                hi = lo + need
            else:
                raise Unsupported(f"{v.name}: view does not fit a {need}-channel window")
        v.lo, v.width, v.perm = lo, hi - lo, p - lo
        return v

    def fmt(self, t, need=None):
        if t.root is not None:
            if t.perm is None or (need is not None and t.width != need):
                self.window(t, need)
        return t

    def run(self):
        L_all = self.layers
        inp = self.imp.inp
        inp.width, inp.perm = inp.c, np.arange(inp.c)
        # consumers (layer list) of each root
        self.readers = {}
        for L in L_all:
            for v in self.reads(L):
                self.readers.setdefault(v.base_root(), []).append(L)
        # instruction order: layers that emit (concat pseudo-layers do not)
        t = 0
        for L in L_all:
            if L.kind not in ("concat", "ishuffle"):
                L.t = t
                t += 1
        # formats, in order
        for L in L_all:
            self.format(L)
        self.place()
        self.allocate()
        for L in L_all:
            self.emit(L)
        return self

    def reads(self, L):
        if L.kind in ("concat", "ishuffle"):
            return []
        r = [L.x]
        if L.add is not None:
            r.append(L.add)
        if L.ib is not None:
            r.append(L.ib)
        return r

    def format(self, L):
        o = L.out
        if L.kind in ("conv", "fc"):
            self.fmt(L.x)
            o.width = rup(o.c) if L.kind == "conv" else o.c
            o.perm = np.arange(o.c)
            if L.add is not None:
                a = self.fmt(L.add, o.width)
                if not np.array_equal(a.perm, o.perm):
                    raise Unsupported(f"{L.name}: residual operand has a different channel order")
        elif L.kind in ("dw", "pool", "gap"):
            x = self.fmt(L.x)
            o.width, o.perm = x.width, x.perm.copy()
        elif L.kind == "concat":
            off, perm = 0, []
            for p in L.parts:
                self._own_part(L, p)
                perm.append(p.perm + off)
                off += p.width
            o.width, o.perm = off, np.concatenate(perm)
        elif L.kind == "ishuffle":
            p1, p2 = L.parts
            C2 = p2.producer
            if p2.root is not None or C2 is None or C2.kind != "conv" or C2.add is not None \
                    or len(self.readers.get(p2, [])) != 0:
                raise Unsupported(f"{L.name}: the second concat input must come straight "
                                  "from a conv used only by the concat")
            H = p2.width
            self.fmt(p1, H)
            if p1.root is None:
                if p1.width != H:
                    raise Unsupported(f"{L.name}: concat halves of different widths")
            o.width = 2 * H
            o.perm = np.empty(o.c, np.int64)
            o.perm[0::2] = 2 * p1.perm
            o.perm[1::2] = 2 * p2.perm + 1
            C2.epi, C2.ib = isa.EPI_INTERLEAVE, p1
            C2.out_full = o
            # p1 is read by the INTERLEAVE conv
            self.readers.setdefault(p1.base_root(), []).append(C2)

    def _own_part(self, L, p):
        if p.root is not None or p.producer is None or p.producer.kind not in ("conv", "dw", "pool") \
                or self.readers.get(p):
            raise Unsupported(f"{L.name}: concat input {p.name} must come straight from a "
                              "conv / pool used only by the concat")

    # storage --------------------------------------------------------------
    def place(self):
        """Storage groups, with the aliasing the epilogues and concat allow."""
        self.groups = []

        def new(t, size_pix=None):
            t.pix = size_pix or t.width
            t.grp = Storage(t.h * t.w * t.pix)
            t.off = 0
            self.groups.append(t.grp)

        for L in self.layers:
            o = L.out
            if L.kind == "gap":
                o.grp, o.pix, o.off = Storage(o.width, buf=isa.BUF_V), o.width, 0
            elif L.kind == "fc":
                o.grp, o.pix, o.off = Storage(o.c, buf=isa.BUF_OUT), o.c, 0
            elif L.kind == "concat":
                new(o)
                off = 0
                for p in L.parts:
                    p.grp, p.pix, p.off = o.grp, o.pix, off
                    off += p.width
            elif L.kind == "ishuffle":
                p1, p2 = L.parts
                C2 = p2.producer
                H = p2.width
                r = p1.base_root()
                last_other = max([R.t for R in self.readers.get(r, []) if R is not C2], default=-1)
                if r.grp is not None and r.pix == 2 * H and (r.off + p1.lo if p1.root else r.off) == r.off \
                        and (p1.lo if p1.root else 0) == 0 and last_other < C2.t \
                        and r.grp.buf not in (isa.BUF_IN, isa.BUF_V, isa.BUF_OUT):
                    o.grp, o.pix, o.off = r.grp, r.pix, r.off       # in place over p1's tensor
                elif p1.root is None and p1.producer is not None and \
                        p1.producer.kind in ("conv", "dw", "pool") and \
                        self.readers.get(p1) == [C2]:
                    new(o)
                    p1.grp, p1.pix, p1.off = o.grp, o.pix, 0          # p1 written into place
                else:
                    new(o)
                p2.grp, p2.pix, p2.off = o.grp, o.pix, 0              # never stored on its own
            elif o is not None and o.grp is None:
                if L.add is not None:
                    r = L.add.base_root()
                    last_other = max([R.t for R in self.readers.get(r, []) if R is not L], default=-1)
                    if L.add.root is None and r.grp is not None and last_other < L.t and \
                            r.pix == o.width and r.grp.buf == -1:
                        o.grp, o.pix, o.off = r.grp, r.pix, r.off    # in place over B
                        continue
                new(o)

    def allocate(self):
        imp = self.imp
        inp = imp.inp
        inp.grp, inp.pix, inp.off = Storage(inp.h * inp.w * inp.c, buf=isa.BUF_IN), inp.c, 0
        # liveness
        for L in self.layers:
            if L.t < 0:
                continue
            w = L.out.grp if L.out is not None else None
            if L.epi == isa.EPI_INTERLEAVE:
                w = L.out_full.grp
            if w is not None:
                w.first = min(w.first, L.t)
                w.last = max(w.last, L.t)
            for v in self.reads(L):
                gr = v.base_root().grp
                gr.first = min(gr.first, L.t)
                gr.last = max(gr.last, L.t)
        # different-buffer pairs
        avoid = {}
        for L in self.layers:
            if L.t < 0:
                continue
            o = (L.out_full if L.epi == isa.EPI_INTERLEAVE else L.out).grp
            a = L.x.base_root().grp
            pairs = [(a, o)]
            for b in (L.add, L.ib):
                if b is not None:
                    pairs.append((a, b.base_root().grp))
            for p, q in pairs:
                if p is not q:
                    avoid.setdefault(id(p), []).append(q)
                    avoid.setdefault(id(q), []).append(p)
        placed = {b: [] for b in HW_BUFS}
        peak = {b: 0 for b in HW_BUFS}
        for gr in sorted((g for g in self.groups if g.buf == -1 and g.last >= 0),
                         key=lambda g: g.first):
            bad = {q.buf for q in avoid.get(id(gr), []) if q.buf != -1}
            best = None
            for b in HW_BUFS:
                if b in bad:
                    continue
                live = sorted((q.base, q.base + q.size) for q in placed[b]
                              if not (q.last < gr.first or q.first > gr.last))
                base = 0
                for lo, hi in live:
                    if base + gr.size <= lo:
                        break
                    base = max(base, hi)
                cost = max(peak[b], base + gr.size) - peak[b]
                if best is None or (cost, base) < best[0]:
                    best = ((cost, base), b, base)
            _, gr.buf, gr.base = best
            placed[gr.buf].append(gr)
            peak[gr.buf] = max(peak[gr.buf], gr.base + gr.size)
        self.peak = {b: n for b, n in peak.items() if n}
        self.peak[isa.BUF_IN] = inp.h * inp.w * inp.c
        vs = [L.out.width for L in self.layers if L.kind == "gap"]
        if vs:
            self.peak[isa.BUF_V] = max(vs)
        self.peak[isa.BUF_OUT] = sum(L.out.c for L in self.layers if L.kind == "fc")

    # emission -------------------------------------------------------------
    @staticmethod
    def loc(v):
        """(buf, base, pix, off) of a view."""
        r = v.base_root()
        return r.grp.buf, r.grp.base, r.pix, r.off + (v.lo if v.root is not None else 0)

    def requant(self, L, m_scale, name):
        o = L.out
        ov = self.imp.requant.get(L.name)
        if ov is not None:
            return np.array(ov["M"], np.int64), np.array(ov["S"], np.int64)
        MS = [multiplier(float(m)) for m in m_scale]
        return np.array([a for a, _ in MS], np.int64), np.array([b for _, b in MS], np.int64)

    def emit(self, L):
        if L.t < 0:
            return
        x = L.x
        ab, abase, apix, aoff = self.loc(x)
        I = Instr(a_buf=ab, a_base=abase, a_h=x.h, a_w=x.w, a_c=x.width, a_pix=apix, a_off=aoff,
                  a_signed=int(x.signed), a_bits=x.bits)
        o = L.out
        oo = L.out_full if L.epi == isa.EPI_INTERLEAVE else o
        ob, obase, opix, ooff = self.loc(oo)
        if L.epi == isa.EPI_INTERLEAVE:
            ooff = oo.off
        I.o_buf, I.o_base, I.o_pix, I.o_off = ob, obase, opix, ooff
        I.o_h, I.o_w = o.h, o.w
        I.o_signed, I.o_bits = int(o.signed), o.bits
        for k, v in L.attrs.items():
            setattr(I, k, v)
        if L.kind == "pool":
            I.op, I.o_c = isa.OP_POOL_MAX, x.width
        elif L.kind == "gap":
            I.op, I.o_c, I.o_h, I.o_w = isa.OP_GAP, x.width, 1, 1
        else:
            self.conv_params(L, I, x, o)
        self.prog.append(I)
        self.names.append(L.name)

    def conv_params(self, L, I, x, o):
        imp = self.imp
        dw = L.kind == "dw"
        fc = L.kind == "fc"
        wq = L.w.astype(np.int64)
        cout = wq.shape[0]
        kh, kw = wq.shape[2], wq.shape[3]
        # integer bias (accumulator units) and requant, logical channels
        acc_scale = L.w_scale * x.scale
        bias = np.zeros(cout, np.int64) if L.b is None else np.round(L.b / acc_scale).astype(np.int64)
        if np.any(np.asarray(x.zp) != 0):        # input zero point: fold into the bias
            z = np.broadcast_to(np.asarray(x.zp, np.int64), (x.c,))
            if dw:
                bias = bias - wq.reshape(cout, -1).sum(1) * z
            else:
                bias = bias - (wq.sum(axis=(2, 3)) * z[None, :]).sum(1)
        I.op = isa.OP_CONV
        if fc:
            width = x.width
            w = np.zeros((cout, width, 1, 1), np.int64)
            w[:, x.perm] = wq
            I.o_c, I.raw_out, I.a_signed = cout, 1, 1
            par = dict(w=w, bias=bias, M=np.zeros(cout, np.int64), S=np.ones(cout, np.int64))
            self.out_scale = acc_scale
        else:
            if dw:
                width = x.width
                w = np.zeros((width, 1, kh, kw), np.int64)
                w[x.perm] = wq
                ow, operm = width, x.perm
                I.groups = width
            else:
                ow, operm = o.width, o.perm
                w = np.zeros((ow, x.width, kh, kw), np.int64)
                w[np.ix_(operm, x.perm)] = wq
            I.o_c = ow
            m = acc_scale / o.scale
            par = dict(w=w)
            full = lambda a, fill: self._scatter(a, operm, ow, fill)   # noqa: E731
            if L.add is not None:
                I.epi = isa.EPI_ADD
                b = L.add
                bb, bbase, bpix, boff = self.loc(b)
                I.b_buf, I.b_base, I.b_pix, I.b_off, I.b_signed = bb, bbase, bpix, boff, int(b.signed)
                trip = [multiplier2(float(ma), b.scale / o.scale) for ma in m]
                par.update(M=full(np.array([t[0] for t in trip]), 0),
                           Mb=full(np.array([t[1] for t in trip]), 0),
                           S=full(np.array([t[2] for t in trip]), 1))
            else:
                M, S = self.requant(L, m, L.name)
                par.update(M=full(M, 0), S=full(S, 1))
            if L.epi == isa.EPI_INTERLEAVE:
                I.epi = isa.EPI_INTERLEAVE
                bb, bbase, bpix, boff = self.loc(L.ib)
                I.b_buf, I.b_base, I.b_pix, I.b_off = bb, bbase, bpix, boff
                I.b_signed = int(L.ib.signed)
            par["bias"] = full(bias, 0)
            if L.act != isa.ACT_NONE:
                I.act = L.act
                if L.act == isa.ACT_RELU6:
                    hi = (1 << o.bits) - 1 if not o.signed else (1 << (o.bits - 1)) - 1
                    q = int(round(L.act_hi / o.scale))
                    if q >= hi:
                        I.act = isa.ACT_RELU
                    else:
                        I.act_max = q
            if np.any(np.asarray(x.zp) != 0):
                zp = np.zeros(x.width, np.int64)
                zp[x.perm] = np.broadcast_to(np.asarray(x.zp, np.int64), (x.c,))
                if any(L.attrs.get(k) for k in ("pad_t", "pad_l", "pad_b", "pad_r")):
                    I.pad_mode = isa.PAD_CONST
                    par["pad"] = zp
        I.w_base = len(self.params)
        self.params.append(par)

    @staticmethod
    def _scatter(a, perm, width, fill):
        out = np.full(width, fill, np.int64)
        out[perm] = a
        return out


def import_model(model):
    """ONNX ModelProto or path -> Program."""
    if isinstance(model, str):
        model = onnx.load(model)
    imp = Importer(model).run()
    low = Lowering(imp).run()
    s, z, off = imp.in_info
    return Program(low.prog, low.params, low.names, low.peak,
                   (imp.inp.h, imp.inp.w, imp.inp.c), imp.inp.signed, off, s, z,
                   np.asarray(low.out_scale), imp.notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    p = import_model(a.model)
    print(f"{len(p.prog)} instructions; input {p.in_shape} "
          f"{'int8' if p.in_signed else 'uint8'}; buffer peaks (elements): "
          f"{ {isa.BUF_NAMES[b]: n for b, n in sorted(p.peak.items())} }")
    for n in p.notes:
        print("note:", n)
    if a.list:
        for n, I in zip(p.names, p.prog):
            print(f"{n[:28]:28s} op={I.op} epi={I.epi} act={I.act} k={I.kh}x{I.kw} s={I.sh} "
                  f"g={I.groups} {isa.BUF_NAMES[I.a_buf]}{I.a_h}x{I.a_w}x{I.a_c}+{I.a_off}/"
                  f"{I.a_pix} -> {isa.BUF_NAMES[I.o_buf]}{I.o_h}x{I.o_w}x{I.out_channels()}"
                  f"+{I.o_off}/{I.o_pix}")


if __name__ == "__main__":
    main()
