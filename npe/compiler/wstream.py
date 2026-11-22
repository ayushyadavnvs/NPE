"""Which layers stream their weights from DRAM (npe/hls/npe_wstream.h).

  none  everything on chip (the default; small models)
  all   every streamable dense conv streams
  auto  on chip up to a budget of W bytes, the rest streams. Layers that
        cannot stream (image layer, fc) are on chip anyway. Of the others,
        the ones that need the most weight bytes per compute cycle stay on
        chip: those would stall on DRAM (late 7x7 layers, the 1x1 convs
        before the head). Early layers reuse each weight over many pixels,
        so streaming them costs nothing.
"""
from npe.compiler import pack

P = pack.P


def layer_bytes(I, po, pi):
    """Bytes of W words the layer takes."""
    if pack.is_fc(I):    # one P x P block per W word
        return pack.cdiv(I.o_c, P) * (I.a_c // P) * P * P * po * pi
    cin_g = pack.cdiv(pack.cdiv(I.a_c, P), pi)
    cout_g = pack.cdiv(pack.cdiv(I.o_c, P), po)
    return cout_g * I.kh * I.kw * cin_g * P * P * po * pi


def choose(prog, mode="none", budget_kb=None, po=1, pi=1, pc=1):
    """Set of program indices whose weights stream."""
    cand = [n for n, I in enumerate(prog) if pack.streamable(I)]
    if mode == "none":
        return set()
    if mode == "all":
        return set(cand)
    assert mode == "auto" and budget_kb is not None
    from npe.compiler import cycles   # imports build, which imports this module
    budget = budget_kb * 1024
    used = sum(layer_bytes(I, po, pi) for n, I in enumerate(prog)
               if n not in cand and I.op == pack.isa.OP_CONV and not pack.is_depthwise(I))
    need = {n: layer_bytes(prog[n], po, pi) / cycles.instr_cycles(prog[n], po, pi, pc)[1] for n in cand}
    stream = set(cand)
    for n in sorted(cand, key=lambda n: -need[n]):
        b = layer_bytes(prog[n], po, pi)
        if used + b <= budget:
            used += b
            stream.discard(n)
    return stream
