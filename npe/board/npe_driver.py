"""Host driver for the NPE engine on a Kria KV260 with PYNQ (Kria-PYNQ).

  python3 npe_driver.py <dir> [image dirs...] [--runs N]

<dir> holds npe.bit, npe.hwh (from bd_kv260.tcl), load.bin, board.json and
weights_dram.bin (from npe/compiler/build.py).

Streamed weights: if the build streams any layer (board.json ws_beats > 0),
weights_dram.bin is copied once into a DDR buffer and the fetcher IP
(npe_wfetch) gets its address and length, then runs in auto-restart: it
streams the weights again for every frame by itself, paced by the engine.
Nothing per frame on the ARM. Each image dir holds input.bin (H x W x C
uint8) and, optionally, logits.bin (int32) to check against.

It loads the program and parameters once, then runs every image, checks the
logits, and reports the time per frame measured on the ARM (DMA in, compute,
DMA out) and the fps. Board power is printed if PYNQ can read the rails.
"""
import argparse
import json
import os
import time

import numpy as np
from pynq import Overlay, allocate

NPE_RUN = 0
AP_START, AP_DONE, AP_IDLE, AUTO_RESTART = 1, 2, 4, 0x80


class NPE:
    def __init__(self, d):
        self.dir = d
        self.ol = Overlay(os.path.join(d, "npe.bit"))
        self.ip = self.ol.npe_top_0
        self.dma_load = self.ol.axi_dma_load
        self.dma_io = self.ol.axi_dma_io
        with open(os.path.join(d, "board.json")) as f:
            self.info = json.load(f)
        self.load_bin = np.fromfile(os.path.join(d, "load.bin"), np.uint8)
        rm = self.ip.register_map
        self.off = {n: getattr(rm, n).address for n in ("mode", "count", "n_out")}
        info = self.info
        beat_bytes = info["in_beat_bits"] // 8
        n_beats = -(-info["in_pixels"] // info["in_ppb"])
        self.img = allocate((n_beats, beat_bytes), np.uint8)
        self.res = allocate(info["n_out"], np.int32)

    def _call(self, mode, count, n_out=0):
        self.ip.write(self.off["mode"], mode)
        self.ip.write(self.off["count"], count)
        self.ip.write(self.off["n_out"], n_out)
        self.ip.write(0x00, AP_START)

    def _wait(self):
        while not self.ip.read(0x00) & (AP_DONE | AP_IDLE):
            pass

    def start_fetcher(self):
        """Streamed weights: DDR buffer + fetcher in auto-restart, once."""
        n = self.info.get("ws_beats", 0)
        if not n:
            return
        data = np.fromfile(os.path.join(self.dir, "weights_dram.bin"), np.uint8)
        self.wbuf = allocate(data.size, np.uint8)
        self.wbuf[:] = data
        self.wbuf.flush()
        f = self.ol.npe_wfetch_0
        rm = f.register_map
        addr = self.wbuf.device_address
        if hasattr(rm, "wmem_1"):       # 64-bit address in two registers
            f.write(rm.wmem_1.address, addr & 0xFFFFFFFF)
            f.write(rm.wmem_2.address, addr >> 32)
        else:
            f.write(rm.wmem.address, addr)
        f.write(rm.n_beats.address, n)
        f.write(0x00, AP_START | AUTO_RESTART)

    def load(self):
        """Program and parameters, once (they stay on chip)."""
        self.start_fetcher()
        for ld in self.info["loads"]:
            if ld["count"] == 0:
                continue
            buf = allocate(8 * ld["beats"], np.uint8)
            buf[:] = self.load_bin[ld["offset"]:ld["offset"] + 8 * ld["beats"]]
            buf.flush()
            self._call(ld["mode"], ld["count"])
            self.dma_load.sendchannel.transfer(buf)
            self.dma_load.sendchannel.wait()
            self._wait()
            buf.freebuffer()

    def set_image(self, x):
        """H x W x C uint8 -> beats of PPB pixels (npe_top.h)."""
        info = self.info
        px = np.zeros((self.img.shape[0] * info["in_ppb"], info["in_c"]), np.uint8)
        px[:info["in_pixels"]] = x.reshape(-1, info["in_c"])
        flat = px.reshape(self.img.shape[0], -1)
        self.img[:] = 0
        self.img[:, :flat.shape[1]] = flat
        self.img.flush()

    def run(self):
        """One frame: image in, all instructions, logits out."""
        self._call(NPE_RUN, self.info["n_instr"], self.info["n_out"])
        self.dma_io.recvchannel.transfer(self.res)
        self.dma_io.sendchannel.transfer(self.img)
        self.dma_io.sendchannel.wait()
        self.dma_io.recvchannel.wait()
        self._wait()
        self.res.invalidate()
        return np.array(self.res)


def power_w():
    try:
        from pynq import get_rails
        rails = get_rails()
        return sum(r.power.value for r in rails.values() if r.power is not None)
    except Exception:   # not available on this image
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--runs", type=int, default=100, help="timed frames per image")
    a = ap.parse_args()
    npe = NPE(a.dir)
    t = time.perf_counter()
    npe.load()
    print(f"load: {1e3 * (time.perf_counter() - t):.1f} ms")
    ok = bad = 0
    times = []
    p_idle = power_w()
    p_run = None
    for d in a.images:
        x = np.fromfile(os.path.join(d, "input.bin"), np.uint8)
        npe.set_image(x)
        y = npe.run()
        gold = os.path.join(d, "logits.bin")
        if os.path.exists(gold):
            match = np.array_equal(y, np.fromfile(gold, "<i4"))
            ok, bad = ok + match, bad + (not match)
        t = time.perf_counter()
        for _ in range(a.runs):
            npe.run()
        times.append((time.perf_counter() - t) / a.runs)
        p_run = power_w()
    if times:
        ms = 1e3 * np.median(times)
        print(f"{len(times)} images: {ms:.3f} ms per frame (median), {1e3 / ms:.0f} fps")
    print(f"logits: {ok} match, {bad} differ")
    if p_idle is not None:
        print(f"board power: {p_idle:.2f} W before, {p_run:.2f} W after running")


if __name__ == "__main__":
    main()
