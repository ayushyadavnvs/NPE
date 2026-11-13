# NPE: general CNN engine (work in progress)

The generic successor of `older_shufflnet_work/`. Plan and instruction-format
rationale: the "Generalization plan" section of the project analysis doc.

| Path | What it is |
| --- | --- |
| `compiler/isa.py` | The instruction set: one 16-bit-field instruction per layer, semantics in its docstring. `python3 -m npe.compiler.isa` regenerates `isa/npe_isa.h` |
| `ref/isa_ref.py` | NumPy interpreter of the instruction set; the definition every other part is checked against |
| `compiler/lower_shufflenet.py` | Stand-in front end: lowers the old ShuffleNet `weights.bin` to generic instructions (the ONNX importer replaces it in step 3) |
| `compiler/pack.py` | Packs a program's weights and quant constants into the engine's word memories and writes `npe_config.h` |
| `hls/` | The HLS engine: `npe_decode.h` (instruction to datapath settings), `npe_conv.h` (dense KxK conv unit with padding), `npe_dwconv.h` (depthwise conv), `npe_pool.h` (max pool), `npe_gap.h` (global average pool into V), `npe_fc.h` (fully connected, int32 out to the OUT stream), `npe_engine.h` (dispatch) |
| `tb/tb_engine.cpp`, `tests/test_engine.py` | Engine compiled with g++ vs the reference, instruction by instruction (ShuffleNet + random cases) |
| `hls/npe_top.h/.cpp` | Top level: AXI-Lite control, program/weight loading and image input over AXI-Stream, int32 results out over AXI-Stream |
| `compiler/build.py` | Writes `npe_config.h` and `load.bin` (the host's load stream) for a model |
| `hls/csynth.tcl` | Vitis HLS synthesis (`vitis_hls -f npe/hls/csynth.tcl` after `build.py`) |
| `tb/tb_top.cpp`, `tests/test_top.py` | ShuffleNet through `npe_top`'s ports only, vs the golden logits |
| `tb/tb_run.cpp`, `tests/test_engine_e2e.py` | Whole ShuffleNet on the engine, image in, logits out, vs the golden logits |
| `compiler/lower_mobilenetv2.py` | MobileNetV2-shaped test network (real layer structure, random calibrated int8 weights): exercises the fused ADD and ReLU6 |
| `tests/test_shufflenet_isa.py` | ShuffleNet on the generic instructions vs the 104 golden logits |

Run from the repo root:

```
python3 -m npe.tests.test_shufflenet_isa        # all golden images, ~15 s
python3 -m npe.compiler.lower_shufflenet --list # the 59-instruction program
python3 -m npe.tests.test_engine                # HLS engine vs reference, CPU only
python3 -m npe.tests.test_engine_e2e 5          # whole network on the engine, 5 images
python3 -m npe.tests.test_top 5                 # same, through the AXI top level
python3 -m npe.tests.test_top --model mobilenetv2  # MobileNetV2-shaped net vs the reference
python3 -m npe.compiler.build && vitis_hls -f npe/hls/csynth.tcl   # synthesis
```

Status: the HLS engine runs ShuffleNetV2 end to end (all 104 golden images
bit-exact through the AXI top level) and a MobileNetV2-shaped network with
residual blocks (fused ADD epilogue, ReLU6) bit-exact against the reference.
Synthesized for the KV260 (7c0b738, before ADD): 3.62 ns, every loop II=1.
The ADD epilogue is built only when the program uses it (NPE_HAS_ADD).
