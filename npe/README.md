# NPE: general CNN engine (work in progress)

The generic successor of `older_shufflnet_work/`. Plan and instruction-format
rationale: the "Generalization plan" section of the project analysis doc.

| Path | What it is |
| --- | --- |
| `compiler/isa.py` | The instruction set: one 16-bit-field instruction per layer, semantics in its docstring. `python3 -m npe.compiler.isa` regenerates `isa/npe_isa.h` |
| `ref/isa_ref.py` | NumPy interpreter of the instruction set; the definition every other part is checked against |
| `compiler/lower_shufflenet.py` | Stand-in front end: lowers the old ShuffleNet `weights.bin` to generic instructions (the ONNX importer replaces it in step 3) |
| `tests/test_shufflenet_isa.py` | ShuffleNet on the generic instructions vs the 104 golden logits |

Run from the repo root:

```
python3 -m npe.tests.test_shufflenet_isa        # all golden images, ~15 s
python3 -m npe.compiler.lower_shufflenet --list # the 59-instruction program
```

Status (step 1 of 5): the instruction set expresses the whole ShuffleNetV2,
stem and head included, and the reference matches all 104 golden logits.
Next: the HLS engine that executes these instructions, then MobileNetV2.
