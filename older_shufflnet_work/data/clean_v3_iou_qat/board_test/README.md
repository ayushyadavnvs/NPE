# Labelled board test set

490 real images, seed 42. Class order: 0 other / 1 bird / 2 drone.
Sampled without replacement with NumPy default_rng(42), in split order test,
test_cars, test_halm_unseen and class order 0/1/2 (CARS 1/2), using manifest
row order. No generated fallback or class shortage. Excluded generated,
step3_generated and gen flags; source/pool prefixes paste_, hn_, aug_, tight,
empty; and air_to_air. Candidates and input hashes: validation.json.

| Set | Letter | Other | Bird | Drone | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| test | T | 100 | 100 | 100 | 300 |
| test_cars | C | 0 | 50 | 50 | 100 |
| test_halm_unseen | H | 30 | 30 | 30 | 90 |
| Total | | 130 | 180 | 180 | 490 |

Every <tag>.bin is exactly 150,528 bytes: 224×224×3 RGB uint8 HWC, C order,
no header and no normalization. Tags are O/B/D + T/C/H + three digits,
numbered from 001 within each set/class (e.g. DT001.bin, BC012.bin, OH030.bin).
These image filenames fit FAT 8.3. Raw image payload totals 73,758,720 bytes.

Preprocessing directly calls code/port/step6_common.raw224, exactly as Step 6
golden export and INT8 evaluation: open original_path with PIL, convert RGB,
recovered train_v2.window evaluation crop (normally centre 16:9, respecting
crop_axis/crop_lo/crop_hi/crop_box), integer-rounded crop coordinates, then
PIL bilinear squash to 224×224. All 16 existing golden images re-created through
this path were byte-identical. Do not use generic golden/img2bin.py defaults.
The board receives raw bytes; ImageNet normalization is folded into INT8 conv1.

FP32: runs/clean_train_v3_iou/checkpoints/best.pt, epoch 25, CPU float32,
no autocast. Each board image after recovered ImageNet ToTensor/Normalize
was asserted equal to train_v2.V2DS(False) input before FP32 inference.
INT8: code/accel3class_ref/golden/ref_int8.py Int8Model, reading existing
export/weights.bin, run directly on each .bin. int8_logits are three signed
int32 values in class order, serialized as JSON integers. First index wins
argmax ties for both models. No model training or checkpoint selection.
FP32 uses the original checkpoint; INT8 uses the Step 6 QAT export.

manifest.jsonl records tag (without extension), label, set (split stem), source,
original_path, original row_index, predictions, logits and image SHA256.
Crop overrides and Halmstad class/visibility metadata are retained when present.
metrics.json includes overall/per-set confusion matrices (true rows, predicted
columns), class supports/recalls, macro-F1 and model agreement. Accuracy by class
below means recall on that true class. Absent CARS other is N/A.

Halmstad follows test_halm_unseen.jsonl labels without adding a visibility filter: 2 selected frames have visible=0; 0 have has_object=false. The exclusion of tight/empty rows refers to generated/virtual crops. Real sky remains a valid other example; CARS rows have no boxes by design.

## Accuracy on this sampled set

| Set | Class | n | FP32 correct | FP32 accuracy | INT8 correct | INT8 accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all | all | 490 | 392 | 80.00% | 384 | 78.37% |
| all | other | 130 | 100 | 76.92% | 95 | 73.08% |
| all | bird | 180 | 143 | 79.44% | 144 | 80.00% |
| all | drone | 180 | 149 | 82.78% | 145 | 80.56% |
| test | all | 300 | 269 | 89.67% | 268 | 89.33% |
| test | other | 100 | 78 | 78.00% | 74 | 74.00% |
| test | bird | 100 | 97 | 97.00% | 99 | 99.00% |
| test | drone | 100 | 94 | 94.00% | 95 | 95.00% |
| test_cars | all | 100 | 82 | 82.00% | 79 | 79.00% |
| test_cars | other | 0 | 0 | N/A | 0 | N/A |
| test_cars | bird | 50 | 38 | 76.00% | 38 | 76.00% |
| test_cars | drone | 50 | 44 | 88.00% | 41 | 82.00% |
| test_halm_unseen | all | 90 | 41 | 45.56% | 37 | 41.11% |
| test_halm_unseen | other | 30 | 22 | 73.33% | 21 | 70.00% |
| test_halm_unseen | bird | 30 | 8 | 26.67% | 7 | 23.33% |
| test_halm_unseen | drone | 30 | 11 | 36.67% | 9 | 30.00% |

## Counts by original source

| Source | Other (0) | Bird (1) | Drone (2) | Total |
| --- | ---: | ---: | ---: | ---: |
| CARS | 0 | 50 | 50 | 100 |
| aod4_mendeley | 70 | 17 | 5 | 92 |
| bird_classification_rf | 0 | 47 | 0 | 47 |
| bird_only_rf | 0 | 25 | 0 | 25 |
| birdvsdrone1 | 0 | 8 | 2 | 10 |
| drone_bird_backup | 0 | 2 | 3 | 5 |
| drone_detection_new | 0 | 1 | 1 | 2 |
| halmstad | 30 | 30 | 30 | 90 |
| pathikg_drone | 0 | 0 | 89 | 89 |
| planes_helicopters | 22 | 0 | 0 | 22 |
| small_aerial | 2 | 0 | 0 | 2 |
| zenodo_sky | 6 | 0 | 0 | 6 |

SHA256SUMS covers every bundle file except itself; run `sha256sum -c SHA256SUMS`
inside board_test/. validation.json records source/checkpoint/reference hashes,
candidate counts, golden byte comparisons and existing-export preservation.
Reproduce/resume from WORK: `bash code/port/run_step7.sh`.
