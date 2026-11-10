# clean_v3_iou QAT board export

FP32 source: `/media/pglab/HDD1/DRDO_BvsD/ayush_clean/runs/clean_train_v3_iou/checkpoints/best.pt` (epoch 25), SHA256 `698147e8d0746c84d7ac6b3ceb96e492fb775b7685deabbb6f9bee2c8665c11b`.
QAT selected epoch: 2; selection: clean_v3_carsdirect val macro-F1 only.
Class order: **0 other / 1 bird / 2 drone**, ShuffleNet V2 x0.5, FC 1024→3.
Training-only GIoU box head dropped; only backbone + FC are exported.

Input files: exactly 224×224×3 bytes, uint8 RGB HWC. Host applies recovered train_v2.window evaluation crop (normally centre 16:9, with manifest crop overrides), then PIL bilinear squash to 224×224. No aspect-preserving padding. ImageNet mean [0.485,0.456,0.406] and std [0.229,0.224,0.225] are folded into conv1; feed raw 0..255 pixels to the board. PAD_VALUE=[124,116,104]. Integer mean padding approximates FP32 zero padding: rounding 255*mean gives raw residual [0.325,-0.280,0.470], normalized residual [0.0055656,-0.0049020,0.0081917] at the outer stem border.

The reference golden/img2bin.py default resizes the full frame and its --crop option uses short-side-256/centre-224; neither reproduces train_v2 for a non-16:9 source. Use step6_common.raw224 or the already-preprocessed golden inputs here. Recovered board_eval.common.squash agrees with train_v2 for ordinary frames; manifest crop_box/crop_axis overrides must also be honoured for dataset evaluation.

Quantization: recovered golden/quantize.py, explicit QAT activation scales, no external calibration; per-output-channel symmetric INT8 weights (FC one tensor scale), uint8 ReLU outputs, signed int8 depthwise, shared stage scales, integer bias and M/S requant, GAP int32 sum and three comparable raw int32 logits. First index wins argmax ties.
Recovered QNet uses float biases/accumulation and fake quantization; it is not bit-exact with this integer model. It is never substituted for ref_int8 in reported INT8 metrics.

Golden set: 16 held-out real images, 3 little-endian int32 logits each; NumPy/C++ status: ALL_TENSORS_BIT_EXACT.
See golden/list.json for image provenance, true labels, expected logits and hashes.

## SHA256

| File | SHA256 |
| --- | --- |
| golden/G000/input.bin | `8f34c99ae0282c5473886c6baa0b8cd8eaccdf1468fe04c6b050fe8cf0095a51` |
| golden/G000/logits.bin | `9cf3795a7fd91bea61871d1c15187fe95cfd51e9ed71981efddcc0b280472d02` |
| golden/G001/input.bin | `dda30f7a3882832a6ebebf5a4a561892b0180d0ff20239c108ea32a573c98e7f` |
| golden/G001/logits.bin | `788756417e2990f0cbe445c92eeb4dc66739954f7a377797d3585cd6bfb96568` |
| golden/G002/input.bin | `6ff1efccf0c7af0cbafb574436e0ca64b1ad1283887443cdbec62ede0a5e1228` |
| golden/G002/logits.bin | `4a8ce083e694cadb5ea7714c059535270901b6262d79327dd15d53a5572e86ff` |
| golden/G003/input.bin | `712a117e1ad06478bf4e87d12976e44c4a19b0a021b8bd0bce06c9590ed053e4` |
| golden/G003/logits.bin | `95e16d7edb9e76abd5792fded7da6a9b921adefb1299d016b85b0e65b2c8cc68` |
| golden/G004/input.bin | `e1aaea5066868c3b3a3d7460378ff6484da938da10a880957226ff4566d4cfb7` |
| golden/G004/logits.bin | `808a863dfdd111dc45aef2b3e37436e86f6865f5100f63cdfb7cdac5019e775f` |
| golden/G005/input.bin | `891a22f947414f2defc30e531629ac4c730a913fae50b3097713387af771051d` |
| golden/G005/logits.bin | `c2d51ff5e2bf69dbaf4b3ebd81e623e696e347da8828dd56074478ae0dba8de1` |
| golden/G006/input.bin | `fef9615e3d84e84863ab387a91fac54130c93af13ce6186d3a259ded799819bb` |
| golden/G006/logits.bin | `097795420547b036e8c7e9e471423b2cdb9e4b23e960aec20c63d705a3d5ac91` |
| golden/G007/input.bin | `d412e61504fc70458ff32524254770fc32b2d1aa447beaac1cc56b692120f24b` |
| golden/G007/logits.bin | `8094d17b1fa46bf42d1de0a52330d6d271fdf26f02cd28540d48b185e636be4b` |
| golden/G008/input.bin | `32bc344c2a94625e04ad3a380527cb5f0c4cc1a049e13591d2f43374a12015b3` |
| golden/G008/logits.bin | `fd2709e0164ceff05889f81664e130d48e40ecdeb5c119c8f86bf5ff26b70b7e` |
| golden/G009/input.bin | `0432f6e7fdd874b539371306e10aa3d63d6143f7895e2e3737bc386b3f48d7f0` |
| golden/G009/logits.bin | `26c51fd830e0d536aa467a4a26a4c80970b6ba6a5406ab26442e738bd6400630` |
| golden/G010/input.bin | `c2719092b729566fad9ad3071af06421d5b7898215b39a87a900f285885d85cd` |
| golden/G010/logits.bin | `22a081f03cfee9ca613b8d81b037a4f4ebeb0ef62a1702089a86b1e9147da87d` |
| golden/G011/input.bin | `a17f67930a3fd50de598f4efd52547fb96817ad7a5a14b9b45727601315a1527` |
| golden/G011/logits.bin | `dc9ea7f84206b9cd8ba64cc03e6a5f53cf2e9c1ed29c839bedc0301e73a465a4` |
| golden/G012/input.bin | `02834306d6126ba4c6261c84f44c71910822bed1aea4a330e9e63e5f54650a0f` |
| golden/G012/logits.bin | `ef9b209ff28d28fe4454b1cb1d8653952a2ffa19bf622fa5e2314a70a046deb3` |
| golden/G013/input.bin | `bfcff64785e88678e5f49204c44b5444854eee6075c59280be96843f595c8bb0` |
| golden/G013/logits.bin | `a676240eb5a90ef43f4727b15064df325fb4c006e9d23f6f96d3065913566947` |
| golden/G014/input.bin | `d5656aafeab22726d1503e114c8d62f4bd174763ff559b27796639d8177a85de` |
| golden/G014/logits.bin | `9dd846faa6303f8f0e84a8ed4bb1e5447125c74ac1050dbf0d4088ac5179817d` |
| golden/G015/input.bin | `135a568696521effbbaaeb08b861c05cd059a21799ccefb2ae3c1299dd3145cc` |
| golden/G015/logits.bin | `3ca44ce3367c560a7e86c1606b839fac2d6504e8cabc12e6a4b0bdb2d8641e9f` |
| golden/list.json | `0fdd2615dd9c069608cc41bbc1563f177ef6221b8a5d22ca610294e4877155fa` |
| qat_folded.npz | `9809f4cca44e33ccf90d41126cf1823af87cf47b5df37f97456d6648c5d3b458` |
| quant_meta.json | `1761f8e9db6dc82a11d18ca91b78e3079a148cbd827cbb6e89582352823921bd` |
| scales.json | `21d9b73b07a35ca3cdee06857b23d5dcb8ebda23c2589c7c9590266c35225f82` |
| weights.bin | `804c7a346be1df7020b1fbb36298a41f34f08fe5b70bf0fdf4a0775539f30729` |
