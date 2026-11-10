#ifndef BD_MODEL_CONFIG_H_
#define BD_MODEL_CONFIG_H_

// ShuffleNetV2 x0.5 (torchvision layout) with a 3-class head.
// Shapes only; trained parameters arrive later from the model team as INT8
// tensors plus per-channel requantization constants (see PLAN.md).

// Input frame. Change IMG_SIZE together with the model team's training size.
#define IMG_SIZE 224
#define IMG_CHANNELS 3

// Stage widths for the x0.5 variant.
#define STEM_OUT 24
#define STAGE2_OUT 48
#define STAGE3_OUT 96
#define STAGE4_OUT 192
#define CONV5_OUT 1024
#define NUM_CLASSES 3
// Class order of the fc rows and the logit outputs (clean_v2 label order).
#define CLASS_OTHER 0
#define CLASS_BIRD 1
#define CLASS_DRONE 2

// Units per stage (first unit of each stage is the stride-2 downsample unit).
#define STAGE2_UNITS 4
#define STAGE3_UNITS 8
#define STAGE4_UNITS 4
#define TOTAL_UNITS (STAGE2_UNITS + STAGE3_UNITS + STAGE4_UNITS)

// Spatial sizes, derived from IMG_SIZE.
#define STEM_SIZE ((IMG_SIZE + 1) / 2)    // conv1 3x3 stride 2
#define POOL_SIZE ((STEM_SIZE + 1) / 2)   // maxpool 3x3 stride 2
#define STAGE2_SIZE ((POOL_SIZE + 1) / 2) // 28 at 224
#define STAGE3_SIZE ((STAGE2_SIZE + 1) / 2)
#define STAGE4_SIZE ((STAGE3_SIZE + 1) / 2)

// One entry per shuffle unit; the unit engine walks this table at run time.
struct unit_cfg_t {
  short in_size;  // input height == width
  short in_ch;    // channels entering the unit
  short out_ch;   // channels leaving the unit
  bool downsample; // stride-2 unit with two convolution branches
};

static const unit_cfg_t UNIT_TABLE[TOTAL_UNITS] = {
    {POOL_SIZE, STEM_OUT, STAGE2_OUT, true},
    {STAGE2_SIZE, STAGE2_OUT, STAGE2_OUT, false},
    {STAGE2_SIZE, STAGE2_OUT, STAGE2_OUT, false},
    {STAGE2_SIZE, STAGE2_OUT, STAGE2_OUT, false},
    {STAGE2_SIZE, STAGE2_OUT, STAGE3_OUT, true},
    {STAGE3_SIZE, STAGE3_OUT, STAGE3_OUT, false},
    {STAGE3_SIZE, STAGE3_OUT, STAGE3_OUT, false},
    {STAGE3_SIZE, STAGE3_OUT, STAGE3_OUT, false},
    {STAGE3_SIZE, STAGE3_OUT, STAGE3_OUT, false},
    {STAGE3_SIZE, STAGE3_OUT, STAGE3_OUT, false},
    {STAGE3_SIZE, STAGE3_OUT, STAGE3_OUT, false},
    {STAGE3_SIZE, STAGE3_OUT, STAGE3_OUT, false},
    {STAGE3_SIZE, STAGE3_OUT, STAGE4_OUT, true},
    {STAGE4_SIZE, STAGE4_OUT, STAGE4_OUT, false},
    {STAGE4_SIZE, STAGE4_OUT, STAGE4_OUT, false},
    {STAGE4_SIZE, STAGE4_OUT, STAGE4_OUT, false},
};

#endif // BD_MODEL_CONFIG_H_
