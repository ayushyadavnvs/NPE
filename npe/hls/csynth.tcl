# Vitis HLS synthesis of the NPE engine for one model build.
#
#   python3 -m npe.compiler.build                 # writes npe/build/shufflenet_hw
#   vitis_hls -f npe/hls/csynth.tcl               # from the repo root
#
#   python3 -m npe.compiler.build --model mobilenetv2
#   NPE_CFG=npe/build/mobilenetv2_hw NPE_PRJ=hls_mobilenetv2 vitis_hls -f npe/hls/csynth.tcl
#
# Environment overrides: NPE_CFG (build dir with npe_config.h),
# NPE_PART (default KV260: xck26-sfvc784-2LV-c), NPE_CLOCK_NS (default 5),
# NPE_PRJ (project name under npe/build, default hls_npe),
# NPE_EXPORT=1 (also export the IP for Vivado: npe/build/<NPE_PRJ>/sol/impl/ip),
# NPE_IMPL=1 (also run Vivado place and route on the IP alone: post-route
# utilization and timing in sol/impl/report/verilog),
# NPE_STEPS (default "csynth"; any of csim csynth cosim, in that order),
# NPE_TB_ARGS (testbench arguments for csim / cosim: "load.bin images.txt",
# as written by: python3 -m npe.tests.test_top N --prepare-only ...).
#
# Example, C sim + synthesis + C/RTL co-simulation of one MobileNetV2 image:
#   python3 -m npe.tests.test_top 1 --model mobilenetv2 --po 4 --pi 2 --pc 8 \
#     --rq 1 --rqc 1 --prepare-only
#   D=npe/build/top_mobilenetv2_po4_pi2_pc8_rq1_rqc1_ppb4
#   NPE_CFG=$D NPE_PRJ=hls_cosim NPE_STEPS="csim csynth cosim" \
#     NPE_TB_ARGS="$PWD/$D/load.bin $PWD/$D/images.txt" vitis_hls -f npe/hls/csynth.tcl
set root [file normalize [file join [file dirname [info script]] ..]]
proc env_or {name default} {
  if {[info exists ::env($name)]} { return $::env($name) }
  return $default
}
set cfg   [file normalize [env_or NPE_CFG [file join $root build shufflenet_hw]]]
set part  [env_or NPE_PART xck26-sfvc784-2LV-c]
set clock [env_or NPE_CLOCK_NS 5]

# open_project takes a name, not a path: run it from inside npe/build.
file mkdir [file join $root build]
cd [file join $root build]
open_project -reset [env_or NPE_PRJ hls_npe]
set_top npe_top
set cflags "-I$cfg -I$root/hls -I$root/isa"
add_files [file join $root hls npe_top.cpp] -cflags $cflags
add_files -tb [file join $root tb tb_top.cpp] -cflags $cflags
open_solution -reset sol -flow_target vivado
set_part $part
create_clock -period $clock
set steps [env_or NPE_STEPS csynth]
set tb_args [env_or NPE_TB_ARGS ""]
if {[lsearch $steps csim] >= 0} { csim_design -argv $tb_args }
if {[lsearch $steps csynth] >= 0 || [lsearch $steps cosim] >= 0} { csynth_design }
if {[lsearch $steps cosim] >= 0} {
  cosim_design -argv $tb_args -trace_level none -rtl verilog
}
if {[env_or NPE_IMPL 0]} {
  export_design -format ip_catalog -flow impl
} elseif {[env_or NPE_EXPORT 0]} {
  export_design -format ip_catalog
}
exit
