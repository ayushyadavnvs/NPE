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
# NPE_PRJ (project name under npe/build, default hls_npe).
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
add_files [file join $root hls npe_top.cpp] -cflags "-I$cfg -I$root/hls -I$root/isa"
open_solution -reset sol -flow_target vivado
set_part $part
create_clock -period $clock
csynth_design
exit
