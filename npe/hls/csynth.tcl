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
# NPE_TOP=npe_wfetch (synthesize the weight fetcher IP instead; it does not
# depend on the build: NPE_PRJ=hls_wfetch NPE_TOP=npe_wfetch NPE_EXPORT=1).
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
set top [env_or NPE_TOP npe_top]
set_top $top
add_files [file join $root hls $top.cpp] -cflags "-I$cfg -I$root/hls -I$root/isa"
open_solution -reset sol -flow_target vivado
set_part $part
create_clock -period $clock
csynth_design
if {[env_or NPE_IMPL 0]} {
  export_design -format ip_catalog -flow impl
} elseif {[env_or NPE_EXPORT 0]} {
  export_design -format ip_catalog
}
exit
