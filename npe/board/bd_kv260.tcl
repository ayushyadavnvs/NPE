# Vivado block design and bitstream for the NPE engine on the KV260.
#
#   1. HLS IP:   NPE_CFG=npe/build/<build> NPE_PRJ=hls_<build> NPE_EXPORT=1 \
#                  vitis_hls -f npe/hls/csynth.tcl
#   2. this:     NPE_IP=npe/build/hls_<build>/sol/impl/ip NPE_CFG=npe/build/<build> \
#                  vivado -mode batch -source npe/board/bd_kv260.tcl
#
# Output: npe/build/board_<build>/npe.bit and npe.hwh (copy both, with
# load.bin and board.json from NPE_CFG, to the board; see npe_driver.py).
#
# Design: Zynq MPSoC PS; PL clock 0 at NPE_MHZ (default 200) for everything.
#   axi_dma_load  MM2S, 64 bit           -> npe_top load_in (program, weights)
#   axi_dma_io    MM2S, NPE_IN_BEAT_BITS -> npe_top img_in (image)
#                 S2MM, 32 bit           <- npe_top out (int32 logits)
#   Control (npe_top s_axi_CTRL, both DMA register files) from M_AXI_HPM0_FPD;
#   DMA memory traffic to S_AXI_HP0_FPD. No interrupts: the driver polls.
set root [file normalize [file join [file dirname [info script]] ..]]
proc env_or {name default} {
  if {[info exists ::env($name)]} { return $::env($name) }
  return $default
}
set ip_dir [file normalize [env_or NPE_IP [file join $root build hls_npe sol impl ip]]]
set cfg    [file normalize [env_or NPE_CFG [file join $root build shufflenet_hw]]]
set mhz    [env_or NPE_MHZ 200]
set out    [file join $root build "board_[file tail $cfg]"]

# Image stream width from the build's npe_config.h.
set beat 32
set fh [open [file join $cfg npe_config.h]]
foreach line [split [read $fh] "\n"] {
  if {[regexp {#define NPE_IN_BEAT_BITS (\d+)} $line -> b]} { set beat $b }
}
close $fh

create_project -force npe_kv260 [file join $out vivado] -part xck26-sfvc784-2LV-c
# The KV260 board files give the PS its real configuration (DDR, clocks).
set bp [lindex [get_board_parts -quiet -latest_file_version *kv260_som*] 0]
if {$bp ne ""} {
  set_property board_part $bp [current_project]
} else {
  puts "WARNING: KV260 board files not installed; PS uses part defaults"
}
set_property ip_repo_paths $ip_dir [current_project]
update_ip_catalog

create_bd_design npe
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
if {$bp ne ""} {
  apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} $ps
}
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $mhz] $ps

set npe [create_bd_cell -type ip -vlnv [lindex [get_ipdefs *npe_top*] 0] npe_top_0]

set dl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_load]
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_include_s2mm {0} \
  CONFIG.c_sg_length_width {26} CONFIG.c_m_axis_mm2s_tdata_width {64} \
  CONFIG.c_m_axi_mm2s_data_width {64} CONFIG.c_mm2s_burst_size {64}] $dl

set dio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_io]
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {26} \
  CONFIG.c_m_axis_mm2s_tdata_width $beat CONFIG.c_m_axi_mm2s_data_width [expr {max($beat, 64)}] \
  CONFIG.c_s_axis_s2mm_tdata_width {32} CONFIG.c_m_axi_s2mm_data_width {64} \
  CONFIG.c_mm2s_burst_size {64} CONFIG.c_s2mm_burst_size {64}] $dio

connect_bd_intf_net [get_bd_intf_pins axi_dma_load/M_AXIS_MM2S] [get_bd_intf_pins npe_top_0/load_in]
connect_bd_intf_net [get_bd_intf_pins axi_dma_io/M_AXIS_MM2S] [get_bd_intf_pins npe_top_0/img_in]
# The output stream is npe_top's only AXI-Stream master (HLS may rename "out").
set out_pin [get_bd_intf_pins -of_objects $npe -filter {MODE == Master && VLNV =~ "*axis*"}]
connect_bd_intf_net $out_pin [get_bd_intf_pins axi_dma_io/S_AXIS_S2MM]

# Control: PS master -> npe_top and both DMA register files.
foreach slave {npe_top_0/s_axi_CTRL axi_dma_load/S_AXI_LITE axi_dma_io/S_AXI_LITE} {
  apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {/zynq_ultra_ps_e_0/pl_clk0 (200 MHz)} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave "/$slave" ddr_seg {Auto} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins $slave]
}
# Data: DMA masters -> PS HP0.
foreach master {axi_dma_load/M_AXI_MM2S axi_dma_io/M_AXI_MM2S axi_dma_io/M_AXI_S2MM} {
  apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master "/$master" Slave {/zynq_ultra_ps_e_0/S_AXI_HP0_FPD} ddr_seg {Auto} \
    intc_ip {Auto} master_apm {0}] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
}
# npe_top's clock and reset follow the control bus (pl_clk0).
assign_bd_address
validate_bd_design
save_bd_design

set wrapper [make_wrapper -files [get_files npe.bd] -top]
add_files -norecurse $wrapper
set_property top npe_wrapper [current_fileset]
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

set impl [get_runs impl_1]
if {[get_property PROGRESS $impl] ne "100%"} {
  puts "ERROR: implementation did not finish"
  exit 1
}
open_run impl_1
report_utilization -file [file join $out utilization.rpt]
report_timing_summary -file [file join $out timing.rpt]
file copy -force [glob [file join $out vivado npe_kv260.runs impl_1 *.bit]] [file join $out npe.bit]
file copy -force [glob [file join $out vivado npe_kv260.gen sources_1 bd npe hw_handoff npe.hwh]] [file join $out npe.hwh]
puts "Wrote [file join $out npe.bit] and npe.hwh"
exit
