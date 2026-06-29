# 012 Ternary Processing Unit — Vivado Synthesis Script
# Target: Xilinx Ultrascale+ ZCU102 (xczu9eg-ffvb1156-2-e)
# Run: vivado -mode batch -source vivado_synth.tcl

set project_name "trit_cognition"
set part         "xczu9eg-ffvb1156-2-e"
set output_dir   "./vivado_output"

file mkdir $output_dir
create_project $project_name $output_dir/$project_name -part $part -force
set_property target_language SystemVerilog [current_project]

set rtl_dir "../hardware"
add_files -norecurse [list \
    $rtl_dir/trit_pkg.sv       \
    $rtl_dir/trit_register.sv  \
    $rtl_dir/trit_not.sv       \
    $rtl_dir/trit_add.sv       \
    $rtl_dir/trit_mac.sv       \
    $rtl_dir/consensus_gate.sv \
    $rtl_dir/triadic_pe.sv     \
    $rtl_dir/012_top.sv        \
]

set_property file_type {SystemVerilog} [get_files *.sv]
update_compile_order -fileset sources_1

set constraints_file "$output_dir/trit_constraints.xdc"
set fp [open $constraints_file w]
puts $fp "create_clock -period 4.000 -name clk \[get_ports clk\]"
puts $fp "set_input_delay  -clock clk 0.5 \[all_inputs\]"
puts $fp "set_output_delay -clock clk 0.5 \[all_outputs\]"
close $fp
add_files -fileset constrs_1 $constraints_file

synth_design \
    -top trit_cognition_top \
    -part $part \
    -directive PerformanceOptimized \
    -flatten_hierarchy rebuilt

report_utilization    -file $output_dir/utilization.rpt -hierarchical
report_timing_summary -file $output_dir/timing.rpt
report_power          -file $output_dir/power.rpt

opt_design
place_design
route_design

report_utilization    -file $output_dir/utilization_post_route.rpt
report_timing_summary -file $output_dir/timing_post_route.rpt -warn_on_violation

write_bitstream -force $output_dir/$project_name.bit

puts "Done. Reports in $output_dir/"
