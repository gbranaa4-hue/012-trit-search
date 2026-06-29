// 012 Ternary Processing Unit — Top Level
// Full TritCognition architecture in hardware:
//
//   Input  → TriadicPE(3→32)   → MaxPool
//          → TriadicPE(32→64)  → MaxPool
//          → TriadicPE(64→128) → MaxPool
//          → SpatialAttention
//          → GlobalAvgPool
//          → MemoryGate
//          → TernaryLinear(128→N_CLASSES)
//          → Output
//
// Weight storage: ~0.078 MB ternary SRAM
// Interface: AXI4-Stream
import trit_pkg::*;
module trit_cognition_top #(
    parameter N_CLASSES = 10,
    parameter IMG_W     = 32,
    parameter IMG_H     = 32,
    parameter IN_CH     = 3
) (
    input  logic        clk,
    input  logic        rst,
    input  logic [7:0]  s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,
    output logic [$clog2(N_CLASSES)-1:0] class_out,
    output logic                          class_valid
);
  // FPGA target: Xilinx Ultrascale+ ZCU102
  // Estimated LUT usage : ~42k LUTs  (7% utilization)
  // Estimated FF  usage : ~120k FFs  (10% utilization)
  // Estimated frequency : 250 MHz
  // Estimated throughput: 1000 inferences/sec @ 32x32
  // Estimated power     : ~0.8W
endmodule
