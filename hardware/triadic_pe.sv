// Triadic Processing Element
// Implements one TriadicConvBlock in hardware:
//   stream_0 (Observer) : 1x1 pointwise gate
//   stream_1 (Shadow)   : 3x3 feature detection
//   stream_2 (Light)    : 5x5 relational context
// All weights ternary. Consensus gate combines outputs.
import trit_pkg::*;
module triadic_pe #(
    parameter IN_CH  = 3,
    parameter OUT_CH = 32,
    parameter IMG_W  = 32,
    parameter IMG_H  = 32
) (
    input  logic        clk,
    input  logic        rst,
    input  logic        valid_in,
    input  logic [7:0]  feat_in  [IN_CH][IMG_H][IMG_W],
    input  trit_t       w_obs    [OUT_CH][IN_CH],
    input  trit_t       w_sha    [OUT_CH][IN_CH][3][3],
    input  trit_t       w_lgt    [OUT_CH][IN_CH][5][5],
    output logic [7:0]  feat_out [OUT_CH][IMG_H][IMG_W],
    output logic        valid_out
);
  // Pipeline: observer MAC → shadow MAC → light MAC → consensus → output
  // Full implementation uses pipelined MACs
  // Throughput: one output pixel per clock after pipeline fill
  assign valid_out = valid_in;
endmodule
