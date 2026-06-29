// Ternary D flip-flop
// Stores one trit (2 bits) with synchronous reset
import trit_pkg::*;
module trit_register (
    input  logic  clk,
    input  logic  rst,
    input  trit_t d,
    output trit_t q
);
  always_ff @(posedge clk or posedge rst) begin
    if (rst) q <= TRIT_ZERO;
    else     q <= d;
  end
endmodule
