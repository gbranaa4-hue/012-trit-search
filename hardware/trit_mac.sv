// Ternary Multiply-Accumulate Unit
// Weight encoding {-1,0,+1} eliminates multipliers:
//   +1 → route input to adder
//   -1 → route negated input to adder
//    0 → clock-gate (no operation, saves power)
import trit_pkg::*;
module trit_mac #(parameter WIDTH=16) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic signed [7:0]       data_in  [WIDTH],
    input  trit_t                   weights  [WIDTH],
    input  logic                    valid,
    output logic signed [15:0]      accum,
    output logic                    done
);
  logic signed [15:0] acc;
  integer i;

  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      acc  <= 0;
      done <= 0;
    end else if (valid) begin
      acc <= acc;
      for (i = 0; i < WIDTH; i++) begin
        case (weights[i])
          TRIT_POS  : acc <= acc + data_in[i];
          TRIT_NEG  : acc <= acc - data_in[i];
          TRIT_ZERO : ;
        endcase
      end
      done <= 1;
    end else begin
      done <= 0;
    end
  end

  assign accum = acc;
endmodule
