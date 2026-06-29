// Ternary saturating adder
// Inputs: two trits {-1, 0, +1}
// Output: sum trit + carry trit
import trit_pkg::*;
module trit_add (
    input  trit_t a,
    input  trit_t b,
    output trit_t sum,
    output trit_t carry
);
  logic signed [2:0] raw;
  logic signed [1:0] sa, sb;

  always_comb begin
    sa    = (a == TRIT_NEG) ? -1 : (a == TRIT_POS) ? 1 : 0;
    sb    = (b == TRIT_NEG) ? -1 : (b == TRIT_POS) ? 1 : 0;
    raw   = sa + sb;

    if      (raw ==  2) begin sum = TRIT_ZERO; carry = TRIT_POS;  end
    else if (raw == -2) begin sum = TRIT_ZERO; carry = TRIT_NEG;  end
    else if (raw ==  1) begin sum = TRIT_POS;  carry = TRIT_ZERO; end
    else if (raw == -1) begin sum = TRIT_NEG;  carry = TRIT_ZERO; end
    else                begin sum = TRIT_ZERO; carry = TRIT_ZERO; end
  end
endmodule
