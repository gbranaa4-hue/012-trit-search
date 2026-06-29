// 012 Consensus Gate — the native ternary logic gate
// Output = majority vote of three trits {-1, 0, +1}
// Maps to: Observer (0) + Shadow (1) + Light (2) → decision
import trit_pkg::*;
module consensus_gate (
    input  trit_t a,    // Observer  (stream_0)
    input  trit_t b,    // Shadow    (stream_1)
    input  trit_t c,    // Light     (stream_2)
    output trit_t out
);
  logic signed [2:0] sum;
  logic signed [1:0] sa, sb, sc;

  always_comb begin
    sa  = (a == TRIT_NEG) ? -1 : (a == TRIT_POS) ? 1 : 0;
    sb  = (b == TRIT_NEG) ? -1 : (b == TRIT_POS) ? 1 : 0;
    sc  = (c == TRIT_NEG) ? -1 : (c == TRIT_POS) ? 1 : 0;
    sum = sa + sb + sc;

    if      (sum > 0) out = TRIT_POS;
    else if (sum < 0) out = TRIT_NEG;
    else              out = TRIT_ZERO;
  end
endmodule
