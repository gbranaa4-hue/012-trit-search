// Ternary NOT: negate the trit (flip sign)
import trit_pkg::*;
module trit_not (
    input  trit_t a,
    output trit_t out
);
  always_comb begin
    case (a)
      TRIT_NEG  : out = TRIT_POS;
      TRIT_POS  : out = TRIT_NEG;
      default   : out = TRIT_ZERO;
    endcase
  end
endmodule
