// 012 Ternary Computing Package
// Trit encoding: 2'b00 = -1 (suppress)
//                2'b01 =  0 (neutral)
//                2'b10 = +1 (activate)
package trit_pkg;
  typedef logic [1:0] trit_t;
  localparam trit_t TRIT_NEG  = 2'b00;
  localparam trit_t TRIT_ZERO = 2'b01;
  localparam trit_t TRIT_POS  = 2'b10;
endpackage
