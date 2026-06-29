`timescale 1ns/1ps
import trit_pkg::*;

module testbench;
  logic clk = 0;
  always #2 clk = ~clk;

  trit_t cg_a, cg_b, cg_c, cg_out;
  consensus_gate DUT_CG (.a(cg_a), .b(cg_b), .c(cg_c), .out(cg_out));

  trit_t add_a, add_b, add_sum, add_carry;
  trit_add DUT_ADD (.a(add_a), .b(add_b), .sum(add_sum), .carry(add_carry));

  logic   rst = 1;
  trit_t  reg_d, reg_q;
  trit_register DUT_REG (.clk(clk), .rst(rst), .d(reg_d), .q(reg_q));

  int pass_count = 0;
  int fail_count = 0;

  task check_consensus(input trit_t a, b, c, expected);
    cg_a = a; cg_b = b; cg_c = c; #1;
    if (cg_out === expected) pass_count++;
    else begin
      $display("FAIL consensus: got %0d expected %0d",
        (cg_out==TRIT_NEG)?-1:(cg_out==TRIT_POS)?1:0,
        (expected==TRIT_NEG)?-1:(expected==TRIT_POS)?1:0);
      fail_count++;
    end
  endtask

  task check_add(input trit_t a, b, exp_sum, exp_carry);
    add_a = a; add_b = b; #1;
    if (add_sum === exp_sum && add_carry === exp_carry) pass_count++;
    else begin
      $display("FAIL add: got sum=%0d carry=%0d expected sum=%0d carry=%0d",
        (add_sum==TRIT_NEG)?-1:(add_sum==TRIT_POS)?1:0,
        (add_carry==TRIT_NEG)?-1:(add_carry==TRIT_POS)?1:0,
        (exp_sum==TRIT_NEG)?-1:(exp_sum==TRIT_POS)?1:0,
        (exp_carry==TRIT_NEG)?-1:(exp_carry==TRIT_POS)?1:0);
      fail_count++;
    end
  endtask

  initial begin
    $display("═══════════════════════════════════");
    $display("  012 Ternary Unit Testbench");
    $display("═══════════════════════════════════");
    #10 rst = 0;

    $display("\n── Consensus Gate (27 cases)");
    check_consensus(TRIT_NEG,  TRIT_NEG,  TRIT_NEG,  TRIT_NEG);
    check_consensus(TRIT_NEG,  TRIT_NEG,  TRIT_ZERO, TRIT_NEG);
    check_consensus(TRIT_NEG,  TRIT_NEG,  TRIT_POS,  TRIT_NEG);
    check_consensus(TRIT_NEG,  TRIT_ZERO, TRIT_NEG,  TRIT_NEG);
    check_consensus(TRIT_NEG,  TRIT_ZERO, TRIT_ZERO, TRIT_NEG);
    check_consensus(TRIT_NEG,  TRIT_ZERO, TRIT_POS,  TRIT_ZERO);
    check_consensus(TRIT_NEG,  TRIT_POS,  TRIT_NEG,  TRIT_NEG);
    check_consensus(TRIT_NEG,  TRIT_POS,  TRIT_ZERO, TRIT_ZERO);
    check_consensus(TRIT_NEG,  TRIT_POS,  TRIT_POS,  TRIT_POS);
    check_consensus(TRIT_ZERO, TRIT_NEG,  TRIT_NEG,  TRIT_NEG);
    check_consensus(TRIT_ZERO, TRIT_NEG,  TRIT_ZERO, TRIT_NEG);
    check_consensus(TRIT_ZERO, TRIT_NEG,  TRIT_POS,  TRIT_ZERO);
    check_consensus(TRIT_ZERO, TRIT_ZERO, TRIT_NEG,  TRIT_NEG);
    check_consensus(TRIT_ZERO, TRIT_ZERO, TRIT_ZERO, TRIT_ZERO);
    check_consensus(TRIT_ZERO, TRIT_ZERO, TRIT_POS,  TRIT_POS);
    check_consensus(TRIT_ZERO, TRIT_POS,  TRIT_NEG,  TRIT_ZERO);
    check_consensus(TRIT_ZERO, TRIT_POS,  TRIT_ZERO, TRIT_POS);
    check_consensus(TRIT_ZERO, TRIT_POS,  TRIT_POS,  TRIT_POS);
    check_consensus(TRIT_POS,  TRIT_NEG,  TRIT_NEG,  TRIT_NEG);
    check_consensus(TRIT_POS,  TRIT_NEG,  TRIT_ZERO, TRIT_ZERO);
    check_consensus(TRIT_POS,  TRIT_NEG,  TRIT_POS,  TRIT_POS);
    check_consensus(TRIT_POS,  TRIT_ZERO, TRIT_NEG,  TRIT_ZERO);
    check_consensus(TRIT_POS,  TRIT_ZERO, TRIT_ZERO, TRIT_POS);
    check_consensus(TRIT_POS,  TRIT_ZERO, TRIT_POS,  TRIT_POS);
    check_consensus(TRIT_POS,  TRIT_POS,  TRIT_NEG,  TRIT_POS);
    check_consensus(TRIT_POS,  TRIT_POS,  TRIT_ZERO, TRIT_POS);
    check_consensus(TRIT_POS,  TRIT_POS,  TRIT_POS,  TRIT_POS);

    $display("\n── Trit Adder");
    check_add(TRIT_NEG,  TRIT_NEG,  TRIT_ZERO, TRIT_NEG);
    check_add(TRIT_NEG,  TRIT_ZERO, TRIT_NEG,  TRIT_ZERO);
    check_add(TRIT_NEG,  TRIT_POS,  TRIT_ZERO, TRIT_ZERO);
    check_add(TRIT_ZERO, TRIT_ZERO, TRIT_ZERO, TRIT_ZERO);
    check_add(TRIT_POS,  TRIT_ZERO, TRIT_POS,  TRIT_ZERO);
    check_add(TRIT_POS,  TRIT_POS,  TRIT_ZERO, TRIT_POS);

    $display("\n── Trit Register");
    reg_d = TRIT_POS;  @(posedge clk); #1;
    if (reg_q === TRIT_POS)  pass_count++; else begin $display("FAIL reg POS");  fail_count++; end
    reg_d = TRIT_NEG;  @(posedge clk); #1;
    if (reg_q === TRIT_NEG)  pass_count++; else begin $display("FAIL reg NEG");  fail_count++; end
    reg_d = TRIT_ZERO; @(posedge clk); #1;
    if (reg_q === TRIT_ZERO) pass_count++; else begin $display("FAIL reg ZERO"); fail_count++; end

    #10;
    $display("\n═══════════════════════════════════");
    $display("  PASSED: %0d", pass_count);
    $display("  FAILED: %0d", fail_count);
    if (fail_count == 0)
      $display("  ALL TESTS PASSED");
    $display("═══════════════════════════════════");
    $finish;
  end

  initial begin #10000; $display("TIMEOUT"); $finish; end
endmodule
