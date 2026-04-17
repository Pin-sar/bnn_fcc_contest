`timescale 1ns / 100ps

module bnn_fcc_coverage_tb;

    bnn_fcc_tb #(
        .USE_CUSTOM_TOPOLOGY(0),
        .TOGGLE_DATA_OUT_READY(1'b1),
        .CONFIG_VALID_PROBABILITY(0.8),
        .DATA_IN_VALID_PROBABILITY(0.8),
        .NUM_TEST_IMAGES(50)
    ) TB();

endmodule
