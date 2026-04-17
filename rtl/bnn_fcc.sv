module bnn_fcc #(
    parameter int INPUT_DATA_WIDTH  = 8,
    parameter int INPUT_BUS_WIDTH   = 64,
    parameter int CONFIG_BUS_WIDTH  = 64,
    parameter int OUTPUT_DATA_WIDTH = 4,
    parameter int OUTPUT_BUS_WIDTH  = 8,

    parameter int TOTAL_LAYERS = 4,
    parameter int TOPOLOGY[TOTAL_LAYERS] = '{0: 784, 1: 256, 2: 256, 3: 10, default: 0},

    parameter int PARALLEL_INPUTS = 8,
    parameter int PARALLEL_NEURONS[TOTAL_LAYERS-1] = '{default: 8},

    localparam int THRESHOLD_DATA_WIDTH = 32
) (
    input logic clk,
    input logic rst,

    input  logic                          config_valid,
    output logic                          config_ready,
    input  logic [  CONFIG_BUS_WIDTH-1:0] config_data,
    input  logic [CONFIG_BUS_WIDTH/8-1:0] config_keep,
    input  logic                          config_last,

    input  logic                         data_in_valid,
    output logic                         data_in_ready,
    input  logic [  INPUT_BUS_WIDTH-1:0] data_in_data,
    input  logic [INPUT_BUS_WIDTH/8-1:0] data_in_keep,
    input  logic                         data_in_last,

    output logic                          data_out_valid,
    input  logic                          data_out_ready,
    output logic [  OUTPUT_BUS_WIDTH-1:0] data_out_data,
    output logic [OUTPUT_BUS_WIDTH/8-1:0] data_out_keep,
    output logic                          data_out_last
);

    localparam int LAYERS = TOTAL_LAYERS - 1;
    localparam int CONFIG_BYTES_PER_BEAT = CONFIG_BUS_WIDTH / 8;
    localparam int INPUT_BYTES_PER_BEAT  = INPUT_BUS_WIDTH / 8;
    localparam int INPUT_BIN_THRESHOLD   = 1 << (INPUT_DATA_WIDTH - 1);

    function automatic int get_max_neurons();
        int max_n;
        begin
            max_n = 0;
            for (int l = 0; l < LAYERS; l++) begin
                if (TOPOLOGY[l+1] > max_n) max_n = TOPOLOGY[l+1];
            end
            return max_n;
        end
    endfunction

    function automatic int get_max_fanin();
        int max_f;
        begin
            max_f = 0;
            for (int l = 0; l < LAYERS; l++) begin
                if (TOPOLOGY[l] > max_f) max_f = TOPOLOGY[l];
            end
            return max_f;
        end
    endfunction

    localparam int MAX_NEURONS = get_max_neurons();
    localparam int MAX_FANIN   = get_max_fanin();

    typedef enum logic [1:0] {
        S_CONFIG  = 2'd0,
        S_INPUT   = 2'd1,
        S_COMPUTE = 2'd2,
        S_OUTPUT  = 2'd3
    } state_t;

    state_t state;

    logic weights [0:LAYERS-1][0:MAX_NEURONS-1][0:MAX_FANIN-1];
    logic [THRESHOLD_DATA_WIDTH-1:0] thresholds [0:LAYERS-2][0:MAX_NEURONS-1];

    logic input_bits   [0:MAX_FANIN-1];
    logic layer_buf_a  [0:MAX_NEURONS-1];
    logic layer_buf_b  [0:MAX_NEURONS-1];
    logic [THRESHOLD_DATA_WIDTH-1:0] output_counts [0:MAX_NEURONS-1];

    logic [OUTPUT_DATA_WIDTH-1:0] predicted_class;
    logic                         output_valid_reg;

    logic [7:0] cfg_header [0:15];
    integer cfg_header_count;
    integer cfg_payload_count;
    integer cfg_total_bytes;
    integer cfg_bytes_per_neuron;
    integer cfg_num_neurons;
    integer cfg_layer_inputs;
    integer cfg_msg_type;
    integer cfg_layer_id;
    logic   cfg_in_payload;

    integer cfg_weight_byte_index;
    integer cfg_thresh_byte_index;

    integer image_pixels_received;

    integer l, n, f, b;
    integer tmp_count;
    integer pop;
    integer argmax_idx;
    integer argmax_max;

    initial begin
        if (INPUT_DATA_WIDTH != 8)  $fatal(1, "bnn_fcc currently requires INPUT_DATA_WIDTH=8");
        if (CONFIG_BUS_WIDTH % 8)   $fatal(1, "CONFIG_BUS_WIDTH must be byte aligned");
        if (INPUT_BUS_WIDTH % 8)    $fatal(1, "INPUT_BUS_WIDTH must be byte aligned");
        if (OUTPUT_BUS_WIDTH % 8)   $fatal(1, "OUTPUT_BUS_WIDTH must be byte aligned");
        if (OUTPUT_BUS_WIDTH < 8)   $fatal(1, "OUTPUT_BUS_WIDTH must be at least 8");
        if (TOTAL_LAYERS < 2)       $fatal(1, "TOTAL_LAYERS must be at least 2");
    end

    task automatic clear_storage;
        begin
            for (l = 0; l < LAYERS; l++) begin
                for (n = 0; n < MAX_NEURONS; n++) begin
                    for (f = 0; f < MAX_FANIN; f++) begin
                        weights[l][n][f] = 1'b0;
                    end
                end
            end

            for (l = 0; l < LAYERS-1; l++) begin
                for (n = 0; n < MAX_NEURONS; n++) begin
                    thresholds[l][n] = '0;
                end
            end

            for (f = 0; f < MAX_FANIN; f++) begin
                input_bits[f] = 1'b0;
            end

            for (n = 0; n < MAX_NEURONS; n++) begin
                layer_buf_a[n]   = 1'b0;
                layer_buf_b[n]   = 1'b0;
                output_counts[n] = '0;
            end
        end
    endtask

    task automatic decode_header;
        begin
            cfg_msg_type         = {24'd0, cfg_header[0]};
            cfg_layer_id         = {24'd0, cfg_header[1]};
            cfg_layer_inputs     = {16'd0, cfg_header[3], cfg_header[2]};
            cfg_num_neurons      = {16'd0, cfg_header[5], cfg_header[4]};
            cfg_bytes_per_neuron = {16'd0, cfg_header[7], cfg_header[6]};
            cfg_total_bytes      = {cfg_header[11], cfg_header[10], cfg_header[9], cfg_header[8]};
            cfg_payload_count    = 0;
            cfg_weight_byte_index = 0;
            cfg_thresh_byte_index = 0;
            cfg_in_payload       = 1'b1;
        end
    endtask

    task automatic parse_config_byte(input logic [7:0] byte_val);
        integer neuron_idx;
        integer byte_in_neuron;
        integer bit_base;
        integer bit_idx;
        integer thresh_idx;
        integer thresh_byte;
        begin
            if (!cfg_in_payload) begin
                cfg_header[cfg_header_count] = byte_val;
                cfg_header_count = cfg_header_count + 1;

                if (cfg_header_count == 16) begin
                    decode_header();
                    cfg_header_count = 0;
                end
            end else begin
                if (cfg_payload_count < cfg_total_bytes) begin
                    if (cfg_msg_type == 0) begin
                        neuron_idx     = cfg_weight_byte_index / cfg_bytes_per_neuron;
                        byte_in_neuron = cfg_weight_byte_index % cfg_bytes_per_neuron;
                        bit_base       = byte_in_neuron * 8;

                        if ((cfg_layer_id >= 0) && (cfg_layer_id < LAYERS) &&
                            (neuron_idx >= 0) && (neuron_idx < TOPOLOGY[cfg_layer_id + 1])) begin
                            for (bit_idx = 0; bit_idx < 8; bit_idx++) begin
                                if ((bit_base + bit_idx) < TOPOLOGY[cfg_layer_id]) begin
                                    weights[cfg_layer_id][neuron_idx][bit_base + bit_idx] = byte_val[bit_idx];
                                end
                            end
                        end

                        cfg_weight_byte_index = cfg_weight_byte_index + 1;
                    end else begin
                        thresh_idx  = cfg_thresh_byte_index >> 2;
                        thresh_byte = cfg_thresh_byte_index & 2'b11;

                        if ((cfg_layer_id >= 0) && (cfg_layer_id < (LAYERS - 1)) &&
                            (thresh_idx >= 0) && (thresh_idx < TOPOLOGY[cfg_layer_id + 1])) begin
                            thresholds[cfg_layer_id][thresh_idx][thresh_byte*8 +: 8] = byte_val;
                        end

                        cfg_thresh_byte_index = cfg_thresh_byte_index + 1;
                    end

                    cfg_payload_count = cfg_payload_count + 1;

                    if (cfg_payload_count == cfg_total_bytes) begin
                        cfg_in_payload = 1'b0;
                    end
                end
            end
        end
    endtask

    task automatic compute_network;
        begin
            for (n = 0; n < MAX_NEURONS; n++) begin
                layer_buf_a[n]   = 1'b0;
                layer_buf_b[n]   = 1'b0;
                output_counts[n] = '0;
            end

            for (l = 0; l < LAYERS; l++) begin
                for (n = 0; n < TOPOLOGY[l+1]; n++) begin
                    pop = 0;

                    for (f = 0; f < TOPOLOGY[l]; f++) begin
                        if (l == 0) begin
                            if (input_bits[f] == weights[l][n][f]) pop = pop + 1;
                        end else if ((l & 1) == 1) begin
                            if (layer_buf_a[f] == weights[l][n][f]) pop = pop + 1;
                        end else begin
                            if (layer_buf_b[f] == weights[l][n][f]) pop = pop + 1;
                        end
                    end

                    if (l == LAYERS - 1) begin
                        output_counts[n] = pop;
                    end else if (l == 0) begin
                        layer_buf_a[n] = (pop >= thresholds[l][n]);
                    end else if ((l & 1) == 1) begin
                        layer_buf_b[n] = (pop >= thresholds[l][n]);
                    end else begin
                        layer_buf_a[n] = (pop >= thresholds[l][n]);
                    end
                end
            end

            argmax_idx = 0;
            argmax_max = output_counts[0];

            for (n = 1; n < TOPOLOGY[LAYERS]; n++) begin
                if (output_counts[n] > argmax_max) begin
                    argmax_max = output_counts[n];
                    argmax_idx = n;
                end
            end

            predicted_class = argmax_idx[OUTPUT_DATA_WIDTH-1:0];
        end
    endtask

    assign config_ready   = (state == S_CONFIG);
    assign data_in_ready  = (state == S_INPUT);
    assign data_out_valid = output_valid_reg;
    assign data_out_last  = 1'b1;

    always_comb begin
        data_out_data = '0;
        data_out_data[OUTPUT_DATA_WIDTH-1:0] = predicted_class;

        data_out_keep = '0;
        data_out_keep[0] = 1'b1;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            clear_storage();

            state               <= S_CONFIG;
            output_valid_reg    <= 1'b0;
            predicted_class     <= '0;

            cfg_header_count    <= 0;
            cfg_payload_count   <= 0;
            cfg_total_bytes     <= 0;
            cfg_bytes_per_neuron<= 0;
            cfg_num_neurons     <= 0;
            cfg_layer_inputs    <= 0;
            cfg_msg_type        <= 0;
            cfg_layer_id        <= 0;
            cfg_in_payload      <= 1'b0;
            cfg_weight_byte_index <= 0;
            cfg_thresh_byte_index <= 0;

            image_pixels_received <= 0;
        end else begin
            case (state)
                S_CONFIG: begin
                    output_valid_reg <= 1'b0;

                    if (config_valid && config_ready) begin
                        for (b = 0; b < CONFIG_BYTES_PER_BEAT; b++) begin
                            if (config_keep[b]) begin
                                parse_config_byte(config_data[b*8 +: 8]);
                            end
                        end

                        if (config_last) begin
                            state <= S_INPUT;
                            image_pixels_received <= 0;
                        end
                    end
                end

                S_INPUT: begin
                    if (data_in_valid && data_in_ready) begin
                        tmp_count = image_pixels_received;

                        for (b = 0; b < INPUT_BYTES_PER_BEAT; b++) begin
                            if (data_in_keep[b]) begin
                                if (tmp_count < TOPOLOGY[0]) begin
                                    input_bits[tmp_count] <= (data_in_data[b*8 +: 8] >= INPUT_BIN_THRESHOLD);
                                end
                                tmp_count = tmp_count + 1;
                            end
                        end

                        image_pixels_received <= tmp_count;

                        if (data_in_last) begin
                            state <= S_COMPUTE;
                        end
                    end
                end

                S_COMPUTE: begin
                    compute_network();
                    output_valid_reg <= 1'b1;
                    state <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    if (data_out_valid && data_out_ready) begin
                        output_valid_reg <= 1'b0;
                        image_pixels_received <= 0;
                        state <= S_INPUT;
                    end
                end

                default: begin
                    state <= S_CONFIG;
                end
            endcase
        end
    end



endmodule
