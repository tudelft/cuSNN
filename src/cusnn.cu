#include "cusnn.cuh"
#include "cusnn_kernels.cuh"


#define MAX_THREADS 1024
#define MAX_BLOCKS 65535
#define MAX(a, b) ((a) > (b) ? (a) : (b))


/////////////////////////////////////////////////////////////////////////////////////////////////////
// NETWORK CLASSES
/////////////////////////////////////////////////////////////////////////////////////////////////////


/* NETWORK CLASS */
// constructor
Network::Network(const int inp_size[3], const float inp_scale[2], const float sim_step, float node_refrac,
                 float synapse_trace_init, bool inhibition, bool drop_delays, float drop_delays_th) {

    this->cnt_layers = 0;

    // simulation timestep
    this->h_sim_step = (float *) malloc(sizeof(float));
    cudaMalloc((void **)&this->d_sim_step, sizeof(float));
    this->h_sim_step[0] = sim_step;
    cudaMemcpy(this->d_sim_step, this->h_sim_step, sizeof(float), cudaMemcpyHostToDevice);

    // input scale
    for (int i = 0; i < 2; i++) this->inp_scale[i] = inp_scale[i];

    // input size
    this->h_inp_size = (int *) malloc(sizeof(int) * 3);
    cudaMalloc((void **)&this->d_inp_size, sizeof(int) * 3);
    this->h_inp_size[0] = inp_size[0]; // channels
    this->h_inp_size[1] = (int) ((float) inp_size[1] / this->inp_scale[0]); // height
    this->h_inp_size[2] = (int) ((float) inp_size[2] / this->inp_scale[1]); // width
    cudaMemcpy(this->d_inp_size, this->h_inp_size, sizeof(int) * 3, cudaMemcpyHostToDevice);

    // neuron and synapse params
    this->h_node_refrac = (float *) malloc(sizeof(float));
    this->h_synapse_trace_init = (float *) malloc(sizeof(float));
    cudaMalloc((void **)&this->d_node_refrac, sizeof(float));
    this->h_node_refrac[0] = node_refrac;
    this->h_synapse_trace_init[0] = synapse_trace_init;
    cudaMemcpy(this->d_node_refrac, this->h_node_refrac, sizeof(float), cudaMemcpyHostToDevice);

    // perpendicular inhibition
    this->inhibition = inhibition;

    // drop-delays mechanism
    this->drop_delays = drop_delays;
    this->h_drop_delays_th = (float *) malloc(sizeof(float));
    cudaMalloc((void **)&this->d_drop_delays_th, sizeof(float));
    this->h_drop_delays_th[0] = drop_delays_th;
    cudaMemcpy(this->d_drop_delays_th, this->h_drop_delays_th, sizeof(float), cudaMemcpyHostToDevice);

    // learning params
    this->learning = false;
    this->learning_type = 0;
}


// destructor
Network::~Network(){

    free(this->h_inputs);
    free(this->h_inp_size);
    free(this->h_sim_step);
    free(this->h_node_refrac);
    free(this->h_synapse_trace_init);
    free(this->h_length_delay_inp);

    cudaFree(this->d_inputs);
    cudaFree(this->d_inp_size);
    cudaFree(this->d_sim_step);
    cudaFree(this->d_node_refrac);
    cudaFree(this->d_length_delay_inp);

    // clean layer data
    free(this->h_layers);
    cudaFree(this->h_d_layers);
    cudaFree(this->d_d_layers);
}


// add layer to the architecture
void Network::add_layer(std::string layer_type, bool learning, bool load_weights, bool homeostasis, float Vth,
                        float decay, float alpha, float max_delay, int num_delays, float synapse_inh_scaling,
                        int rf_side, int out_channels, std::string padding, float w_init) {

    this->h_layers[this->cnt_layers] = new Layer(layer_type, learning, load_weights, homeostasis, Vth, decay,
                                                 alpha, max_delay, num_delays, synapse_inh_scaling, rf_side,
                                                 out_channels, padding, w_init, this->h_sim_step[0]);
    this->cnt_layers++;
}


/* LAYER CLASS */
// constructor
Layer::Layer(std::string layer_type, bool learning, bool load_weights, bool homeostasis, float Vth, float decay,
             float alpha, float max_delay, int num_delays, float synapse_inh_scaling, int rf_side, int out_channels,
             std::string padding, float w_init, float sim_step) {

    // layer type and indicator
    this->layer_type = -1;
    this->layer_type_str = layer_type;
    if (this->layer_type_str.compare("Conv2d") == 0) this->layer_type = 0;
    else if (this->layer_type_str.compare("Conv2dSep") == 0) this->layer_type = 1;
    else if (this->layer_type_str.compare("Dense") == 0) this->layer_type = 2;
    else if (this->layer_type_str.compare("Pooling") == 0) this->layer_type = 3;
    else if (this->layer_type_str.compare("Merge") == 0) this->layer_type = 4;

    // layer params
    this->learning = learning;
    this->learning_type = 0;
    this->load_weights = load_weights;
    this->homeostasis = homeostasis;
    this->node_Vth = Vth;
    this->decay = decay;
    this->alpha = alpha;
    this->max_delay = max_delay;
    this->num_delays = num_delays;
    this->synapse_inh_scaling = synapse_inh_scaling;
    this->synapse_w_init = w_init;
    this->out_channels = out_channels;
    this->rf_side = rf_side;
    this->padding_type = padding;
    this->active = false;
    this->firing_node = false;
    this->inhibition_spatial = false;
    this->threshold_diehl = false;

    // input delay
    this->length_delay_inp = (int) ceilf(this->max_delay / sim_step) + 1;
}


// destructor
Layer::~Layer() {

    free(this->h_delay_indices);
    free(this->h_synapse_pretrace);
    free(this->h_kernels_cnvg);
    free(this->h_stdp_precnt);

    cudaFree(this->d_delay_indices);
    cudaFree(this->d_max_kernel);
    cudaFree(this->d_synapse_pretrace);
    cudaFree(this->d_kernels_cnvg);
    cudaFree(this->d_stdp_precnt);

    // clean kernel data
    free(this->h_kernels);
    cudaFree(this->h_d_kernels);
    cudaFree(this->d_d_kernels);
}


// add kernel to a layer
void Layer::add_kernel(int out_node_kernel, int out_nodesep_kernel, int out_maps, int length_delay_out,
                       float node_refrac, int rf_side, int num_delays, float w_init, int kernel_channels,
                       float sim_step) {

    this->h_kernels[this->cnt_kernels] = new Kernel(out_node_kernel, out_nodesep_kernel, out_maps, length_delay_out,
                                                    node_refrac, rf_side, num_delays, w_init, kernel_channels,
                                                    sim_step);
    this->cnt_kernels++;
}


/* KERNEL CLASS */
// constructor
Kernel::Kernel(int out_node_kernel, int out_nodesep_kernel, int out_maps, int length_delay_out, float node_refrac,
               int rf_side, int num_delays, float w_init, int kernel_channels, float sim_step) {

    // output map data
    this->h_node_train = (int *) malloc(sizeof(int) * out_node_kernel * length_delay_out);
    this->h_node_posttrace = (float *) malloc(sizeof(float) * out_node_kernel);
    cudaMalloc((void **)&this->d_node_train, sizeof(int) * out_node_kernel * length_delay_out);
    cudaMalloc((void **)&this->d_node_posttrace, sizeof(float) * out_node_kernel);
    for (int i = 0; i < out_node_kernel; i++) {
        this->h_node_posttrace[i] = 0.f;
        for (int d = 0; d < length_delay_out; d++)
            this->h_node_train[i * length_delay_out + d] = 0;
    }
    cudaMemcpy(this->d_node_train, this->h_node_train,
               sizeof(int) * out_node_kernel * length_delay_out, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_node_posttrace, this->h_node_posttrace,
               sizeof(float) * out_node_kernel, cudaMemcpyHostToDevice);

    this->h_nodesep_train = (int *) malloc(sizeof(int) * out_nodesep_kernel);
    this->h_nodesep_V = (float *) malloc(sizeof(float) * out_nodesep_kernel);
    this->h_nodesep_refrac = (float *) malloc(sizeof(float) * out_nodesep_kernel);
    cudaMalloc((void **)&this->d_nodesep_train, sizeof(int) * out_nodesep_kernel);
    cudaMalloc((void **)&this->d_nodesep_V, sizeof(float) * out_nodesep_kernel);
    cudaMalloc((void **)&this->d_nodesep_channel_input, sizeof(float) * out_nodesep_kernel * kernel_channels);
    cudaMalloc((void **)&this->d_nodesep_input, sizeof(float) * out_nodesep_kernel);
    cudaMalloc((void **)&this->d_nodesep_refrac, sizeof(float) * out_nodesep_kernel);
    cudaMalloc((void **)&this->d_nodesep_pretrace, sizeof(float) * out_nodesep_kernel);
    cudaMalloc((void **)&this->d_nodesep_perpendicular, sizeof(int) * out_nodesep_kernel);
    cudaMalloc((void **)&this->d_nodesep_maxpretrace, sizeof(int) * out_nodesep_kernel);
    for (int i = 0; i < out_nodesep_kernel; i++) {
        this->h_nodesep_V[i] = 0.f;
        this->h_nodesep_train[i] = 0;
        this->h_nodesep_refrac[i] = node_refrac / sim_step;
    }
    cudaMemcpy(this->d_nodesep_train, this->h_nodesep_train,
               sizeof(int) * out_nodesep_kernel, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_nodesep_V, this->h_nodesep_V,
               sizeof(float) * out_nodesep_kernel, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_nodesep_refrac, this->h_nodesep_refrac,
               sizeof(float) * out_nodesep_kernel, cudaMemcpyHostToDevice);

    // synaptic weights
    this->h_weights_exc = (float *) malloc(sizeof(float) * kernel_channels * rf_side * rf_side * num_delays);
    this->h_weights_inh = (float *) malloc(sizeof(float) * kernel_channels * rf_side * rf_side * num_delays);
    cudaMalloc((void **)&this->d_weights_exc, sizeof(float) * kernel_channels * rf_side * rf_side * num_delays);
    cudaMalloc((void **)&this->d_weights_inh, sizeof(float) * kernel_channels * rf_side * rf_side * num_delays);
    for (int ch = 0; ch < kernel_channels; ch++) {
        for (int i = 0; i < rf_side * rf_side; i++) {
            for (int d = 0; d < num_delays; d++) {
                int idx = ch * rf_side * rf_side * num_delays + i * num_delays + d;
                this->h_weights_exc[idx] = w_init;
                this->h_weights_inh[idx] = 0.f;
            }
        }
    }
    cudaMemcpy(this->d_weights_exc, this->h_weights_exc,
               sizeof(float) * kernel_channels * rf_side * rf_side * num_delays, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_weights_inh, this->h_weights_inh,
               sizeof(float) * kernel_channels * rf_side * rf_side * num_delays, cudaMemcpyHostToDevice);

    // delay flags
    this->h_delay_active = (bool *) malloc(sizeof(bool) * num_delays);
    cudaMalloc((void **)&this->d_delay_active, sizeof(bool) * num_delays);
    cudaMalloc((void **)&this->d_sum_exc_weights, sizeof(float) * num_delays);
    for (int d = 0; d < num_delays; d++)
        this->h_delay_active[d] = true;
    this->num_delays_active = num_delays;
    cudaMemcpy(this->d_delay_active, this->h_delay_active, sizeof(bool) * num_delays, cudaMemcpyHostToDevice);

    // perpendicular inhibition
    cudaMalloc((void **)&this->d_V_max, sizeof(float) * out_maps);
    cudaMalloc((void **)&this->d_node_max, sizeof(int) * out_maps);
    cudaMalloc((void **)&this->d_max_channel, sizeof(int) * out_node_kernel);
}


// destructor
Kernel::~Kernel() {

    free(this->h_node_train);
    free(this->h_node_posttrace);
    free(this->h_nodesep_V);
    free(this->h_nodesep_refrac);
    free(this->h_nodesep_train);
    free(this->h_weights_exc);
    free(this->h_weights_inh);
    free(this->h_delay_active);
    free(this->h_stdp_paredes_objective);
    free(this->h_stdp_postcnt);
    free(this->h_threshold_diehl_nodesep_theta);

    cudaFree(this->d_node_train);
    cudaFree(this->d_node_posttrace);
    cudaFree(this->d_nodesep_perpendicular);
    cudaFree(this->d_nodesep_V);
    cudaFree(this->d_nodesep_channel_input);
    cudaFree(this->d_nodesep_input);
    cudaFree(this->d_nodesep_refrac);
    cudaFree(this->d_nodesep_pretrace);
    cudaFree(this->d_nodesep_train);
    cudaFree(this->d_nodesep_maxpretrace);
    cudaFree(this->d_V_max);
    cudaFree(this->d_node_max);
    cudaFree(this->d_max_channel);
    cudaFree(this->d_weights_exc);
    cudaFree(this->d_weights_inh);
    cudaFree(this->d_weights_exc_delta);
    cudaFree(this->d_weights_inh_delta);
    cudaFree(this->d_delay_active);
    cudaFree(this->d_sum_exc_weights);
    cudaFree(this->d_weights_delta_maps);
    cudaFree(this->d_weights_delta_maps_channels);
    cudaFree(this->d_stdp_paredes_objective_avg);
    cudaFree(this->d_stdp_paredes_objective);
    cudaFree(this->d_stdp_postcnt);
    cudaFree(this->d_threshold_diehl_nodesep_theta);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
// NETWORK CREATION
/////////////////////////////////////////////////////////////////////////////////////////////////////


// modify neural map sizes for Conv2DSep
void Network::modify_conv2dsep(int l) {

    // separable convolutional kernels
    this->h_layers[l]->kernel_channels = 1;
}


// modify neural map sizes for Dense
void Network::modify_dense(int l) {

    // receptive field equal to input size
    this->h_layers[l]->strides = 0;
    this->h_layers[l]->rf_side = MAX(this->h_layers[l]->inp_size[1], this->h_layers[l]->inp_size[2]);

    // output size
    this->h_layers[l]->out_size[1] = 1;
    this->h_layers[l]->out_size[2] = 1;
}


// modify neural map sizes for Pooling
void Network::modify_pooling(int l) {

    // params
    this->h_layers[l]->learning = false;
    this->h_layers[l]->synapse_inh_scaling = 0.f;
    this->h_layers[l]->inhibition = false;

    // stride size equal to receptive field size
    this->h_layers[l]->out_size[1] = this->h_layers[l]->inp_size[1] / this->h_layers[l]->rf_side;
    this->h_layers[l]->out_size[2] = this->h_layers[l]->inp_size[2] / this->h_layers[l]->rf_side;
    this->h_layers[l]->strides = this->h_layers[l]->rf_side;

    // same output and input channels
    this->h_layers[l]->out_channels = this->h_layers[l]->inp_size[0];
}


// modify neural map sizes for Merge
void Network::modify_merge(int l) {

    // params
    this->h_layers[l]->learning = false;
    this->h_layers[l]->rf_side = 1;
    this->h_layers[l]->synapse_inh_scaling = 0.f;
    this->h_layers[l]->inhibition = false;

    // single-channel output map
    this->h_layers[l]->out_channels = 1;

    // 'full' padding mode
    this->h_layers[l]->out_size[1] = this->h_layers[l]->inp_size[1];
    this->h_layers[l]->out_size[2] = this->h_layers[l]->inp_size[2];
    this->h_layers[l]->strides = 1;
}


// modify kernels for Pooling
void Network::modify_kernels_pooling(int l) {

    // excitatory weights
    for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++) {
        for (int ch = 0; ch < this->h_layers[l]->kernel_channels; ch++) {
            for (int i = 0; i < this->h_layers[l]->rf_side * this->h_layers[l]->rf_side; i++) {
                for (int d = 0; d < this->h_layers[l]->num_delays; d++) {
                    int idx = ch * this->h_layers[l]->rf_side * this->h_layers[l]->rf_side *
                            this->h_layers[l]->num_delays + i * this->h_layers[l]->num_delays + d;
                    if (k == ch) this->h_layers[l]->h_kernels[k]->h_weights_exc[idx] = 1.f;
                    else this->h_layers[l]->h_kernels[k]->h_weights_exc[idx] = 0.f;
                }
            }
        }
        cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_weights_exc, this->h_layers[l]->h_kernels[k]->h_weights_exc,
                   sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->rf_side *
                   this->h_layers[l]->rf_side * this->h_layers[l]->num_delays,
                   cudaMemcpyHostToDevice);
    }

    // kernel convergence
    for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++)
        this->h_layers[l]->h_kernels_cnvg[k] = true;
    cudaMemcpy(this->h_layers[l]->d_kernels_cnvg, this->h_layers[l]->h_kernels_cnvg,
               sizeof(bool) * this->h_layers[l]->cnt_kernels, cudaMemcpyHostToDevice);
}


// modify kernels for Merge
void Network::modify_kernels_merge(int l) {

    // excitatory weights
    for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++) {
        for (int ch = 0; ch < this->h_layers[l]->kernel_channels; ch++) {
            for (int i = 0; i < this->h_layers[l]->rf_side * this->h_layers[l]->rf_side; i++) {
                for (int d = 0; d < this->h_layers[l]->num_delays; d++) {
                    int idx = ch * this->h_layers[l]->rf_side * this->h_layers[l]->rf_side *
                            this->h_layers[l]->num_delays + i * this->h_layers[l]->num_delays + d;
                    this->h_layers[l]->h_kernels[k]->h_weights_exc[idx] = 1.f;
                }
            }
        }
        cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_weights_exc, this->h_layers[l]->h_kernels[k]->h_weights_exc,
                   sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->rf_side *
                   this->h_layers[l]->rf_side * this->h_layers[l]->num_delays, cudaMemcpyHostToDevice);
    }

    // kernel convergence
    for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++)
        this->h_layers[l]->h_kernels_cnvg[k] = true;
    cudaMemcpy(this->h_layers[l]->d_kernels_cnvg, this->h_layers[l]->h_kernels_cnvg,
               sizeof(bool) * this->h_layers[l]->cnt_kernels, cudaMemcpyHostToDevice);
}


// create network
void Network::create_network(bool& break_fun) {

    for (int l = 0; l < this->cnt_layers; l++) {

        // check layer parameters
        if (this->h_layers[l]->layer_type < 0) {
            printf("Error Layer %i: unrecognized layer type.\n", l);
            break_fun = true;
            return;
        } else if (this->h_layers[l]->node_Vth < 0.f) {
            printf("Error Layer %i: Vth has to be greater than or equal to zero.\n", l);
            break_fun = true;
            return;
        } else if (this->h_layers[l]->decay <= 0.f) {
            printf("Error Layer %i: decay has to be greater than zero.\n", l);
            break_fun = true;
            return;
        } else if (this->h_layers[l]->alpha < 0.f) {
            printf("Error Layer %i: alpha has to be greater than or equal to zero.\n", l);
            break_fun = true;
            return;
        } else if (this->h_layers[l]->max_delay <= 0.f) {
            printf("Error Layer %i: max_delay has to be greater than zero.\n", l);
            break_fun = true;
            return;
        } else if (this->h_layers[l]->num_delays <= 0) {
            printf("Error Layer %i: num_delays has to be greater than zero.\n", l);
            break_fun = true;
            return;
        } else if (this->h_layers[l]->synapse_inh_scaling < 0.f) {
            printf("Error Layer %i: synapse_inh_scaling has to be greater than or equal to zero.\n", l);
            break_fun = true;
            return;
        }

        // input synaptic delay
        this->h_layers[l]->h_delay_indices = (int *) malloc(sizeof(int) * this->h_layers[l]->num_delays);
        cudaMalloc((void **)&this->h_layers[l]->d_delay_indices, sizeof(int) * this->h_layers[l]->num_delays);

        if (this->h_layers[l]->num_delays == 1)
            this->h_layers[l]->h_delay_indices[0] = (int) ceilf(this->h_layers[l]->max_delay / this->h_sim_step[0]);
        else {
            float delta = (this->h_layers[l]->max_delay - 1.f) / ((float) this->h_layers[l]->num_delays - 1.f);
            for (int i = 0; i < this->h_layers[l]->num_delays - 1; i++)
                this->h_layers[l]->h_delay_indices[i] = (int) ceilf((1.f + delta * (float) i) / this->h_sim_step[0]);
            this->h_layers[l]->h_delay_indices[this->h_layers[l]->num_delays - 1] =
                    (int) ceilf(this->h_layers[l]->max_delay / this->h_sim_step[0]);
        }
        cudaMemcpy(this->h_layers[l]->d_delay_indices, this->h_layers[l]->h_delay_indices,
                   sizeof(int) * this->h_layers[l]->num_delays, cudaMemcpyHostToDevice);

        // input size
        if (!l) {
            for (int i = 0; i < 3; i++)
                this->h_layers[l]->inp_size[i] = this->h_inp_size[i];
        } else {
            for (int i = 0; i < 3; i++)
                this->h_layers[l]->inp_size[i] = this->h_layers[l-1]->out_size[i];
        }

        // kernel channels
        this->h_layers[l]->kernel_channels = this->h_layers[l]->inp_size[0];

        // output synaptic delay
        if (l != this->cnt_layers - 1)
            this->h_layers[l]->length_delay_out = this->h_layers[l+1]->length_delay_inp;
        else
            this->h_layers[l]->length_delay_out = this->h_layers[l]->length_delay_inp;

        // output size and strides
        if (this->h_layers[l]->padding_type.compare("half") == 0) {
            this->h_layers[l]->out_size[1] = this->h_layers[l]->inp_size[1] / 2;
            this->h_layers[l]->out_size[2] = this->h_layers[l]->inp_size[2] / 2;
            this->h_layers[l]->strides = 2;
        } else if (this->h_layers[l]->padding_type.compare("full") == 0) {
            this->h_layers[l]->out_size[1] = this->h_layers[l]->inp_size[1];
            this->h_layers[l]->out_size[2] = this->h_layers[l]->inp_size[2];
            this->h_layers[l]->strides = 1;
        } else if (this->h_layers[l]->padding_type.compare("none") == 0) {
            this->h_layers[l]->out_size[1] = this->h_layers[l]->inp_size[1] - this->h_layers[l]->rf_side + 1;
            this->h_layers[l]->out_size[2] = this->h_layers[l]->inp_size[2] - this->h_layers[l]->rf_side + 1;
            this->h_layers[l]->strides = 1;
        } else {
            printf("Error Layer %i: Unrecognized padding format.\n", l);
            break_fun = true;
            return;
        }

        // layer-specific inhibition flag
        this->h_layers[l]->inhibition = this->inhibition;

        // layer-specific settings (differences from conv2d layer)
        if (this->h_layers[l]->layer_type == 1) this->modify_conv2dsep(l);
        else if (this->h_layers[l]->layer_type == 2) this->modify_dense(l);
        else if (this->h_layers[l]->layer_type == 3) this->modify_pooling(l);
        else if (this->h_layers[l]->layer_type == 4) this->modify_merge(l);

        // output maps
        this->h_layers[l]->out_size[0] = this->h_layers[l]->out_channels;

        // check output size
        if (this->h_layers[l]->out_size[1] <= 0 || this->h_layers[l]->out_size[2] <= 0) {
            printf("Error Layer %i: The feature maps cannot be compressed any further.\n", l);
            break_fun = true;
            return;
        }

        // limit the kernel size
        for (int i = 0; i < 2; i++) this->h_layers[l]->rf_side_limits[i] = 0;
        if (this->h_layers[l]->inp_size[1] < this->h_layers[l]->rf_side)
            this->h_layers[l]->rf_side_limits[0] = this->h_layers[l]->rf_side - this->h_layers[l]->inp_size[1];
        if (this->h_layers[l]->inp_size[2] < this->h_layers[l]->rf_side)
            this->h_layers[l]->rf_side_limits[1] = this->h_layers[l]->rf_side - this->h_layers[l]->inp_size[2];

        // padding
        this->h_layers[l]->padding_total[0] = (int) MAX(((float) this->h_layers[l]->out_size[1] - 1) *
                (float) this->h_layers[l]->strides + (float) this->h_layers[l]->rf_side -
                this->h_layers[l]->rf_side_limits[0] - (float) this->h_layers[l]->inp_size[1], 0); // height padding
        this->h_layers[l]->padding_total[1] = (int) MAX(((float) this->h_layers[l]->out_size[2] - 1) *
                (float) this->h_layers[l]->strides + (float) this->h_layers[l]->rf_side -
                this->h_layers[l]->rf_side_limits[1] - (float) this->h_layers[l]->inp_size[2], 0); // width padding
        this->h_layers[l]->padding[0] = (int) floorf((float) this->h_layers[l]->padding_total[0] / 2.f); // top padding
        this->h_layers[l]->padding[1] = (int) floorf((float) this->h_layers[l]->padding_total[1] / 2.f); // left padding

        // number of output maps
        this->h_layers[l]->out_maps = this->h_layers[l]->inp_size[0] - this->h_layers[l]->kernel_channels + 1;

        // total number of node and synapses
        this->h_layers[l]->inp_node_kernel = (this->h_layers[l]->inp_size[1] + this->h_layers[l]->padding_total[0]) *
                (this->h_layers[l]->inp_size[2] + this->h_layers[l]->padding_total[1]);
        this->h_layers[l]->inp_node_total = this->h_layers[l]->inp_size[0] * this->h_layers[l]->inp_node_kernel;
        this->h_layers[l]->inp_synapses_total = this->h_layers[l]->inp_node_total * this->h_layers[l]->num_delays;
        this->h_layers[l]->out_node_kernel = this->h_layers[l]->out_size[1] * this->h_layers[l]->out_size[2];
        this->h_layers[l]->out_nodesep_kernel = this->h_layers[l]->out_maps * this->h_layers[l]->out_node_kernel;

        // spatial perpendicular inhibition
        if (!this->h_layers[l]->strides) this->h_layers[l]->neigh_inh = 0;
        else {
            int max_rf_side_limit = MAX(this->h_layers[l]->rf_side_limits[0], this->h_layers[l]->rf_side_limits[1]);
            this->h_layers[l]->neigh_inh = (int) ceilf((float) (this->h_layers[l]->rf_side - max_rf_side_limit) /
                                                       (float) this->h_layers[l]->strides) - 1;
        }
        cudaMalloc((void **)&this->h_layers[l]->d_max_kernel, sizeof(int) * this->h_layers[l]->out_node_kernel);

        // pre-synaptic trace
        this->h_layers[l]->h_synapse_pretrace = (float *) malloc(sizeof(float) * this->h_layers[l]->inp_synapses_total);
        cudaMalloc((void **)&this->h_layers[l]->d_synapse_pretrace, sizeof(float) *
                   this->h_layers[l]->inp_synapses_total);
        for (int i = 0; i < this->h_layers[l]->inp_synapses_total; i++)
            this->h_layers[l]->h_synapse_pretrace[i] = this->h_synapse_trace_init[0];
        cudaMemcpy(this->h_layers[l]->d_synapse_pretrace, this->h_layers[l]->h_synapse_pretrace,
                   sizeof(float) * this->h_layers[l]->inp_synapses_total, cudaMemcpyHostToDevice);

        // kernel initialization
        this->h_layers[l]->cnt_kernels = 0;
        this->h_layers[l]->h_kernels = (Kernel **) malloc(sizeof(Kernel*) * this->h_layers[l]->out_size[0]);
        for (int i = 0; i < this->h_layers[l]->out_size[0]; i++)
            this->h_layers[l]->add_kernel(this->h_layers[l]->out_node_kernel, this->h_layers[l]->out_nodesep_kernel,
                                          this->h_layers[l]->out_maps, this->h_layers[l]->length_delay_out,
                                          this->h_node_refrac[0], this->h_layers[l]->rf_side,
                                          this->h_layers[l]->num_delays, this->h_layers[l]->synapse_w_init,
                                          this->h_layers[l]->kernel_channels, this->h_sim_step[0]);

        // kernel convergence
        this->h_layers[l]->h_kernels_cnvg = (bool *) malloc(sizeof(bool) * this->h_layers[l]->cnt_kernels);
        cudaMalloc((void **)&this->h_layers[l]->d_kernels_cnvg, sizeof(bool) * this->h_layers[l]->cnt_kernels);
        for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++)
            this->h_layers[l]->h_kernels_cnvg[k] = false;
        cudaMemcpy(this->h_layers[l]->d_kernels_cnvg, this->h_layers[l]->h_kernels_cnvg,
                   sizeof(bool) * this->h_layers[l]->cnt_kernels, cudaMemcpyHostToDevice);

        // set STDP enable flag after layer initialization
        this->h_layers[l]->enable_learning = false;
        if (this->h_layers[l]->learning) {
            this->learning = true;
            this->h_layers[l]->learning = false;
            this->h_layers[l]->enable_learning = true;
            this->h_layers[l]->enable_learning_cnt = 0;
        }
        this->h_layers[l]->limit_learning = false;
        this->h_layers[l]->learning_updates_cnt = 0;
        this->h_layers[l]->learning_warmup_time = 0;
        for (int l2 = 0; l2 <= l; l2++) {
            if (2 * this->h_layers[l2]->length_delay_inp > this->h_layers[l]->learning_warmup_time)
                this->h_layers[l]->learning_warmup_time = 2 * this->h_layers[l2]->length_delay_inp;
        }

        // layer-specific kernel settings (differences from conv2d layer)
        if (this->h_layers[l]->layer_type == 3) this->modify_kernels_pooling(l);
        if (this->h_layers[l]->layer_type == 4) this->modify_kernels_merge(l);

        // kernels to device memory
        this->h_layers[l]->h_d_kernels = (Kernel **) malloc(sizeof(Kernel*) * this->h_layers[l]->out_size[0]);
        for (int i = 0; i < this->h_layers[l]->out_size[0]; i++) {
            cudaMalloc((void**)&this->h_layers[l]->h_d_kernels[i], sizeof(Kernel));
            cudaMemcpy(this->h_layers[l]->h_d_kernels[i], this->h_layers[l]->h_kernels[i],
                       sizeof(Kernel), cudaMemcpyHostToDevice);
        }
        cudaMalloc((void**)&this->h_layers[l]->d_d_kernels, sizeof(Kernel*) * this->h_layers[l]->out_size[0]);
        cudaMemcpy(this->h_layers[l]->d_d_kernels, this->h_layers[l]->h_d_kernels,
                   sizeof(Kernel*) * this->h_layers[l]->out_size[0], cudaMemcpyHostToDevice);
    }

    // network structure device versions
    this->h_d_layers = (Layer **) malloc(sizeof(Layer*) * this->cnt_layers);
    for (int i = 0; i < this->cnt_layers; i++) {
        cudaMalloc((void**)&this->h_d_layers[i], sizeof(Layer));
        cudaMemcpy(this->h_d_layers[i], this->h_layers[i], sizeof(Layer), cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&this->d_d_layers, sizeof(Layer*) * this->cnt_layers);
    cudaMemcpy(this->d_d_layers, this->h_d_layers, sizeof(Layer*) * this->cnt_layers, cudaMemcpyHostToDevice);

    // synapse delay for input data
    this->h_length_delay_inp = (int *) malloc(sizeof(int));
    cudaMalloc((void **)&this->d_length_delay_inp, sizeof(int));
    if (this->cnt_layers > 0) this->h_length_delay_inp[0] = this->h_layers[0]->length_delay_inp;
    else this->h_length_delay_inp[0] = 1;
    cudaMemcpy(this->d_length_delay_inp, this->h_length_delay_inp, sizeof(int), cudaMemcpyHostToDevice);

    // vectors for input data
    this->h_inputs = (int *) malloc(sizeof(int) * this->h_inp_size[0] * this->h_inp_size[1] * this->h_inp_size[2] *
                                    this->h_length_delay_inp[0]);
    cudaMalloc((void **)&this->d_inputs, sizeof(int) * this->h_inp_size[0] * this->h_inp_size[1] * this->h_inp_size[2] *
               this->h_length_delay_inp[0]);
    for (int ch = 0; ch < this->h_inp_size[0]; ch++) {
        for (int i = 0; i < this->h_inp_size[1] * this->h_inp_size[2]; i++) {
            for (int d = 0; d < this->h_length_delay_inp[0]; d++) {
                int idx = ch * this->h_inp_size[1] * this->h_inp_size[2] * this->h_length_delay_inp[0] +
                        i * this->h_length_delay_inp[0] + d;
                this->h_inputs[idx] = 0;
            }
        }
    }
    cudaMemcpy(this->d_inputs, this->h_inputs,
               sizeof(int) * this->h_inp_size[0] * this->h_inp_size[1] * this->h_inp_size[2] *
               this->h_length_delay_inp[0], cudaMemcpyHostToDevice);

    // get maximums dimensions for CUDA kernels
    this->max_inputs = 0;
    this->max_channels = 0;
    this->max_kernels = 0;
    this->max_outputs = 0;
    this->max_delays = 0;
    for (int l = 0; l < this->cnt_layers; l++) {
        if (this->max_inputs < this->h_layers[l]->inp_node_kernel)
            this->max_inputs = this->h_layers[l]->inp_node_kernel;
        if (this->max_channels < this->h_layers[l]->inp_size[0])
            this->max_channels = this->h_layers[l]->inp_size[0];
        if (this->max_kernels < this->h_layers[l]->out_size[0])
            this->max_kernels = this->h_layers[l]->out_size[0];
        if (this->max_outputs < this->h_layers[l]->out_node_kernel)
            this->max_outputs = this->h_layers[l]->out_node_kernel;
        if (this->max_delays < this->h_layers[l]->num_delays)
            this->max_delays = this->h_layers[l]->num_delays;
    }
    if (this->max_inputs > MAX_BLOCKS) {
        printf("Error: max_inputs has to be equal or lower than 65535.\n");
        break_fun = true;
        return;
    } else if (this->max_outputs > MAX_BLOCKS) {
        printf("Error: max_outputs has to be equal or lower than 65535.\n");
        break_fun = true;
        return;
    } else if (this->max_channels > MAX_BLOCKS) {
        printf("Error: max_channels has to be equal or lower than 65535.\n");
        break_fun = true;
        return;
    } else if (this->max_delays > MAX_THREADS) {
        printf("Error: max_delays has to be equal or lower than 1024.\n");
        break_fun = true;
        return;
    } else if (this->max_kernels > MAX_THREADS) {
        printf("Error: max_kernels has to be equal or lower than 1024.\n");
        break_fun = true;
        return;
    }

    // CUDA blocks and threads dimensions
    this->block_0 = dim3((unsigned int) this->h_inp_size[0],
                         (unsigned int) this->h_inp_size[1],
                         (unsigned int) this->h_inp_size[2]);
    this->block_1 = dim3((unsigned int) this->cnt_layers,
                         1,
                         1);
    this->block_2 = dim3((unsigned int) this->cnt_layers,
                         (unsigned int) this->max_inputs,
                         (unsigned int) this->max_channels);
    this->block_3 = dim3((unsigned int) this->cnt_layers,
                         (unsigned int) this->max_outputs,
                         (unsigned int) this->max_channels);
    this->block_4 = dim3((unsigned int) this->cnt_layers,
                         (unsigned int) this->max_outputs,
                         1);
    this->block_5 = dim3((unsigned int) this->cnt_layers,
                         (unsigned int) this->max_channels,
                         1);
    this->block_6 = dim3((unsigned int) this->cnt_layers,
                         (unsigned int) this->max_channels,
                         (unsigned int) this->max_delays);
    this->thread_0 = dim3((unsigned int) this->max_kernels,
                          1,
                          1);
    this->thread_1 = dim3((unsigned int) this->max_delays,
                          1,
                          1);
}


// assign the Paredes' STDP params
void Network::enable_stdp_paredes(float learning_rate, float scale_a, float convg_th, int limit_updates,
                                  bool inhibition_spatial, int stats_window, bool& break_fun) {

    if (break_fun) return;
    if (learning_rate <= 0.f) {
        printf("Error STDP: learning_rate has to be greater than zero.\n");
        break_fun = true;
        return;
    } else if (stats_window <= 0.f) {
        printf("Error STDP: stats_window has to be greater than zero.\n");
        break_fun = true;
        return;
    } else if (convg_th < 0.f) {
        printf("Error STDP: convg_th has to be greater than or equal to zero.\n");
        break_fun = true;
        return;
    } else if (scale_a >= 1.f) {
        printf("Error STDP: scale_a has to be lower than one for stability.\n");
        break_fun = true;
        return;
    } else if (this->learning_type != 0) {
        printf("Error Learning: only one learning rule can be active.\n");
        break_fun = true;
        return;
    }

    this->learning_type = 1;
    for (int l = 0; l < this->cnt_layers; l++) {
        if (!this->h_layers[l]->enable_learning) continue;
        this->h_layers[l]->learning_type = this->learning_type;
        this->h_layers[l]->learning_rate = learning_rate;
        this->h_layers[l]->inhibition_spatial = inhibition_spatial;
        this->h_layers[l]->learning_limit_updates = limit_updates;
        this->h_layers[l]->stdp_paredes_stats_window = stats_window;
        this->h_layers[l]->stdp_paredes_convg_th = convg_th;
        this->h_layers[l]->stdp_paredes_a = scale_a;

        for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++) {

            // channels contributing to weight update
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_delta_maps,
                       sizeof(bool) * this->h_layers[l]->out_maps);
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_delta_maps_channels,
                       sizeof(bool) * this->h_layers[l]->out_maps * this->h_layers[l]->kernel_channels);

            // convergence metrics
            this->h_layers[l]->h_kernels[k]->stdp_paredes_objective_avg = 0.f;
            this->h_layers[l]->h_kernels[k]->h_stdp_paredes_objective = (float *) malloc(sizeof(float) * stats_window);
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_stdp_paredes_objective,
                       sizeof(float) * stats_window);
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_stdp_paredes_objective_avg,
                       sizeof(float) * this->h_layers[l]->out_maps);
            for (int i = 0; i < stats_window; i++)
                this->h_layers[l]->h_kernels[k]->h_stdp_paredes_objective[i] = 0.25f;
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_stdp_paredes_objective,
                       this->h_layers[l]->h_kernels[k]->h_stdp_paredes_objective,
                       sizeof(float) * stats_window, cudaMemcpyHostToDevice);

            // weight updates
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_exc_delta,
                       sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->out_maps *
                       this->h_layers[l]->rf_side * this->h_layers[l]->rf_side * this->h_layers[l]->num_delays);
            if (this->h_layers[l]->synapse_inh_scaling > 0.f)
                cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_inh_delta,
                           sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->out_maps *
                           this->h_layers[l]->rf_side * this->h_layers[l]->rf_side * this->h_layers[l]->num_delays);

            cudaMemcpy(this->h_layers[l]->h_d_kernels[k], this->h_layers[l]->h_kernels[k],
                       sizeof(Kernel), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(this->h_layers[l]->d_d_kernels, this->h_layers[l]->h_d_kernels,
                   sizeof(Kernel*) * this->h_layers[l]->cnt_kernels, cudaMemcpyHostToDevice);
        cudaMemcpy(this->h_d_layers[l], this->h_layers[l], sizeof(Layer), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(this->d_d_layers, this->h_d_layers, sizeof(Layer*) * this->cnt_layers, cudaMemcpyHostToDevice);
}


// assign the Shrestha's STDP params
void Network::enable_stdp_shrestha(float learning_rate, float window_LTP, bool weight_boundaries, float weight_max,
                                   int limit_updates, bool inhibition_spatial, bool& break_fun) {

    if (break_fun) return;
    if (learning_rate <= 0.f) {
        printf("Error STDP: learning_rate has to be greater than zero.\n");
        break_fun = true;
        return;
    } else if (window_LTP <= 0) {
        printf("Error STDP: window_LTP has to be greater than zero.\n");
        break_fun = true;
        return;
    } else if (this->learning_type != 0) {
        printf("Error Learning: only one learning rule can be active.\n");
        break_fun = true;
        return;
    } else if (weight_max <= 0) {
        printf("Error STDP: weight_max has to be greater than zero.\n");
        break_fun = true;
        return;
    }

    this->learning_type = 2;
    for (int l = 0; l < this->cnt_layers; l++) {
        if (!this->h_layers[l]->enable_learning) continue;
        this->h_layers[l]->learning_type = this->learning_type;
        this->h_layers[l]->learning_rate = learning_rate;
        this->h_layers[l]->stdp_shrestha_gerstner_weight_max = weight_max;
        this->h_layers[l]->inhibition_spatial = inhibition_spatial;
        this->h_layers[l]->learning_limit_updates = limit_updates;
        this->h_layers[l]->stdp_shrestha_gerstner_weight_boundaries = weight_boundaries;
        this->h_layers[l]->stdp_shrestha_gerstner_window_LTP = (int) ceilf(window_LTP / this->h_sim_step[0]);

        // counters since last presynaptic spike
        this->h_layers[l]->h_stdp_precnt = (int *) malloc(sizeof(int) * this->h_layers[l]->inp_synapses_total);
        cudaMalloc((void **)&this->h_layers[l]->d_stdp_precnt, sizeof(int) * this->h_layers[l]->inp_synapses_total);
        for (int i = 0; i < this->h_layers[l]->inp_synapses_total; i++)
            this->h_layers[l]->h_stdp_precnt[i] = -1;
        cudaMemcpy(this->h_layers[l]->d_stdp_precnt, this->h_layers[l]->h_stdp_precnt,
                   sizeof(int) * this->h_layers[l]->inp_synapses_total, cudaMemcpyHostToDevice);

        for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++) {

            // channels contributing to weight update
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_delta_maps_channels,
                       sizeof(bool) * this->h_layers[l]->out_maps * this->h_layers[l]->kernel_channels);

            // weight updates
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_exc_delta,
                       sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->out_maps *
                       this->h_layers[l]->rf_side * this->h_layers[l]->rf_side * this->h_layers[l]->num_delays);
            if (this->h_layers[l]->synapse_inh_scaling > 0.f)
                cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_inh_delta,
                           sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->out_maps *
                           this->h_layers[l]->rf_side * this->h_layers[l]->rf_side * this->h_layers[l]->num_delays);

            cudaMemcpy(this->h_layers[l]->h_d_kernels[k], this->h_layers[l]->h_kernels[k],
                       sizeof(Kernel), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(this->h_layers[l]->d_d_kernels, this->h_layers[l]->h_d_kernels,
                   sizeof(Kernel*) * this->h_layers[l]->cnt_kernels, cudaMemcpyHostToDevice);
        cudaMemcpy(this->h_d_layers[l], this->h_layers[l], sizeof(Layer), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(this->d_d_layers, this->h_d_layers, sizeof(Layer*) * this->cnt_layers, cudaMemcpyHostToDevice);
}


// assign the Gerstner's STDP params
void Network::enable_stdp_gerstner(float learning_rate_exc, float window_LTP, float window_LTD, bool weight_dependence,
                                   bool weight_boundaries, float weight_max, int limit_updates, bool inhibition_spatial,
                                   bool& break_fun) {

    if (break_fun) return;
    if (learning_rate_exc <= 0.f) {
        printf("Error STDP: learning_rate_exc has to be greater than zero.\n");
        break_fun = true;
        return;
    } else if (window_LTP <= 0) {
        printf("Error STDP: window_LTP has to be greater than zero.\n");
        break_fun = true;
        return;
    } else if (window_LTD <= 0) {
        printf("Error STDP: window_LTD has to be greater than zero.\n");
        break_fun = true;
        return;
    } else if (weight_max <= 0) {
        printf("Error STDP: weight_max has to be greater than zero.\n");
        break_fun = true;
        return;
    } else if (this->learning_type != 0) {
        printf("Error Learning: only one learning rule can be active.\n");
        break_fun = true;
        return;
    }

    this->learning_type = 3;
    for (int l = 0; l < this->cnt_layers; l++) {
        if (!this->h_layers[l]->enable_learning) continue;
        this->h_layers[l]->learning_type = this->learning_type;
        this->h_layers[l]->learning_rate = learning_rate_exc;
        this->h_layers[l]->stdp_shrestha_gerstner_weight_max = weight_max;
        this->h_layers[l]->inhibition_spatial = inhibition_spatial;
        this->h_layers[l]->learning_limit_updates = limit_updates;
        this->h_layers[l]->stdp_gerstner_weight_dependence = weight_dependence;
        this->h_layers[l]->stdp_shrestha_gerstner_weight_boundaries = weight_boundaries;
        this->h_layers[l]->stdp_shrestha_gerstner_window_LTP = (int) ceilf(window_LTP / this->h_sim_step[0]);
        this->h_layers[l]->stdp_shrestha_gerstner_window_LTD = (int) ceilf(window_LTD / this->h_sim_step[0]);

        // counters since last presynaptic spike
        this->h_layers[l]->h_stdp_precnt = (int *) malloc(sizeof(int) * this->h_layers[l]->inp_synapses_total);
        cudaMalloc((void **)&this->h_layers[l]->d_stdp_precnt, sizeof(int) * this->h_layers[l]->inp_synapses_total);
        for (int i = 0; i < this->h_layers[l]->inp_synapses_total; i++)
            this->h_layers[l]->h_stdp_precnt[i] = -1;
        cudaMemcpy(this->h_layers[l]->d_stdp_precnt, this->h_layers[l]->h_stdp_precnt,
                   sizeof(int) * this->h_layers[l]->inp_synapses_total, cudaMemcpyHostToDevice);

        for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++) {

            // channels contributing to weight update
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_delta_maps_channels,
                       sizeof(bool) * this->h_layers[l]->out_maps * this->h_layers[l]->kernel_channels);

            // counters since last postsynaptic spike
            this->h_layers[l]->h_kernels[k]->h_stdp_postcnt =
                    (int *) malloc(sizeof(int) * this->h_layers[l]->out_nodesep_kernel);
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_stdp_postcnt,
                       sizeof(int) * this->h_layers[l]->out_nodesep_kernel);
            for (int i = 0; i < this->h_layers[l]->out_nodesep_kernel; i++)
                this->h_layers[l]->h_kernels[k]->h_stdp_postcnt[i] = -1;
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_stdp_postcnt, this->h_layers[l]->h_kernels[k]->h_stdp_postcnt,
                       sizeof(int) * this->h_layers[l]->out_nodesep_kernel, cudaMemcpyHostToDevice);

            // weight updates
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_exc_delta,
                       sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->out_maps *
                       this->h_layers[l]->rf_side * this->h_layers[l]->rf_side * this->h_layers[l]->num_delays);
            if (this->h_layers[l]->synapse_inh_scaling > 0.f)
                cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_inh_delta,
                           sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->out_maps *
                           this->h_layers[l]->rf_side * this->h_layers[l]->rf_side * this->h_layers[l]->num_delays);

            cudaMemcpy(this->h_layers[l]->h_d_kernels[k], this->h_layers[l]->h_kernels[k],
                       sizeof(Kernel), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(this->h_layers[l]->d_d_kernels, this->h_layers[l]->h_d_kernels,
                   sizeof(Kernel*) * this->h_layers[l]->cnt_kernels, cudaMemcpyHostToDevice);
        cudaMemcpy(this->h_d_layers[l], this->h_layers[l], sizeof(Layer), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(this->d_d_layers, this->h_d_layers, sizeof(Layer*) * this->cnt_layers, cudaMemcpyHostToDevice);
}


// assign the Kheradpisheh's STDP params
void Network::enable_stdp_kheradpisheh(float learning_rate_exc, int limit_updates, bool inhibition_spatial,
                                       bool& break_fun) {

    if (break_fun) return;
    if (learning_rate_exc <= 0.f) {
        printf("Error STDP: learning_rate_exc has to be greater than zero.\n");
        break_fun = true;
        return;
    } else if (this->learning_type != 0) {
        printf("Error Learning: only one learning rule can be active.\n");
        break_fun = true;
        return;
    }

    this->learning_type = 4;
    for (int l = 0; l < this->cnt_layers; l++) {
        if (!this->h_layers[l]->enable_learning) continue;
        this->h_layers[l]->learning_type = this->learning_type;
        this->h_layers[l]->learning_rate = learning_rate_exc;
        this->h_layers[l]->inhibition_spatial = inhibition_spatial;
        this->h_layers[l]->learning_limit_updates = limit_updates;

        // counters since last presynaptic spike
        this->h_layers[l]->h_stdp_precnt = (int *) malloc(sizeof(int) * this->h_layers[l]->inp_synapses_total);
        cudaMalloc((void **)&this->h_layers[l]->d_stdp_precnt, sizeof(int) * this->h_layers[l]->inp_synapses_total);
        for (int i = 0; i < this->h_layers[l]->inp_synapses_total; i++)
            this->h_layers[l]->h_stdp_precnt[i] = -1;
        cudaMemcpy(this->h_layers[l]->d_stdp_precnt, this->h_layers[l]->h_stdp_precnt,
                   sizeof(int) * this->h_layers[l]->inp_synapses_total, cudaMemcpyHostToDevice);

        for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++) {

            // channels contributing to weight update
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_delta_maps_channels,
                       sizeof(bool) * this->h_layers[l]->out_maps * this->h_layers[l]->kernel_channels);

            // counters since last postsynaptic spike
            this->h_layers[l]->h_kernels[k]->h_stdp_postcnt =
                    (int *) malloc(sizeof(int) * this->h_layers[l]->out_nodesep_kernel);
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_stdp_postcnt,
                    sizeof(int) * this->h_layers[l]->out_nodesep_kernel);
            for (int i = 0; i < this->h_layers[l]->out_nodesep_kernel; i++)
                this->h_layers[l]->h_kernels[k]->h_stdp_postcnt[i] = -1;
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_stdp_postcnt, this->h_layers[l]->h_kernels[k]->h_stdp_postcnt,
                       sizeof(int) * this->h_layers[l]->out_nodesep_kernel, cudaMemcpyHostToDevice);

            // weight updates
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_exc_delta,
                       sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->out_maps *
                       this->h_layers[l]->rf_side * this->h_layers[l]->rf_side * this->h_layers[l]->num_delays);
            if (this->h_layers[l]->synapse_inh_scaling > 0.f)
                cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_weights_inh_delta,
                           sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->out_maps *
                           this->h_layers[l]->rf_side * this->h_layers[l]->rf_side * this->h_layers[l]->num_delays);

            cudaMemcpy(this->h_layers[l]->h_d_kernels[k], this->h_layers[l]->h_kernels[k],
                       sizeof(Kernel), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(this->h_layers[l]->d_d_kernels, this->h_layers[l]->h_d_kernels,
                   sizeof(Kernel*) * this->h_layers[l]->cnt_kernels, cudaMemcpyHostToDevice);
        cudaMemcpy(this->h_d_layers[l], this->h_layers[l], sizeof(Layer), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(this->d_d_layers, this->h_d_layers, sizeof(Layer*) * this->cnt_layers, cudaMemcpyHostToDevice);
}


// assign Diehl's adaptive threshold params
void Network::enable_adaptive_threshold_diehl(float threshold_delta, bool& break_fun) {

    if (break_fun) return;
    if (threshold_delta < 0.f) {
        printf("Error Adaptive Threshold: threshold_delta has to be greater or equal to zero.\n");
        break_fun = true;
        return;
    }

    for (int l = 0; l < this->cnt_layers; l++) {
        this->h_layers[l]->threshold_diehl = true;
        this->h_layers[l]->threshold_diehl_increase = threshold_delta;
        if (this->h_layers[l]->homeostasis && this->h_layers[l]->threshold_diehl)
            printf("Warning Layer %i: Using Paredes' and Diehl's homeostasis mechanism together.\n\n", l);

        for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++) {
            this->h_layers[l]->h_kernels[k]->h_threshold_diehl_nodesep_theta =
                    (float *) malloc(sizeof(float) * this->h_layers[l]->out_nodesep_kernel);
            cudaMalloc((void **)&this->h_layers[l]->h_kernels[k]->d_threshold_diehl_nodesep_theta,
                       sizeof(float) * this->h_layers[l]->out_nodesep_kernel);
            for (int i = 0; i < this->h_layers[l]->out_nodesep_kernel; i++)
                this->h_layers[l]->h_kernels[k]->h_threshold_diehl_nodesep_theta[i] = 0.f;
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_threshold_diehl_nodesep_theta,
                       this->h_layers[l]->h_kernels[k]->h_threshold_diehl_nodesep_theta,
                       sizeof(float) * this->h_layers[l]->out_nodesep_kernel, cudaMemcpyHostToDevice);

            cudaMemcpy(this->h_layers[l]->h_d_kernels[k], this->h_layers[l]->h_kernels[k],
                       sizeof(Kernel), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(this->h_layers[l]->d_d_kernels, this->h_layers[l]->h_d_kernels,
                   sizeof(Kernel*) * this->h_layers[l]->cnt_kernels, cudaMemcpyHostToDevice);
        cudaMemcpy(this->h_d_layers[l], this->h_layers[l], sizeof(Layer), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(this->d_d_layers, this->h_d_layers, sizeof(Layer*) * this->cnt_layers, cudaMemcpyHostToDevice);
}


// model's summary
void Network::summary() {

    int neurons = 0, trainable = 0, nontrainable = 0;
    printf("----------------------------------------------------------\n");
    printf("Layer (Type) \t\t Output Shape \t Param #\n");
    printf("==========================================================\n");
    printf("Input \t\t\t (%i, %i, %i)\n", this->h_inp_size[0], this->h_inp_size[1], this->h_inp_size[2]);
    printf("----------------------------------------------------------\n");
    for (int l = 0; l < this->cnt_layers; l++) {
        int params = (this->h_layers[l]->rf_side - this->h_layers[l]->rf_side_limits[0]) * 
                (this->h_layers[l]->rf_side - this->h_layers[l]->rf_side_limits[1]) * this->h_layers[l]->num_delays;
        for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++) {
            if (!this->h_layers[l]->h_kernels_cnvg[k] && this->h_layers[l]->enable_learning) trainable += params;
            else nontrainable += params;
            if (this->h_layers[l]->synapse_inh_scaling > 0.f) {
                if (!this->h_layers[l]->h_kernels_cnvg[k] && this->h_layers[l]->enable_learning) trainable += params;
                else nontrainable += params;
            }
        }
        neurons += this->h_layers[l]->out_nodesep_kernel * this->h_layers[l]->out_size[0];

        printf("layer_%i (%s) \t (%i, %i, %i) \t (%i, %i, %i, %i, %i)\n", l,
               this->h_layers[l]->layer_type_str.c_str(), this->h_layers[l]->out_size[0],
               this->h_layers[l]->out_size[1], this->h_layers[l]->out_size[2], this->h_layers[l]->out_size[0],
               this->h_layers[l]->synapse_inh_scaling > 0.f ? 2 : 1, this->h_layers[l]->num_delays,
               this->h_layers[l]->rf_side - this->h_layers[l]->rf_side_limits[0],
               this->h_layers[l]->rf_side - this->h_layers[l]->rf_side_limits[1]);
        if (l != this->cnt_layers - 1)
            printf("----------------------------------------------------------\n");
        else
            printf("==========================================================\n");
    }
    printf("Total neurons: %i\n", neurons);
    printf("Total params: %i\n", trainable + nontrainable);
    printf("Trainable params: %i\n", trainable);
    printf("Non-trainable params: %i\n\n", nontrainable);

    for (int l = 0; l < this->cnt_layers; l++) {
        if (this->h_layers[l]->enable_learning && this->h_layers[l]->learning_type == 0)
            printf("Warning Layer %i: Learning enabled but rule not selected.\n\n", l);
        else if (this->h_layers[l]->enable_learning && this->h_layers[l]->learning_type == 1)
            printf("Paredes' STDP Enabled\n");
        else if (this->h_layers[l]->enable_learning && this->h_layers[l]->learning_type == 2)
            printf("Shrestha's STDP Enabled\n");
        else if (this->h_layers[l]->enable_learning && this->h_layers[l]->learning_type == 3)
            printf("Gerstner's STDP Enabled\n");
        else if (this->h_layers[l]->enable_learning && this->h_layers[l]->learning_type == 4)
            printf("Kheradpisheh's STDP Enabled\n");
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMULATION FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////////////////


// update input spike trains
void Network::update_input() {
    update_input_trains<<<this->block_0, 1>>>(this->d_inputs, this->d_inp_size, this->d_length_delay_inp);
    cudaMemcpy(this->h_inputs, this->d_inputs,
               sizeof(int) * this->h_inp_size[0] * this->h_inp_size[1] * this->h_inp_size[2] *
               this->h_length_delay_inp[0], cudaMemcpyDeviceToHost);
}


// update network's state
void Network::feed(bool& break_fun) {

    // copy inputs to device
    cudaMemcpy(this->d_inputs, this->h_inputs,
               sizeof(int) * this->h_inp_size[0] * this->h_inp_size[1] * this->h_inp_size[2] *
               this->h_length_delay_inp[0], cudaMemcpyHostToDevice);

    // enable learning
    if (this->learning)
        enable_learning<<<this->block_1, 1>>>(this->d_d_layers);

    // update spike trains for input channels
    update_input_channels<<<this->block_2, this->thread_1>>>(this->d_d_layers, this->d_sim_step, this->d_inputs);

    // spike propagation
    propagation<<<this->block_3, this->thread_0>>>(this->d_d_layers, this->d_inputs);
    add_input<<<this->block_3, this->thread_0>>>(this->d_d_layers);
    update_V<<<this->block_3, this->thread_0>>>(this->d_d_layers, this->d_sim_step, this->d_node_refrac);

    // recursive spatial perpendicular inhibition (for learning)
    bool inhibition = this->inhibition;
    while (inhibition && this->learning) {
        spatial_firing_node_kernel_channel<<<this->block_5, this->thread_0>>>(this->d_d_layers);
        spatial_firing_node_kernel<<<this->block_1, this->thread_0>>>(this->d_d_layers);
        spatial_firing_node<<<this->block_1, 1>>>(this->d_d_layers);
        spatial_perpendicular_inhibition<<<this->block_5, this->thread_0>>>(this->d_d_layers);

        inhibition = false;
        for (int l = 0; l < this->cnt_layers; l++) {
            cudaMemcpy(this->h_layers[l], this->h_d_layers[l], sizeof(Layer), cudaMemcpyDeviceToHost);
            if (this->h_layers[l]->kernel_max != -1) inhibition = true;
        }
    }

    // perpendicular inhibition
    inhibition = this->inhibition;
    if (inhibition) {
        firing_node_kernel<<<this->block_4, this->thread_0>>>(this->d_d_layers);
        firing_node<<<this->block_4, 1>>>(this->d_d_layers);
        perpendicular_inhibition<<<this->block_3, this->thread_0>>>(this->d_d_layers);
    }

    // learning rules
    if (this->learning && this->learning_type == 1) { // Paredes' STDP
        stdp_paredes_kernel_channel<<<this->block_6, this->thread_0>>>(this->d_d_layers);
        learning_update_weights<<<this->block_6, this->thread_0>>>(this->d_d_layers);
        stdp_paredes_track_convergence_channel<<<this->block_5, this->thread_0>>>(this->d_d_layers);
        stdp_paredes_track_convergence<<<this->block_1, this->thread_0>>>(this->d_d_layers);
    } else if (this->learning && this->learning_type == 2) { // Shrestha's STDP
        stdp_shrestha_kernel_channel<<<this->block_6, this->thread_0>>>(this->d_d_layers);
        learning_update_weights<<<this->block_6, this->thread_0>>>(this->d_d_layers);
    } else if (this->learning && this->learning_type == 3) { // Gerstner's STDP
        stdp_gerstner_kernel_channel<<<this->block_6, this->thread_0>>>(this->d_d_layers);
        learning_update_weights<<<this->block_6, this->thread_0>>>(this->d_d_layers);
    } else if (this->learning && this->learning_type == 4) { // Kheradpisheh's STDP
        stdp_kheradpisheh_kernel_channel<<<this->block_6, this->thread_0>>>(this->d_d_layers);
        learning_update_weights<<<this->block_6, this->thread_0>>>(this->d_d_layers);
    }

    // drop delays
    if (this->learning && this->drop_delays)
        drop_delays_kernel<<<this->block_1, this->thread_0>>>(this->d_d_layers, this->d_drop_delays_th);

    // update outputs
    update_output_channels<<<this->block_3, this->thread_0>>>(this->d_d_layers);
    update_output<<<this->block_4, this->thread_0>>>(this->d_d_layers, this->d_sim_step);

    // limit learning updates
    if (this->learning) {
        learning_limit_updates<<<this->block_1, 1>>>(this->d_d_layers);
        for (int l = 0; l < this->cnt_layers; l++) {
            cudaMemcpy(this->h_layers[l], this->h_d_layers[l], sizeof(Layer), cudaMemcpyDeviceToHost);
            if (this->h_layers[l]->limit_learning) break_fun = true;
        }
    }
}


// copy network's state from device to host memory
void Network::copy_to_host(){

    for (int l = 0; l < this->cnt_layers; l++) {
        cudaMemcpy(this->h_layers[l], this->h_d_layers[l], sizeof(Layer), cudaMemcpyDeviceToHost);

        // layer data
        cudaMemcpy(this->h_layers[l]->h_kernels_cnvg, this->h_layers[l]->d_kernels_cnvg,
                   sizeof(bool) * this->h_layers[l]->cnt_kernels, cudaMemcpyDeviceToHost);

        // kernel data
        for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++) {
            cudaMemcpy(this->h_layers[l]->h_kernels[k], this->h_layers[l]->h_d_kernels[k],
                       sizeof(Kernel), cudaMemcpyDeviceToHost);

            cudaMemcpy(this->h_layers[l]->h_kernels[k]->h_node_train, this->h_layers[l]->h_kernels[k]->d_node_train,
                       sizeof(int) * this->h_layers[l]->out_node_kernel * this->h_layers[l]->length_delay_out,
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->h_node_posttrace,
                       this->h_layers[l]->h_kernels[k]->d_node_posttrace,
                       sizeof(float) * this->h_layers[l]->out_node_kernel, cudaMemcpyDeviceToHost);
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->h_weights_exc, this->h_layers[l]->h_kernels[k]->d_weights_exc,
                       sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->rf_side *
                       this->h_layers[l]->rf_side * this->h_layers[l]->num_delays, cudaMemcpyDeviceToHost);
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->h_weights_inh, this->h_layers[l]->h_kernels[k]->d_weights_inh,
                       sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->rf_side *
                       this->h_layers[l]->rf_side * this->h_layers[l]->num_delays, cudaMemcpyDeviceToHost);
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->h_delay_active, this->h_layers[l]->h_kernels[k]->d_delay_active,
                       sizeof(bool) * this->h_layers[l]->num_delays, cudaMemcpyDeviceToHost);
        }
    }
}


// init the network
void Network::init(){

    // input spike trains
    for (int ch = 0; ch < this->h_inp_size[0]; ch++) {
        for (int i = 0; i < this->h_inp_size[1] * this->h_inp_size[2]; i++) {
            for (int d = 0; d < this->h_length_delay_inp[0]; d++) {
                int idx = ch * this->h_inp_size[1] * this->h_inp_size[2] * this->h_length_delay_inp[0] +
                        i * this->h_length_delay_inp[0] + d;
                this->h_inputs[idx] = 0;
            }
        }
    }
    cudaMemcpy(this->d_inputs, this->h_inputs,
               sizeof(int) * this->h_inp_size[0] * this->h_inp_size[1] * this->h_inp_size[2] *
               this->h_length_delay_inp[0], cudaMemcpyHostToDevice);

    // layer data
    for (int l = 0; l < this->cnt_layers; l++) {

        this->h_layers[l]->active = false;
        this->h_layers[l]->firing_node = false;
        if (this->h_layers[l]->enable_learning) {
            this->h_layers[l]->learning = false;
            this->h_layers[l]->limit_learning = false;
            this->h_layers[l]->learning_updates_cnt = 0;
            this->h_layers[l]->enable_learning_cnt = 0;
        }

        if (this->h_layers[l]->learning_type == 2 ||
            this->h_layers[l]->learning_type == 3 ||
            this->h_layers[l]->learning_type == 4) {
            for (int i = 0; i < this->h_layers[l]->inp_synapses_total; i++)
                this->h_layers[l]->h_stdp_precnt[i] = -1;
            cudaMemcpy(this->h_layers[l]->d_stdp_precnt, this->h_layers[l]->h_stdp_precnt,
                       sizeof(int) * this->h_layers[l]->inp_synapses_total, cudaMemcpyHostToDevice);
        }

        for (int i = 0; i < this->h_layers[l]->inp_synapses_total; i++)
            this->h_layers[l]->h_synapse_pretrace[i] = this->h_synapse_trace_init[0];
        cudaMemcpy(this->h_layers[l]->d_synapse_pretrace, this->h_layers[l]->h_synapse_pretrace,
                   sizeof(float) * this->h_layers[l]->inp_synapses_total, cudaMemcpyHostToDevice);

        // kernel data
        for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++) {

            for (int i = 0; i < h_layers[l]->out_node_kernel; i++) {
                this->h_layers[l]->h_kernels[k]->h_node_posttrace[i] = 0.f;
                for (int d = 0; d < h_layers[l]->length_delay_out; d++)
                    this->h_layers[l]->h_kernels[k]->h_node_train[i * h_layers[l]->length_delay_out + d] = 0;
            }
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_node_train, this->h_layers[l]->h_kernels[k]->h_node_train,
                       sizeof(int) * this->h_layers[l]->out_node_kernel * this->h_layers[l]->length_delay_out,
                       cudaMemcpyHostToDevice);
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_node_posttrace,
                       this->h_layers[l]->h_kernels[k]->h_node_posttrace,
                       sizeof(float) * this->h_layers[l]->out_node_kernel, cudaMemcpyHostToDevice);

            for (int i = 0; i < this->h_layers[l]->out_nodesep_kernel; i++) {
                this->h_layers[l]->h_kernels[k]->h_nodesep_V[i] = 0.f;
                this->h_layers[l]->h_kernels[k]->h_nodesep_train[i] = 0;
                this->h_layers[l]->h_kernels[k]->h_nodesep_refrac[i] = this->h_node_refrac[0] / this->h_sim_step[0];
                if (this->h_layers[l]->threshold_diehl)
                    this->h_layers[l]->h_kernels[k]->h_threshold_diehl_nodesep_theta[i] = 0.f;
            }
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_nodesep_train,
                       this->h_layers[l]->h_kernels[k]->h_nodesep_train,
                       sizeof(int) * this->h_layers[l]->out_nodesep_kernel, cudaMemcpyHostToDevice);
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_nodesep_V, this->h_layers[l]->h_kernels[k]->h_nodesep_V,
                       sizeof(float) * this->h_layers[l]->out_nodesep_kernel, cudaMemcpyHostToDevice);
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_nodesep_refrac,
                       this->h_layers[l]->h_kernels[k]->h_nodesep_refrac,
                       sizeof(float) * this->h_layers[l]->out_nodesep_kernel, cudaMemcpyHostToDevice);
            if (this->h_layers[l]->threshold_diehl)
                cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_threshold_diehl_nodesep_theta,
                           this->h_layers[l]->h_kernels[k]->h_threshold_diehl_nodesep_theta,
                           sizeof(float) * this->h_layers[l]->out_nodesep_kernel, cudaMemcpyHostToDevice);

            if (this->h_layers[l]->learning_type == 3 ||
                this->h_layers[l]->learning_type == 4) {
                for (int i = 0; i < this->h_layers[l]->out_nodesep_kernel; i++)
                    this->h_layers[l]->h_kernels[k]->h_stdp_postcnt[i] = -1;
                cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_stdp_postcnt,
                           this->h_layers[l]->h_kernels[k]->h_stdp_postcnt,
                           sizeof(int) * this->h_layers[l]->out_nodesep_kernel, cudaMemcpyHostToDevice);
            }

            cudaMemcpy(this->h_layers[l]->h_d_kernels[k], this->h_layers[l]->h_kernels[k],
                       sizeof(Kernel), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(this->h_layers[l]->d_d_kernels, this->h_layers[l]->h_d_kernels,
                   sizeof(Kernel*) * this->h_layers[l]->cnt_kernels, cudaMemcpyHostToDevice);
        cudaMemcpy(this->h_d_layers[l], this->h_layers[l], sizeof(Layer), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(this->d_d_layers, this->h_d_layers, sizeof(Layer*) * this->cnt_layers, cudaMemcpyHostToDevice);
}


// weights to device
void Network::weights_to_device() {

    for (int l = 0; l < this->cnt_layers; l++) {
        for (int k = 0; k < this->h_layers[l]->cnt_kernels; k++) {
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_weights_exc, this->h_layers[l]->h_kernels[k]->h_weights_exc,
                       sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->rf_side *
                       this->h_layers[l]->rf_side * this->h_layers[l]->num_delays, cudaMemcpyHostToDevice);
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_weights_inh, this->h_layers[l]->h_kernels[k]->h_weights_inh,
                       sizeof(float) * this->h_layers[l]->kernel_channels * this->h_layers[l]->rf_side *
                       this->h_layers[l]->rf_side * this->h_layers[l]->num_delays, cudaMemcpyHostToDevice);
            cudaMemcpy(this->h_layers[l]->h_kernels[k]->d_delay_active, this->h_layers[l]->h_kernels[k]->h_delay_active,
                       sizeof(bool) * this->h_layers[l]->num_delays, cudaMemcpyHostToDevice);
            cudaMemcpy(this->h_layers[l]->h_d_kernels[k], this->h_layers[l]->h_kernels[k],
                       sizeof(Kernel), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(this->h_layers[l]->d_d_kernels, this->h_layers[l]->h_d_kernels,
                   sizeof(Kernel*) * this->h_layers[l]->cnt_kernels, cudaMemcpyHostToDevice);
        cudaMemcpy(this->h_layers[l]->d_kernels_cnvg, this->h_layers[l]->h_kernels_cnvg,
                   sizeof(bool) * this->h_layers[l]->cnt_kernels, cudaMemcpyHostToDevice);
        cudaMemcpy(this->h_d_layers[l], this->h_layers[l], sizeof(Layer), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(this->d_d_layers, this->h_d_layers, sizeof(Layer*) * this->cnt_layers, cudaMemcpyHostToDevice);
}