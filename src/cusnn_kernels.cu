#include "cusnn.cuh"


// update input spike trains
__global__ void update_input_trains(int *inputs, int *input_size, int *delay) {

    int channel = blockIdx.x;
    int idx_y = blockIdx.y;
    int idx_x = blockIdx.z;

    int idx_node = idx_x * input_size[1] + idx_y;
    int begin_vector = channel * input_size[1] * input_size[2] * delay[0] + idx_node * delay[0];
    int end_vector = channel * input_size[1] * input_size[2] * delay[0] + (idx_node + 1) * delay[0];

    for (int i = end_vector - 1; i > begin_vector; i--)
        inputs[i] = inputs[i-1];
    inputs[begin_vector] = 0;
}


// enable STDP when the simulation time is larger than the maximum input delay
__global__ void enable_learning(Layer **layers) {

    int layer = blockIdx.x;

    if (layers[layer]->enable_learning &&
        layers[layer]->learning_type &&
        !layers[layer]->learning &&
        !layers[layer]->limit_learning) {

        if (layers[layer]->enable_learning_cnt > layers[layer]->learning_warmup_time)
            layers[layer]->learning = true;
        else layers[layer]->enable_learning_cnt++;
    }
}


// update spike trains for input channels
__global__ void update_input_channels(Layer **layers, float *sim_step, int *inputs) {

    int layer = blockIdx.x;
    int node = blockIdx.y;
    int channel = blockIdx.z;
    int delay = threadIdx.x;

    if (layers[layer]->inp_size[0] > channel &&
        layers[layer]->inp_node_kernel > node &&
        layers[layer]->num_delays > delay) {

        // global memory optimization
        int learning_type = layers[layer]->learning_type;
        int inp_size[] = {layers[layer]->inp_size[0], layers[layer]->inp_size[1], layers[layer]->inp_size[2]};
        int padding[] = {layers[layer]->padding[0], layers[layer]->padding[1]};
        int padding_total = layers[layer]->padding_total[0];
        int num_delays = layers[layer]->num_delays;
        int length_delay_inp = layers[layer]->length_delay_inp;
        int delay_indices = layers[layer]->d_delay_indices[delay];

        int idx_xpad = node / (inp_size[1] + padding_total);
        int idx_ypad = node % (inp_size[1] + padding_total);

        int idx_syn_inp = channel * layers[layer]->inp_node_kernel * num_delays + node * num_delays + delay;
        float synapse_pretrace = layers[layer]->d_synapse_pretrace[idx_syn_inp];

        if (idx_ypad < padding[0] ||
            idx_ypad >= inp_size[1] + padding[0] ||
            idx_xpad < padding[1] ||
            idx_xpad >= inp_size[2] + padding[1]) {

            // pre-synaptic trace for zero-padding nodes
            if (layers[layer]->learning && layers[layer]->homeostasis) synapse_pretrace = 1000.f;
            else synapse_pretrace = 0.f;

        } else {

            float value;
            int idx_y = idx_ypad - padding[0];
            int idx_x = idx_xpad - padding[1];
            int idx_node = idx_x * inp_size[1] + idx_y;

            // spikes received by the nodes after delay
            if (!layer) {
                int delay_index = channel * inp_size[1] * inp_size[2] * length_delay_inp + idx_node * length_delay_inp + delay_indices;
                value = (float) inputs[delay_index];
            } else {
                int delay_index = idx_node * length_delay_inp + delay_indices;
                value = (float) layers[layer-1]->d_d_kernels[channel]->d_node_train[delay_index];
            }

            // update pre-synaptic traces
            synapse_pretrace += (sim_step[0] / layers[layer]->decay) * (-synapse_pretrace + layers[layer]->alpha * value);
            if (synapse_pretrace < 0.f)
                synapse_pretrace = 0.f;

            // update counters for Shrestha's, Gerstner's, and Kheradpisheh's STDP
            if (learning_type == 2 || learning_type == 3 || learning_type == 4) {
                int stdp_precnt = layers[layer]->d_stdp_precnt[idx_syn_inp];
                if (value > 0.f) stdp_precnt = 0;
                else if (stdp_precnt >= 0) stdp_precnt++;
                layers[layer]->d_stdp_precnt[idx_syn_inp] = stdp_precnt;
            }
        }
        layers[layer]->d_synapse_pretrace[idx_syn_inp] = synapse_pretrace;
    }
}


// spike propagation
__global__ void propagation(Layer **layers, int *inputs) {

    int layer = blockIdx.x;
    int node = blockIdx.y;
    int channel = blockIdx.z;
    int kernel = threadIdx.x;

    if (layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->out_node_kernel > node) {

        // global memory that we read only once
        int rf_side = layers[layer]->rf_side;
        int strides = layers[layer]->strides;
        int num_delays = layers[layer]->num_delays;
        int length_delay_inp = layers[layer]->length_delay_inp;
        int padding[] = {layers[layer]->padding[0], layers[layer]->padding[1]};
        int inp_size[] = {layers[layer]->inp_size[0], layers[layer]->inp_size[1], layers[layer]->inp_size[2]};
        int rf_side_limits[] = {layers[layer]->rf_side_limits[0], layers[layer]->rf_side_limits[1]};
        int out_size = layers[layer]->out_size[1];

        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
        int num_delays_active = kernel_local->num_delays_active;
        float *weights_total = kernel_local->d_weights_total;
        int *delay_indices = layers[layer]->d_delay_indices;
        int *node_train;
        if (layer > 0) node_train = layers[layer-1]->d_d_kernels[channel]->d_node_train;

        int channel_inp = 0;
        if (layers[layer]->out_maps == 1) channel_inp = channel;

        int idx_x_rf = node / out_size;
        int idx_y_rf = node % out_size;

        // prevent kernels that didn't converge to accumulate spikes
        float nodesep_channel_input = 0.f;
        if (layers[layer]->learning || (!layers[layer]->enable_learning && layers[layer]->d_kernels_cnvg[kernel])) {

            layers[layer]->active = true;
            for (int d = 0; d < num_delays_active; d++) {
                for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {

                        int idx_xpad = idx_x_rf * strides + cols;
                        int idx_ypad = idx_y_rf * strides + rows;

                        if (idx_ypad >= padding[0] &&
                            idx_ypad < inp_size[1] + padding[0] &&
                            idx_xpad >= padding[1] &&
                            idx_xpad < inp_size[2] + padding[1]) {

                            float value;
                            int idx_y = idx_ypad - padding[0];
                            int idx_x = idx_xpad - padding[1];
                            int idx_node = idx_x * inp_size[1] + idx_y;

                            // spikes received after synaptic delay
                            if (!layer) {
                                int delay_index = channel * inp_size[1] * inp_size[2] * length_delay_inp +
                                        idx_node * length_delay_inp + delay_indices[d];
                                value = (float) inputs[delay_index];
                            } else {
                                int delay_index = idx_node * length_delay_inp + delay_indices[d];
                                value = (float) node_train[delay_index];
                            }

                            // propagate input spikes
                            int idx_syn = cols * rf_side + rows;
                            int idx_syn_weights = channel_inp * rf_side * rf_side * num_delays + idx_syn * num_delays + d;
                            nodesep_channel_input += value * weights_total[idx_syn_weights];
                        }
                    }
                }
            }
        }

        // write to global memory
        int idx_nodesep_channel = channel * layers[layer]->out_node_kernel + node;
        kernel_local->d_nodesep_channel_input[idx_nodesep_channel] = nodesep_channel_input;
    }
}


// add inputs
__global__ void add_input(Layer **layers) {

    int layer = blockIdx.x;
    int node = blockIdx.y;
    int channel = blockIdx.z;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->inp_size[0] > channel &&
        layers[layer]->out_node_kernel > node &&
        layers[layer]->cnt_kernels > kernel &&
        ((!channel && layers[layer]->out_maps == 1) || layers[layer]->kernel_channels == 1)) {

        // global memory that we read only once
        int rf_side = layers[layer]->rf_side;
        int strides = layers[layer]->strides;
        int num_delays = layers[layer]->num_delays;
        int kernel_channels = layers[layer]->kernel_channels;
        int inp_node_kernel = layers[layer]->inp_node_kernel;
        int out_node_kernel = layers[layer]->out_node_kernel;
        int padding[] = {layers[layer]->padding[0], layers[layer]->padding[1]};
        int inp_size[] = {layers[layer]->inp_size[0], layers[layer]->inp_size[1], layers[layer]->inp_size[2]};
        int rf_side_limits[] = {layers[layer]->rf_side_limits[0], layers[layer]->rf_side_limits[1]};
        int padding_total = layers[layer]->padding_total[0];
        int out_size = layers[layer]->out_size[1];
        float *synapse_pretrace = layers[layer]->d_synapse_pretrace;

        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
        int num_delays_active = kernel_local->num_delays_active;
        float *nodesep_channel_input = kernel_local->d_nodesep_channel_input;

        // add inputs
        float nodesep_input = 0.f;
        for (int ch = 0; ch < kernel_channels; ch++) {
            int idx_nodesep_aux = (ch + channel) * out_node_kernel + node;
            nodesep_input += nodesep_channel_input[idx_nodesep_aux];
        }

        // pretrace data
        float nodesep_pretrace = 0.f;
        float nodesep_maxpretrace = 0.f;
        int idx_x_rf = node / out_size;
        int idx_y_rf = node % out_size;
        for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
            for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {

                int idx_xpad = idx_x_rf * strides + cols;
                int idx_ypad = idx_y_rf * strides + rows;

                if (idx_ypad >= 0 &&
                    idx_ypad < inp_size[1] + padding[0] &&
                    idx_xpad >= 0 &&
                    idx_xpad < inp_size[2] + padding[1]) {

                    int idx_nodepad = idx_xpad * (inp_size[1] + padding_total) + idx_ypad;
                    for (int ch = 0; ch < kernel_channels; ch++) {
                        for (int d = 0; d < num_delays_active; d++) {
                            int idx_syn_inp = (ch + channel) * inp_node_kernel * num_delays + idx_nodepad * num_delays + d;

                            // node cumulative pretrace
                            nodesep_pretrace += synapse_pretrace[idx_syn_inp];

                            // max pretrace of receptive field
                            if (synapse_pretrace[idx_syn_inp] > nodesep_maxpretrace)
                                nodesep_maxpretrace = synapse_pretrace[idx_syn_inp];
                        }
                    }
                }
            }
        }

        // write to global memory
        int idx_nodesep = channel * out_node_kernel + node;
        kernel_local->learning_trigger = false;
        kernel_local->d_nodesep_input[idx_nodesep] = nodesep_input;
        kernel_local->d_nodesep_pretrace[idx_nodesep] = nodesep_pretrace;
        kernel_local->d_nodesep_maxpretrace[idx_nodesep] = nodesep_maxpretrace;
    }
}


// update node dynamics
__global__ void update_V(Layer **layers, float *sim_step, float *refrac) {

    int layer = blockIdx.x;
    int node = blockIdx.y;
    int channel = blockIdx.z;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->out_node_kernel > node &&
        ((!channel && layers[layer]->out_maps == 1) || layers[layer]->kernel_channels == 1)) {

        // global memory that we read only once
        int out_node_kernel = layers[layer]->out_node_kernel;
        int out_size = layers[layer]->out_size[1];
        float decay = layers[layer]->decay;
        int learning_type = layers[layer]->learning_type;
        float node_Vth = layers[layer]->node_Vth;

        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
        float *nodesep_refrac = kernel_local->d_nodesep_refrac;
        float *nodesep_V = kernel_local->d_nodesep_V;
        int *nodesep_train = kernel_local->d_nodesep_train;
        int *nodesep_perpendicular = kernel_local->d_nodesep_perpendicular;
        float *nodesep_pretrace;
        if (layers[layer]->homeostasis) nodesep_pretrace = kernel_local->d_nodesep_pretrace;
        int *stdp_postcnt;
        if (learning_type == 3 || learning_type == 4) stdp_postcnt = kernel_local->d_stdp_postcnt;

        int idx_x_rf = node / out_size;
        int idx_y_rf = node % out_size;
        int idx_nodesep = channel * out_node_kernel + node;

        // homeostasis mechanism
        float max_trace = 0.f;
        if (layers[layer]->homeostasis) {
            for (int rows = -1; rows <= 1; rows++) {
                for (int cols = -1; cols <= 1; cols++) {
                    int idx_node = (idx_x_rf + cols) * out_size + idx_y_rf + rows;
                    int idx_nodesep_aux = channel * out_node_kernel + idx_node;
                    if (idx_node >= 0 && idx_node < out_node_kernel) {
                        if (max_trace < nodesep_pretrace[idx_nodesep_aux])
                            max_trace = nodesep_pretrace[idx_nodesep_aux];
                    }
                }
            }
        }

        // update refractory counter
        nodesep_refrac[idx_nodesep]++;

        // update V if node is not in refractory period
        if (nodesep_refrac[idx_nodesep] * sim_step[0] >= refrac[0])
            nodesep_V[idx_nodesep] += (sim_step[0]/decay) * (-nodesep_V[idx_nodesep] - max_trace +
                    kernel_local->d_nodesep_input[idx_nodesep]);
        if (nodesep_V[idx_nodesep] < 0.f) nodesep_V[idx_nodesep] = 0.f;

        // spike generation
        nodesep_perpendicular[idx_nodesep] = 0;
        if (nodesep_V[idx_nodesep] >= node_Vth) {
            nodesep_train[idx_nodesep] = 1;
            layers[layer]->firing_node = true;
            nodesep_perpendicular[idx_nodesep] = 1;
            if (layers[layer]->learning && !layers[layer]->inhibition)
                kernel_local->learning_trigger = true;
        }

        // update counters for Gerstner's and Kheradpisheh's STDP
        if (learning_type == 3 || learning_type == 4) {
            if (nodesep_train[idx_nodesep]) stdp_postcnt[idx_nodesep] = 0;
            else if (stdp_postcnt[idx_nodesep] >= 0) stdp_postcnt[idx_nodesep]++;
        }
    }
}


// STDP select firing node in each channel of each kernel
__global__ void spatial_firing_node_kernel_channel(Layer **layers) {

    int layer = blockIdx.x;
    int channel = blockIdx.y;
    int kernel = threadIdx.x;

    if (layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        ((!channel && layers[layer]->out_maps == 1) || layers[layer]->kernel_channels == 1)) {

        // global memory optimization
        int node_max = -1;
        float V_max = 0.f;
        int out_node_kernel = layers[layer]->out_node_kernel;
        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];

        if (layers[layer]->active &&
            layers[layer]->firing_node &&
            layers[layer]->inhibition &&
            layers[layer]->learning &&
            layers[layer]->learning_type &&
            layers[layer]->inhibition_spatial) {

            int *nodesep_perpendicular = kernel_local->d_nodesep_perpendicular;
            float *nodesep_V = kernel_local->d_nodesep_V;

            for (int node = 0; node < out_node_kernel; node++) {
                int idx_nodesep = channel * out_node_kernel + node;
                if (nodesep_perpendicular[idx_nodesep] && nodesep_V[idx_nodesep] > V_max) {
                    node_max = node;
                    V_max = nodesep_V[idx_nodesep];
                }
            }
        }
        kernel_local->d_V_max[channel] = V_max;
        kernel_local->d_node_max[channel] = node_max;
    }
}


// STDP select firing node in each kernel
__global__ void spatial_firing_node_kernel(Layer **layers) {

    int layer = blockIdx.x;
    int kernel = threadIdx.x;

    if (layers[layer]->cnt_kernels > kernel) {

        // global memory optimization
        float V_max = 0.f;
        int node_max = -1;
        int channel_max = -1;
        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];

        if (layers[layer]->active &&
            layers[layer]->firing_node &&
            layers[layer]->inhibition &&
            layers[layer]->learning &&
            layers[layer]->learning_type &&
            layers[layer]->inhibition_spatial) {

            float *V_max_ch = kernel_local->d_V_max;
            int *node_max_ch = kernel_local->d_node_max;

            for (int ch = 0; ch < layers[layer]->out_maps; ch++) {
                if (V_max_ch[ch] > V_max) {
                    V_max = V_max_ch[ch];
                    node_max = node_max_ch[ch];
                    channel_max = ch;
                }
            }
        }
        kernel_local->V_max = V_max;
        kernel_local->node_max = node_max;
        kernel_local->channel_max = channel_max;
    }
}


// STDP select firing node
__global__ void spatial_firing_node(Layer **layers) {

    int layer = blockIdx.x;

    int kernel_max = -1;
    if (layers[layer]->active &&
        layers[layer]->firing_node &&
        layers[layer]->inhibition &&
        layers[layer]->learning &&
        layers[layer]->learning_type &&
        layers[layer]->inhibition_spatial) {

        float V_max = 0.f;
        for (int k = 0; k < layers[layer]->cnt_kernels; k++) {
            if (layers[layer]->d_d_kernels[k]->V_max > V_max) {
                V_max = layers[layer]->d_d_kernels[k]->V_max;
                kernel_max = k;
            }
        }

        if (kernel_max != -1) {
            int node = layers[layer]->d_d_kernels[kernel_max]->node_max;
            int channel = layers[layer]->d_d_kernels[kernel_max]->channel_max;
            int idx_nodesep = channel * layers[layer]->out_node_kernel + node;
            layers[layer]->d_d_kernels[kernel_max]->d_nodesep_perpendicular[idx_nodesep] = 0;
            layers[layer]->d_d_kernels[kernel_max]->learning_trigger = true;
        }
    }
    layers[layer]->kernel_max = kernel_max;
}


// STDP perpendicular inhibition
__global__ void spatial_perpendicular_inhibition(Layer **layers) {

    int layer = blockIdx.x;
    int channel = blockIdx.y;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->learning &&
        layers[layer]->learning_type &&
        layers[layer]->inhibition &&
        layers[layer]->inhibition_spatial &&
        layers[layer]->firing_node &&
        layers[layer]->kernel_max != -1 &&
        layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        ((!channel && layers[layer]->out_maps == 1) || layers[layer]->kernel_channels == 1)) {

        // global memory optimization
        int kernel_max = layers[layer]->kernel_max;
        int channel_max = layers[layer]->d_d_kernels[kernel_max]->channel_max;
        int node_max = layers[layer]->d_d_kernels[kernel_max]->node_max;
        int neigh_inh = layers[layer]->neigh_inh;
        int out_size = layers[layer]->out_size[1];
        int out_node_kernel = layers[layer]->out_node_kernel;
        int learning_type = layers[layer]->learning_type;

        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
        int *nodesep_train = kernel_local->d_nodesep_train;
        float *nodesep_V = kernel_local->d_nodesep_V;
        float *nodesep_refrac = kernel_local->d_nodesep_refrac;

        int *stdp_postcnt;
        if (learning_type == 3 || learning_type == 4)
            stdp_postcnt = kernel_local->d_stdp_postcnt;

        int idx_x_rf = node_max / out_size;
        int idx_y_rf = node_max % out_size;

        for (int rows = -neigh_inh; rows <= neigh_inh; rows++) {
            for (int cols = -neigh_inh; cols <= neigh_inh; cols++) {

                int idx_x_rf2 = idx_x_rf + cols;
                int idx_y_rf2 = idx_y_rf + rows;
                int idx_node = idx_x_rf2 * out_size + idx_y_rf2;

                if (idx_node >= 0 && idx_node < out_node_kernel) {
                    if (kernel != kernel_max ||
                        (kernel == kernel_max && channel != channel_max) ||
                        (kernel == kernel_max && channel == channel_max && idx_node != node_max)) {

                        int idx_nodesep = channel * out_node_kernel + idx_node;
                        nodesep_train[idx_nodesep] = 0;
                        nodesep_V[idx_nodesep] = 0.f;
                        nodesep_refrac[idx_nodesep] = 0.f;

                        // update counters for Gerstner's and Kheradpisheh's STDP
                        if (learning_type == 3 || learning_type == 4)
                            stdp_postcnt[idx_nodesep] = -1;
                    }
                }
            }
        }
    }
}


// Paredes' STDP
__global__ void stdp_paredes_kernel_channel(Layer **layers) {

    int layer = blockIdx.x;
    int channel = blockIdx.y;
    int delay = blockIdx.z;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->num_delays > delay &&
        layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->learning &&
        layers[layer]->learning_type == 1 &&
        !layers[layer]->d_kernels_cnvg[kernel]) {

        int kernel_channels = layers[layer]->kernel_channels;
        int channel_inp = 0;
        int channel_out = 0;
        if (layers[layer]->out_maps == 1) channel_inp = channel;
        else if (kernel_channels == 1) channel_out = channel;

        bool weights_delta_maps_channels = false;
        if (layers[layer]->d_d_kernels[kernel]->learning_trigger &&
            layers[layer]->d_d_kernels[kernel]->d_delay_active[delay]) {

            // global memory optimization
            int rf_side = layers[layer]->rf_side;
            int num_delays = layers[layer]->num_delays;
            int inp_node_kernel = layers[layer]->inp_node_kernel;
            int strides = layers[layer]->strides;
            int out_size = layers[layer]->out_size[1];
            int inp_size = layers[layer]->inp_size[1];
            int rf_side_limits[] = {layers[layer]->rf_side_limits[0], layers[layer]->rf_side_limits[1]};
            float synapse_inh_scaling = layers[layer]->synapse_inh_scaling;
            int padding_total = layers[layer]->padding_total[0];
            int out_node_kernel = layers[layer]->out_node_kernel;
            float synapse_w_init = layers[layer]->synapse_w_init;
            float stdp_paredes_a = layers[layer]->stdp_paredes_a;
            float learning_rate = layers[layer]->learning_rate;
            float *synapse_pretrace = layers[layer]->d_synapse_pretrace;

            Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
            int *nodesep_train = kernel_local->d_nodesep_train;
            float *nodesep_maxpretrace = kernel_local->d_nodesep_maxpretrace;
            float *weights_exc_delta = kernel_local->d_weights_exc_delta;
            float *weights_exc = kernel_local->d_weights_exc;

            float *weights_inh_delta, *weights_inh;
            if (synapse_inh_scaling > 0.f) {
                weights_inh_delta = kernel_local->d_weights_inh_delta;
                weights_inh = kernel_local->d_weights_inh;
            }

            for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                    int idx_syn = cols * rf_side + rows;
                    int idx_syn_delta = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                    weights_exc_delta[idx_syn_delta] = 0.f;
                    if (synapse_inh_scaling > 0.f)
                        weights_inh_delta[idx_syn_delta] = 0.f;
                }
            }

            int cnt_nodes = 0;
            for (int node = 0; node < out_node_kernel; node++) {
                int idx_nodesep = channel_out * out_node_kernel + node;
                if (nodesep_train[idx_nodesep]) {

                    cnt_nodes++;
                    weights_delta_maps_channels = true;

                    int idx_x_rf = node / out_size;
                    int idx_y_rf = node % out_size;

                    for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                        for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {

                            int idx_xpad = idx_x_rf * strides + cols;
                            int idx_ypad = idx_y_rf * strides + rows;
                            int idx_nodepad = idx_xpad * (inp_size + padding_total) + idx_ypad;

                            int idx_syn = cols * rf_side + rows;
                            int idx_syn_weights = channel_inp * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                            int idx_syn_delta = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                            int idx_syn_inp = channel * inp_node_kernel * num_delays + idx_nodepad * num_delays + delay;

                            /* Long-Term Potentiation (LTP) and Long-Term Depression (LTD) */
                            // excitatory synapses
                            float diff_weights = weights_exc[idx_syn_weights] - synapse_w_init;
                            float trace_norm = synapse_pretrace[idx_syn_inp] / nodesep_maxpretrace[idx_nodesep];
                            float LTP = exp(-diff_weights) * (exp(trace_norm) - stdp_paredes_a);
                            float LTD = exp(diff_weights) * (exp(1.f - trace_norm) - stdp_paredes_a);
                            weights_exc_delta[idx_syn_delta] += learning_rate * (LTP - LTD);

                            // inhibitory synapses
                            if (synapse_inh_scaling > 0.f) {
                                diff_weights = weights_inh[idx_syn_weights] + synapse_w_init;
                                LTP = exp(-diff_weights) * (exp(trace_norm) - stdp_paredes_a);
                                LTD = exp(diff_weights) * (exp(1.f - trace_norm) - stdp_paredes_a);
                                weights_inh_delta[idx_syn_delta] += learning_rate * (LTP - LTD);
                            }
                        }
                    }
                }
            }

            // average weight update
            if (cnt_nodes > 0) {
                for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                        int idx_syn = cols * rf_side + rows;
                        int idx_syn_delta = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                        weights_exc_delta[idx_syn_delta] /= (float) cnt_nodes;
                        if (synapse_inh_scaling > 0.f)
                            weights_inh_delta[idx_syn_delta] /= (float) cnt_nodes;
                    }
                }
            }
        }
        int idx_channel = channel_out * kernel_channels + channel_inp;
        layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = weights_delta_maps_channels;
    }
}


// update synaptic weights
__global__ void learning_update_weights(Layer **layers) {

    int layer = blockIdx.x;
    int channel = blockIdx.y;
    int delay = blockIdx.z;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->num_delays > delay &&
        layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->learning &&
        !layers[layer]->d_kernels_cnvg[kernel] &&
        ((!channel && layers[layer]->kernel_channels == 1) || layers[layer]->out_maps == 1)) {

        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];

        if (kernel_local->learning_trigger &&
            kernel_local->d_delay_active[delay]) {

            // global memory optimization
            int kernel_channels = layers[layer]->kernel_channels;
            int rf_side = layers[layer]->rf_side;
            int rf_side_limits[] = {layers[layer]->rf_side_limits[0], layers[layer]->rf_side_limits[1]};
            int out_maps = layers[layer]->out_maps;
            int num_delays = layers[layer]->num_delays;
            float synapse_inh_scaling = layers[layer]->synapse_inh_scaling;
            int learning_type = layers[layer]->learning_type;

            bool* weights_delta_maps_channels = kernel_local->d_weights_delta_maps_channels;
            float *weights_exc_delta = kernel_local->d_weights_exc_delta;
            float *weights_exc = kernel_local->d_weights_exc;
            float *weights_inh = kernel_local->d_weights_inh;
            float *weights_total = kernel_local->d_weights_total;

            float *weights_inh_delta;
            if (synapse_inh_scaling > 0.f)
                weights_inh_delta = kernel_local->d_weights_inh_delta;

            // number of output maps contributing in this weight update
            int cnt_maps = 0;
            for (int m = 0; m < out_maps; m++) {
                for (int ch = 0; ch < kernel_channels; ch++) {
                    int idx_channel = m * kernel_channels + ch;
                    if (weights_delta_maps_channels[idx_channel]) {
                        cnt_maps++;
                        break;
                    }
                }
            }

            // weight update
            if (cnt_maps > 0) {
                for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                        float accum_exc = 0.f, accum_inh = 0.f;
                        int idx_syn = cols * rf_side + rows;
                        for (int ch = 0; ch < out_maps; ch++) {
                            int idx_syn_delta = (ch + channel) * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                            accum_exc += weights_exc_delta[idx_syn_delta];
                            if (synapse_inh_scaling > 0.f)
                                accum_inh += weights_inh_delta[idx_syn_delta];
                        }
                        int idx_syn_weights = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                        weights_exc[idx_syn_weights] += accum_exc / (float) cnt_maps;
                        if (synapse_inh_scaling > 0.f)
                            weights_inh[idx_syn_weights] += accum_inh / (float) cnt_maps;
                    }
                }
            }

            // prevent weights from exploding (if needed)
            if (learning_type == 2 || learning_type == 3) {
                if (layers[layer]->stdp_shrestha_gerstner_weight_boundaries) {
                    float stdp_shrestha_gerstner_weight_max = layers[layer]->stdp_shrestha_gerstner_weight_max;
                    for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                        for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                            int idx_syn = cols * rf_side + rows;
                            int idx_syn_weights = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;

                            if (weights_exc[idx_syn_weights] < 0.f)
                                weights_exc[idx_syn_weights] = 0.f;
                            else if (weights_exc[idx_syn_weights] > stdp_shrestha_gerstner_weight_max)
                                weights_exc[idx_syn_weights] = stdp_shrestha_gerstner_weight_max;

                            if (synapse_inh_scaling > 0.f) {
                                if (weights_inh[idx_syn_weights] > 0.f)
                                    weights_inh[idx_syn_weights] = 0.f;
                                else if (weights_inh[idx_syn_weights] < -stdp_shrestha_gerstner_weight_max)
                                    weights_inh[idx_syn_weights] = -stdp_shrestha_gerstner_weight_max;
                            }
                        }
                    }
                }
            }

            // update weights total (for this kernel, channel, and temporal slice)
            for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                    int idx_syn = cols * rf_side + rows;
                    int idx_syn_weights = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                    weights_total[idx_syn_weights] = weights_exc[idx_syn_weights] +
                            synapse_inh_scaling * weights_inh[idx_syn_weights];
                }
            }
        }
    }
}


// Shrestha's STDP
__global__ void stdp_shrestha_kernel_channel(Layer **layers) {

    int layer = blockIdx.x;
    int channel = blockIdx.y;
    int delay = blockIdx.z;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->num_delays > delay &&
        layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->learning &&
        layers[layer]->learning_type == 2 &&
        !layers[layer]->d_kernels_cnvg[kernel]) {

        int kernel_channels = layers[layer]->kernel_channels;
        int channel_inp = 0;
        int channel_out = 0;
        if (layers[layer]->out_maps == 1) channel_inp = channel;
        else if (kernel_channels == 1) channel_out = channel;

        bool weights_delta_maps_channels = false;
        if (layers[layer]->d_d_kernels[kernel]->learning_trigger &&
            layers[layer]->d_d_kernels[kernel]->d_delay_active[delay]) {

            // global memory optimization
            int rf_side = layers[layer]->rf_side;
            int rf_side_limits[] = {layers[layer]->rf_side_limits[0], layers[layer]->rf_side_limits[1]};
            int num_delays = layers[layer]->num_delays;
            float synapse_inh_scaling = layers[layer]->synapse_inh_scaling;
            int out_size = layers[layer]->out_size[1];
            int out_node_kernel = layers[layer]->out_node_kernel;
            int strides = layers[layer]->strides;
            int inp_size = layers[layer]->inp_size[1];
            int padding_total = layers[layer]->padding_total[0];
            int inp_node_kernel = layers[layer]->inp_node_kernel;
            int stdp_shrestha_gerstner_window_LTP = layers[layer]->stdp_shrestha_gerstner_window_LTP;
            float synapse_w_init = layers[layer]->synapse_w_init;
            float learning_rate = layers[layer]->learning_rate;
            int *stdp_precnt = layers[layer]->d_stdp_precnt;

            Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
            int *nodesep_train = kernel_local->d_nodesep_train;
            float *weights_exc_delta = kernel_local->d_weights_exc_delta;
            float *weights_exc = kernel_local->d_weights_exc;

            float *weights_inh_delta, *weights_inh;
            if (synapse_inh_scaling > 0.f) {
                weights_inh_delta = kernel_local->d_weights_inh_delta;
                weights_inh = kernel_local->d_weights_inh;
            }

            for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                    int idx_syn = cols * rf_side + rows;
                    int idx_syn_delta = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                    weights_exc_delta[idx_syn_delta] = 0.f;
                    if (synapse_inh_scaling > 0.f)
                        weights_inh_delta[idx_syn_delta] = 0.f;
                }
            }

            int cnt_nodes = 0;
            for (int node = 0; node < out_node_kernel; node++) {

                int idx_x_rf = node / out_size;
                int idx_y_rf = node % out_size;
                int idx_nodesep = channel_out * out_node_kernel + node;

                if (!nodesep_train[idx_nodesep])
                    continue;

                weights_delta_maps_channels = true;
                cnt_nodes++;

                for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                        int idx_xpad = idx_x_rf * strides + cols;
                        int idx_ypad = idx_y_rf * strides + rows;
                        int idx_nodepad = idx_xpad * (inp_size + padding_total) + idx_ypad;
                        int idx_syn_inp = channel * inp_node_kernel * num_delays + idx_nodepad * num_delays + delay;

                        int idx_syn = cols * rf_side + rows;
                        int idx_syn_weights = channel_inp * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                        int idx_syn_delta = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;

                        float diff_weights;
                        int delta_T = stdp_precnt[idx_syn_inp];

                        /* LTP */
                        if (delta_T >= 0 && delta_T < stdp_shrestha_gerstner_window_LTP) {

                            // excitatory synapses
                            diff_weights = weights_exc[idx_syn_weights] - synapse_w_init;
                            weights_exc_delta[idx_syn_delta] += learning_rate * exp(-diff_weights);

                            // inhibitory synapses
                            if (synapse_inh_scaling > 0.f) {
                                diff_weights = weights_inh[idx_syn_weights] + synapse_w_init;
                                weights_inh_delta[idx_syn_delta] += learning_rate * exp(-diff_weights);
                            }

                        /* LTD */
                        } else {

                            // excitatory synapses
                            diff_weights = weights_exc[idx_syn_weights] - synapse_w_init;
                            weights_exc_delta[idx_syn_delta] -= learning_rate * exp(diff_weights);

                            // inhibitory synapses
                            if (synapse_inh_scaling > 0.f) {
                                diff_weights = weights_inh[idx_syn_weights] + synapse_w_init;
                                weights_inh_delta[idx_syn_delta] -= learning_rate * exp(diff_weights);
                            }
                        }

                        // reset counters
                        stdp_precnt[idx_syn_inp] = -1;
                    }
                }
            }

            // average weight update
            if (cnt_nodes > 0) {
                for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                        int idx_syn = cols * rf_side + rows;
                        int idx_syn_delta = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                        weights_exc_delta[idx_syn_delta] /= (float) cnt_nodes;
                        if (synapse_inh_scaling > 0.f)
                            weights_inh_delta[idx_syn_delta] /= (float) cnt_nodes;
                    }
                }
            }
        }
        int idx_channel = channel_out * kernel_channels + channel_inp;
        layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = weights_delta_maps_channels;
    }
}


// Gerstner's STDP
__global__ void stdp_gerstner_kernel_channel(Layer **layers) {

    int layer = blockIdx.x;
    int channel = blockIdx.y;
    int delay = blockIdx.z;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->num_delays > delay &&
        layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->learning &&
        layers[layer]->learning_type == 3 &&
        !layers[layer]->d_kernels_cnvg[kernel]) {

        int kernel_channels = layers[layer]->kernel_channels;
        int channel_inp = 0;
        int channel_out = 0;
        if (layers[layer]->out_maps == 1) channel_inp = channel;
        else if (kernel_channels == 1) channel_out = channel;

        bool weights_delta_maps_channels = false;
        if (layers[layer]->d_d_kernels[kernel]->d_delay_active[delay]) {

            // global memory optimization
            int rf_side = layers[layer]->rf_side;
            int rf_side_limits[] = {layers[layer]->rf_side_limits[0], layers[layer]->rf_side_limits[1]};
            int num_delays = layers[layer]->num_delays;
            float synapse_inh_scaling = layers[layer]->synapse_inh_scaling;
            int out_node_kernel = layers[layer]->out_node_kernel;
            int out_size = layers[layer]->out_size[1];
            int strides = layers[layer]->strides;
            int inp_size = layers[layer]->inp_size[1];
            int padding_total = layers[layer]->padding_total[0];
            int inp_node_kernel = layers[layer]->inp_node_kernel;
            float learning_rate = layers[layer]->learning_rate;
            int stdp_shrestha_gerstner_window_LTP = layers[layer]->stdp_shrestha_gerstner_window_LTP;
            int stdp_shrestha_gerstner_window_LTD = layers[layer]->stdp_shrestha_gerstner_window_LTD;
            float stdp_shrestha_gerstner_weight_max = layers[layer]->stdp_shrestha_gerstner_weight_max;
            bool stdp_gerstner_weight_dependence = layers[layer]->stdp_gerstner_weight_dependence;
            int *stdp_precnt = layers[layer]->d_stdp_precnt;

            Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
            int *stdp_postcnt = kernel_local->d_stdp_postcnt;
            float *weights_exc_delta = kernel_local->d_weights_exc_delta;
            float *weights_exc = kernel_local->d_weights_exc;

            float *weights_inh_delta, *weights_inh;
            if (synapse_inh_scaling > 0.f) {
                weights_inh_delta = kernel_local->d_weights_inh_delta;
                weights_inh = kernel_local->d_weights_inh;
            }

            for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                    int idx_syn = cols * rf_side + rows;
                    int idx_syn_delta = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                    weights_exc_delta[idx_syn_delta] = 0.f;
                    if (synapse_inh_scaling > 0.f)
                        weights_inh_delta[idx_syn_delta] = 0.f;
                }
            }

            int cnt_nodes = 0;
            for (int node = 0; node < out_node_kernel; node++) {

                bool node_contrib = false;
                int idx_x_rf = node / out_size;
                int idx_y_rf = node % out_size;
                int idx_nodesep = channel_out * out_node_kernel + node;

                for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                        int idx_xpad = idx_x_rf * strides + cols;
                        int idx_ypad = idx_y_rf * strides + rows;
                        int idx_nodepad = idx_xpad * (inp_size + padding_total) + idx_ypad;
                        int idx_syn_inp = channel * inp_node_kernel * num_delays + idx_nodepad * num_delays + delay;

                        if (stdp_postcnt[idx_nodesep] == -1) continue;
                        int delta_T = stdp_precnt[idx_syn_inp] - stdp_postcnt[idx_nodesep];

                        int idx_syn = cols * rf_side + rows;
                        int idx_syn_weights = channel_inp * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                        int idx_syn_delta = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;

                        /* LTP */
                        if (delta_T >= 0 && delta_T <= stdp_shrestha_gerstner_window_LTP && !stdp_postcnt[idx_nodesep]) {

                            if (!node_contrib) {
                                cnt_nodes++;
                                node_contrib = true;
                                weights_delta_maps_channels = true;
                            }

                            float update = learning_rate * exp(-(float) delta_T / (float) stdp_shrestha_gerstner_window_LTP);

                            // excitatory synapses
                            if (stdp_gerstner_weight_dependence)
                                weights_exc_delta[idx_syn_delta] += update *
                                        (stdp_shrestha_gerstner_weight_max - weights_exc[idx_syn_weights]);
                            else weights_exc_delta[idx_syn_delta] += update;

                            // inhibitory synapses
                            if (synapse_inh_scaling > 0.f) {
                                if (stdp_gerstner_weight_dependence)
                                    weights_inh_delta[idx_syn_delta] -= update *
                                            abs(-stdp_shrestha_gerstner_weight_max - weights_inh[idx_syn_weights]);
                                else weights_inh_delta[idx_syn_delta] -= update;
                            }

                        /* LTD */
                        } else if (delta_T > stdp_shrestha_gerstner_window_LTP && !stdp_postcnt[idx_nodesep]) {

                            if (!node_contrib) {
                                cnt_nodes++;
                                node_contrib = true;
                                weights_delta_maps_channels = true;
                            }

                            // excitatory synapses
                            weights_exc_delta[idx_syn_delta] += -learning_rate;

                            // inhibitory synapses
                            if (synapse_inh_scaling > 0.f)
                                weights_inh_delta[idx_syn_delta] -= -learning_rate;

                        /* LTD */
                        } else if (delta_T < 0 && !stdp_precnt[idx_syn_inp]) {

                            if (!node_contrib) {
                                cnt_nodes++;
                                node_contrib = true;
                                weights_delta_maps_channels = true;
                            }

                            float update = -learning_rate * exp((float) delta_T / (float) stdp_shrestha_gerstner_window_LTD);

                            // excitatory synapses
                            if (stdp_gerstner_weight_dependence)
                                weights_exc_delta[idx_syn_delta] += update * weights_exc[idx_syn_weights];
                            else weights_exc_delta[idx_syn_delta] += update;

                            // inhibitory synapses
                            if (synapse_inh_scaling > 0.f) {
                                if (stdp_gerstner_weight_dependence)
                                    weights_inh_delta[idx_syn_delta] -= update * abs(weights_inh[idx_syn_weights]);
                                else weights_inh_delta[idx_syn_delta] -= update;
                            }
                        }
                    }
                }
            }

            // average weight update
            if (cnt_nodes > 0) {
                for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                        int idx_syn = cols * rf_side + rows;
                        int idx_syn_delta = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                        weights_exc_delta[idx_syn_delta] /= (float) cnt_nodes;
                        if (synapse_inh_scaling > 0.f)
                            weights_inh_delta[idx_syn_delta] /= (float) cnt_nodes;
                    }
                }
            }
        }
        int idx_channel = channel_out * kernel_channels + channel_inp;
        layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = weights_delta_maps_channels;
    }
}


// Kheradpisheh's STDP
__global__ void stdp_kheradpisheh_kernel_channel(Layer **layers) {

    int layer = blockIdx.x;
    int channel = blockIdx.y;
    int delay = blockIdx.z;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->num_delays > delay &&
        layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->learning &&
        layers[layer]->learning_type == 4 &&
        !layers[layer]->d_kernels_cnvg[kernel]) {

        int kernel_channels = layers[layer]->kernel_channels;
        int channel_inp = 0;
        int channel_out = 0;
        if (layers[layer]->out_maps == 1) channel_inp = channel;
        else if (kernel_channels == 1) channel_out = channel;

        bool weights_delta_maps_channels = false;
        if (layers[layer]->d_d_kernels[kernel]->d_delay_active[delay]) {

            // global memory optimization
            int rf_side = layers[layer]->rf_side;
            int rf_side_limits[] = {layers[layer]->rf_side_limits[0], layers[layer]->rf_side_limits[1]};
            int num_delays = layers[layer]->num_delays;
            float synapse_inh_scaling = layers[layer]->synapse_inh_scaling;
            int out_node_kernel = layers[layer]->out_node_kernel;
            int out_size = layers[layer]->out_size[1];
            int strides = layers[layer]->strides;
            int inp_size = layers[layer]->inp_size[1];
            int padding_total = layers[layer]->padding_total[0];
            int inp_node_kernel = layers[layer]->inp_node_kernel;
            float learning_rate = layers[layer]->learning_rate;
            int *stdp_precnt = layers[layer]->d_stdp_precnt;

            Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
            int *stdp_postcnt = kernel_local->d_stdp_postcnt;
            float *weights_exc_delta = kernel_local->d_weights_exc_delta;
            float *weights_exc = kernel_local->d_weights_exc;

            float *weights_inh_delta, *weights_inh;
            if (synapse_inh_scaling > 0.f) {
                weights_inh_delta = kernel_local->d_weights_inh_delta;
                weights_inh = kernel_local->d_weights_inh;
            }

            for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                    int idx_syn = cols * rf_side + rows;
                    int idx_syn_delta = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                    weights_exc_delta[idx_syn_delta] = 0.f;
                    if (synapse_inh_scaling > 0.f)
                        weights_inh_delta[idx_syn_delta] = 0.f;
                }
            }

            int cnt_nodes = 0;
            for (int node = 0; node < out_node_kernel; node++) {

                bool node_contrib = false;
                int idx_x_rf = node / out_size;
                int idx_y_rf = node % out_size;
                int idx_nodesep = channel_out * out_node_kernel + node;

                for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                        int idx_xpad = idx_x_rf * strides + cols;
                        int idx_ypad = idx_y_rf * strides + rows;
                        int idx_nodepad = idx_xpad * (inp_size + padding_total) + idx_ypad;
                        int idx_syn_inp = channel * inp_node_kernel * num_delays + idx_nodepad * num_delays + delay;

                        if (stdp_postcnt[idx_nodesep] == -1)
                            continue;
                        int delta_T = stdp_precnt[idx_syn_inp] - stdp_postcnt[idx_nodesep];

                        int idx_syn = cols * rf_side + rows;
                        int idx_syn_weights = channel_inp * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;
                        int idx_syn_delta = channel * rf_side * rf_side * num_delays + idx_syn * num_delays + delay;

                        /* LTP */
                        if (delta_T >= 0 && !stdp_postcnt[idx_nodesep]) {

                            if (!node_contrib) {
                                cnt_nodes++;
                                node_contrib = true;
                                weights_delta_maps_channels = true;
                            }

                            // excitatory synapses
                            weights_exc_delta[idx_syn_delta] += learning_rate *
                                    weights_exc[idx_syn_weights] * (1.f - weights_exc[idx_syn_weights]);

                            // inhibitory synapses
                            if (synapse_inh_scaling > 0.f)
                                weights_inh_delta[idx_syn_delta] -= learning_rate *
                                        abs(weights_inh[idx_syn_weights]) * (1.f - abs(weights_inh[idx_syn_weights]));

                        /* LTD */
                        } else if (delta_T < 0) {

                            if (!node_contrib) {
                                cnt_nodes++;
                                node_contrib = true;
                                weights_delta_maps_channels = true;
                            }

                            // excitatory synapses
                            weights_exc_delta[idx_syn_delta] -= learning_rate *
                                    weights_exc[idx_syn_weights] * (1.f - weights_exc[idx_syn_weights]);

                            // inhibitory synapses
                            if (synapse_inh_scaling > 0.f)
                                weights_inh_delta[idx_syn_delta] += learning_rate *
                                        abs(weights_inh[idx_syn_weights]) * (1.f - abs(weights_inh[idx_syn_weights]));
                        }
                    }
                }
            }

            // average weight update
            if (cnt_nodes > 0) {
                for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                        int idx_syn = cols * rf_side + rows;
                        int idx_syn_delta = channel * rf_side * rf_side *
                                            num_delays + idx_syn * num_delays + delay;
                        weights_exc_delta[idx_syn_delta] /= (float) cnt_nodes;
                        if (synapse_inh_scaling > 0.f)
                            weights_inh_delta[idx_syn_delta] /= (float) cnt_nodes;
                    }
                }
            }
        }
        int idx_channel = channel_out * kernel_channels + channel_inp;
        layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = weights_delta_maps_channels;
    }
}


// adapt delays
__global__ void drop_delays_kernel(Layer **layers, float *drop_delays_th) {

    int layer = blockIdx.x;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->learning &&
        layers[layer]->learning_type &&
        !layers[layer]->d_kernels_cnvg[kernel]) {

        if (layers[layer]->d_d_kernels[kernel]->learning_trigger &&
            layers[layer]->d_d_kernels[kernel]->num_delays_active > 1) {

            // global memory optimization
            int rf_side = layers[layer]->rf_side;
            int rf_side_limits[] = {layers[layer]->rf_side_limits[0], layers[layer]->rf_side_limits[1]};
            int kernel_channels = layers[layer]->kernel_channels;
            int num_delays = layers[layer]->num_delays;

            Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
            float *weights_exc = kernel_local->d_weights_exc;
            float *sum_exc_weights = kernel_local->d_sum_exc_weights;
            bool *delay_active = kernel_local->d_delay_active;
            int num_delays_active = kernel_local->num_delays_active;

            int delay_max_weights = -1;
            float max_weights = 0.f;
            for (int d = 0; d < num_delays_active; d++) {
                sum_exc_weights[d] = 0.f;
                for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                        for (int ch = 0; ch < kernel_channels; ch++) {
                            int idx_syn = cols * rf_side + rows;
                            int idx_syn_weights = ch * rf_side * rf_side * num_delays + idx_syn * num_delays + d;
                            sum_exc_weights[d] += weights_exc[idx_syn_weights];
                        }
                    }
                }
                if (sum_exc_weights[d] >= max_weights) {
                    delay_max_weights = d;
                    max_weights = sum_exc_weights[d];
                }
            }

            bool drop_delays = false;
            for (int d = delay_max_weights + 1; d < num_delays_active; d++) {
                if (drop_delays || sum_exc_weights[d] < sum_exc_weights[delay_max_weights] * drop_delays_th[0]) {
                    delay_active[d] = false;
                    drop_delays = true;
                }
            }

            num_delays_active = 0;
            for (int d = 0; d < num_delays; d++) {
                if (delay_active[d])
                    num_delays_active++;
            }
        }
    }
}


// firing nodes in a kernel
__global__ void firing_node_kernel(Layer **layers) {

    int layer = blockIdx.x;
    int node = blockIdx.y;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->inhibition &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->out_node_kernel > node &&
        (!layers[layer]->learning || !layers[layer]->inhibition_spatial)) {

        // global memory that we read only once
        int out_maps = layers[layer]->out_maps;
        int out_node_kernel = layers[layer]->out_node_kernel;
        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];

        int max_channel = -1;
        if (layers[layer]->firing_node) {
            float V_max = 0.f;
            float *nodesep_V = kernel_local->d_nodesep_V;
            int *nodesep_train = kernel_local->d_nodesep_train;
            for (int ch = 0; ch < out_maps; ch++) {
                int idx_nodesep = ch * out_node_kernel + node;
                if (nodesep_train[idx_nodesep] && nodesep_V[idx_nodesep] > V_max) {
                    V_max = nodesep_V[idx_nodesep];
                    max_channel = ch;
                }
            }
        }
        kernel_local->d_max_channel[node] = max_channel;
    }
}


// firing nodes in a layer
__global__ void firing_node(Layer **layers) {

    int layer = blockIdx.x;
    int node = blockIdx.y;

    if (layers[layer]->active &&
        layers[layer]->inhibition &&
        layers[layer]->out_node_kernel > node &&
        (!layers[layer]->learning || !layers[layer]->inhibition_spatial)) {

        // global memory that we read only once
        int cnt_kernels = layers[layer]->cnt_kernels;
        int out_node_kernel = layers[layer]->out_node_kernel;

        int max_kernel = -1;
        if (layers[layer]->firing_node) {
            float V_max = 0.f;
            for (int k = 0; k < cnt_kernels; k++) {
                Kernel *kernel_local = layers[layer]->d_d_kernels[k];
                int max_channel = kernel_local->d_max_channel[node];
                if (max_channel != -1) {
                    int idx_nodesep = max_channel * out_node_kernel + node;
                    float nodesep_V = kernel_local->d_nodesep_V[idx_nodesep];
                    if (nodesep_V > V_max) {
                        V_max = nodesep_V;
                        max_kernel = k;
                    }
                }
            }
        }
        layers[layer]->d_max_kernel[node] = max_kernel;
    }
}


// node-specific perpendicular inhibition
__global__ void perpendicular_inhibition(Layer **layers) {

    int layer = blockIdx.x;
    int node = blockIdx.y;
    int channel = blockIdx.z;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->inhibition &&
        layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->out_node_kernel > node &&
        (!layers[layer]->learning || !layers[layer]->inhibition_spatial) &&
        ((!channel && layers[layer]->out_maps == 1) || layers[layer]->kernel_channels == 1)) {

        int kernel_max = layers[layer]->d_max_kernel[node];
        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
        Kernel *kernel_max_local = layers[layer]->d_d_kernels[kernel_max];

        if (kernel_max != -1) {
            int channel_max = kernel_max_local->d_max_channel[node];
            if (layers[layer]->learning) kernel_max_local->learning_trigger = true;
            if (kernel != kernel_max || (kernel == kernel_max && channel != channel_max)) {
                int idx_nodesep = channel * layers[layer]->out_node_kernel + node;
                kernel_local->d_nodesep_train[idx_nodesep] = 0;
                kernel_local->d_nodesep_V[idx_nodesep] = 0.f;
                kernel_local->d_nodesep_refrac[idx_nodesep] = 0.f;

                // update counters for Gerstner's and Kheradpisheh's STDP
                if (layers[layer]->learning_type == 3 ||
                    layers[layer]->learning_type == 4)
                    kernel_local->d_stdp_postcnt[idx_nodesep] = -1;
            }
        }
    }
}


// update convergence tracking vectors
__global__ void stdp_paredes_track_convergence_channel(Layer **layers) {

    int layer = blockIdx.x;
    int channel = blockIdx.y;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->learning &&
        layers[layer]->learning_type == 1 &&
        layers[layer]->firing_node &&
        !layers[layer]->d_kernels_cnvg[kernel] &&
        ((!channel && layers[layer]->out_maps == 1) || layers[layer]->kernel_channels == 1)) {

        layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps[channel] = false;

        // global memory optimization
        int rf_side = layers[layer]->rf_side;
        int rf_side_limits[] = {layers[layer]->rf_side_limits[0], layers[layer]->rf_side_limits[1]};
        int num_delays = layers[layer]->num_delays;
        float synapse_inh_scaling = layers[layer]->synapse_inh_scaling;
        int kernel_channels = layers[layer]->kernel_channels;
        int num_delays_active = layers[layer]->d_d_kernels[kernel]->num_delays_active;
        float synapse_w_init = layers[layer]->synapse_w_init;
        int out_node_kernel = layers[layer]->out_node_kernel;
        int out_size = layers[layer]->out_size[1];
        int strides = layers[layer]->strides;
        int inp_size = layers[layer]->inp_size[1];
        int padding_total = layers[layer]->padding_total[0];
        int inp_node_kernel = layers[layer]->inp_node_kernel;
        float *synapse_pretrace = layers[layer]->d_synapse_pretrace;

        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
        int *nodesep_train = kernel_local->d_nodesep_train;
        float *nodesep_maxpretrace = kernel_local->d_nodesep_maxpretrace;
        float *weights_exc = kernel_local->d_weights_exc;

        float *weights_inh;
        if (synapse_inh_scaling > 0.f)
            weights_inh = kernel_local->d_weights_inh;

        // max weights in a kernel
        float max_weight_exc = 0.f;
        float max_weight_inh = 0.f;
        for (int ch = 0; ch < kernel_channels; ch++) {
            for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                    for (int d = 0; d < num_delays_active; d++) {
                        int idx_syn = cols * rf_side + rows;
                        int idx_syn_weights = ch * rf_side * rf_side * num_delays + idx_syn * num_delays + d;
                        if (max_weight_exc < weights_exc[idx_syn_weights])
                            max_weight_exc = weights_exc[idx_syn_weights];
                        if (synapse_inh_scaling > 0.f &&
                            max_weight_inh < weights_inh[idx_syn_weights] + 2.f * synapse_w_init)
                            max_weight_inh = weights_inh[idx_syn_weights] + 2.f * synapse_w_init;
                    }
                }
            }
        }

        // objective function computation (MSE trace-weight)
        int stdp_objective_cnt = 0;
        float stdp_paredes_objective_avg = 0.f;
        bool weights_delta_maps = false;
        for (int node = 0; node < out_node_kernel; node++) {

            int idx_nodesep = channel * out_node_kernel + node;
            if (nodesep_train[idx_nodesep]) {
                weights_delta_maps = true;
                int idx_x_rf = node / out_size;
                int idx_y_rf = node % out_size;

                for (int ch = 0; ch < kernel_channels; ch++) {
                    for (int rows = 0; rows < rf_side - rf_side_limits[0]; rows++) {
                        for (int cols = 0; cols < rf_side - rf_side_limits[1]; cols++) {
                            int idx_syn = cols * rf_side + rows;
                            int idx_xpad = idx_x_rf * strides + cols;
                            int idx_ypad = idx_y_rf * strides + rows;
                            int idx_nodepad = idx_xpad * (inp_size + padding_total) + idx_ypad;

                            for (int d = 0; d < num_delays_active; d++) {
                                int idx_syn_weights = ch * rf_side * rf_side * num_delays + idx_syn * num_delays + d;
                                int idx_syn_inp = (ch + channel) * inp_node_kernel * num_delays + idx_nodepad * num_delays + d;

                                // excitatory synapses
                                stdp_paredes_objective_avg += pow(synapse_pretrace[idx_syn_inp] / nodesep_maxpretrace[idx_nodesep] -
                                        weights_exc[idx_syn_weights] / max_weight_exc, 2.f);

                                // inhibitory synapses
                                if (synapse_inh_scaling > 0.f)
                                    stdp_paredes_objective_avg += pow(synapse_pretrace[idx_syn_inp] / nodesep_maxpretrace[idx_nodesep] -
                                                (weights_inh[idx_syn_weights] + 2.f * synapse_w_init) / max_weight_inh, 2.f);
                            }
                        }
                    }
                }
                stdp_objective_cnt++;
            }
        }

        // update convergence data
        if (stdp_objective_cnt > 0) {
            stdp_paredes_objective_avg /= (float) (stdp_objective_cnt * rf_side * rf_side * num_delays_active);
            if (synapse_inh_scaling > 0.f)
                stdp_paredes_objective_avg /= 2.f;
        }
        kernel_local->d_stdp_paredes_objective_avg[channel] = stdp_paredes_objective_avg;
        kernel_local->d_weights_delta_maps[channel] = weights_delta_maps;
    }
}


// update convergence tracking vectors
__global__ void stdp_paredes_track_convergence(Layer **layers) {

    int layer = blockIdx.x;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->learning &&
        layers[layer]->learning_type == 1 &&
        layers[layer]->firing_node &&
        !layers[layer]->d_kernels_cnvg[kernel]) {

        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
        bool* weights_delta_maps = kernel_local->d_weights_delta_maps;
        float* stdp_paredes_objective_avg_ch = kernel_local->d_stdp_paredes_objective_avg;

        // compute objective function
        float accum = 0.f;
        int stdp_objective_cnt = 0;
        for (int ch = 0; ch < layers[layer]->out_maps; ch++) {
            if (weights_delta_maps[ch]) {
                accum += stdp_paredes_objective_avg_ch[ch];
                stdp_objective_cnt++;
            }
        }

        // compute moving average of objective function
        if (stdp_objective_cnt > 0) {
            int stdp_paredes_stats_window = layers[layer]->stdp_paredes_stats_window;
            float* stdp_paredes_objective = kernel_local->d_stdp_paredes_objective;

            for (int i = stdp_paredes_stats_window; i > 0; i--)
                stdp_paredes_objective[i] = stdp_paredes_objective[i-1];
            stdp_paredes_objective[0] = accum / (float) stdp_objective_cnt;

            float stdp_paredes_objective_avg = 0.f;
            for (int i = 0; i < stdp_paredes_stats_window; i++)
                stdp_paredes_objective_avg += stdp_paredes_objective[i];
            stdp_paredes_objective_avg /= (float) stdp_paredes_stats_window;
            kernel_local->stdp_paredes_objective_avg = stdp_paredes_objective_avg;

            if (stdp_paredes_objective_avg < layers[layer]->stdp_paredes_convg_th)
                layers[layer]->d_kernels_cnvg[kernel] = true;
        }
    }
}


// update output information
__global__ void update_output_channels(Layer **layers) {

    int layer = blockIdx.x;
    int node = blockIdx.y;
    int channel = blockIdx.z;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->inp_size[0] > channel &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->out_node_kernel > node &&
        ((!channel && layers[layer]->out_maps == 1) || layers[layer]->kernel_channels == 1)) {

        int idx_nodesep = channel * layers[layer]->out_node_kernel + node;

        // reset firing nuron
        if (layers[layer]->d_d_kernels[kernel]->d_nodesep_train[idx_nodesep]) {
            layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep] = 0.f;
            layers[layer]->d_d_kernels[kernel]->d_nodesep_refrac[idx_nodesep] = 0.f;
        }
    }
}


// update output information
__global__ void update_output(Layer **layers, int *histogram, int *histogram_type, int *cnt_layers) {

    int layer = blockIdx.x;
    int node = blockIdx.y;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->out_node_kernel > node) {

        layers[layer]->firing_node = false;
        int out_node_kernel = layers[layer]->out_node_kernel;
        int length_delay_out = layers[layer]->length_delay_out;

        Kernel *kernel_local = layers[layer]->d_d_kernels[kernel];
        int *nodesep_train = kernel_local->d_nodesep_train;
        int *node_train = kernel_local->d_node_train;

        int begin_vector = node * length_delay_out;
        int end_vector = (node + 1) * length_delay_out;
        for (int i = end_vector-1; i > begin_vector; i--)
            node_train[i] = node_train[i-1];

        node_train[begin_vector] = 0;
        for (int ch = 0; ch < layers[layer]->out_maps; ch++) {
            int idx_nodesep = ch * out_node_kernel + node;
            if (nodesep_train[idx_nodesep]) {
                node_train[begin_vector] = 1;
                nodesep_train[idx_nodesep] = 0;
                if (!ch && layer == cnt_layers[0]-1 && histogram_type[0] > 0) // histograms (only for out_maps == 0)
                    histogram[kernel * out_node_kernel + node] += 1;
            }
        }
    }
}


// update SPM histogram
__global__ void update_SPM_histogram(Layer **layers, int *cnt_layers, int *histogram, int *histogram_SPM) {

    int layer = blockIdx.x;
    int kernel = threadIdx.x;

    if (layer == cnt_layers[0]-1 &&
        layers[layer]->cnt_kernels > kernel) {

        int out_size[] = {layers[layer]->out_size[0], layers[layer]->out_size[1], layers[layer]->out_size[2]};
        int cnt_kernels = layers[layer]->cnt_kernels;
        int out_node_kernel = layers[layer]->out_node_kernel;

        for (int i = 0; i < out_node_kernel; i++) {
            int cols = i / out_size[1];
            int rows = i % out_size[1];
            for (int level = 0; level < 3; level++) {
                int row_SPM = (int) pow(2.f, (float) level) * rows / out_size[1];
                int col_SPM = (int) pow(2.f, (float) level) * cols / out_size[2];
                if (row_SPM <= level && col_SPM <= level) {
                    int idx = kernel * (int) pow(2.f, (float) 2*level) + row_SPM * (int) pow(2.f, (float) level) + col_SPM;
                    for (int level_aux = 0; level_aux < level; level_aux++)
                        idx += cnt_kernels * (int) pow(2.f, (float) 2*level_aux);
                    histogram_SPM[idx] += histogram[kernel * out_node_kernel + i];
                }
            }
        }
    }
}


// limit the number of STDP updates
__global__ void learning_limit_updates(Layer **layers) {

    int layer = blockIdx.x;

    if (layers[layer]->active &&
        layers[layer]->learning &&
        layers[layer]->learning_type) {

        int cnt_kernels = layers[layer]->cnt_kernels;
        int out_node_kernel = layers[layer]->out_node_kernel;
        int learning_updates_cnt = layers[layer]->learning_updates_cnt;

        bool spike = false;
        for (int k = 0; k < cnt_kernels; k++) {
            for (int i = 0; i < out_node_kernel; i++) {
                if (layers[layer]->d_d_kernels[k]->d_node_train[i * layers[layer]->length_delay_out]) {
                    learning_updates_cnt++;
                    spike = true;
                    break;
                }
                if (spike) break;
            }
        }

        if (learning_updates_cnt > layers[layer]->learning_limit_updates &&
            layers[layer]->learning_limit_updates > 0) {
            learning_updates_cnt = 0;
            layers[layer]->learning = false;
            layers[layer]->limit_learning = true;
        }
        layers[layer]->learning_updates_cnt = learning_updates_cnt;
    }
}