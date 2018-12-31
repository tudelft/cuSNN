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

        int idx_xpad = node / (layers[layer]->inp_size[1] + layers[layer]->padding_total[0]);
        int idx_ypad = node % (layers[layer]->inp_size[1] + layers[layer]->padding_total[0]);

        int idx_syn_inp = channel * layers[layer]->inp_node_kernel * layers[layer]->num_delays +
                node * layers[layer]->num_delays + delay;

        if (idx_ypad < layers[layer]->padding[0] ||
            idx_ypad >= layers[layer]->inp_size[1] + layers[layer]->padding[0] ||
            idx_xpad < layers[layer]->padding[1] ||
            idx_xpad >= layers[layer]->inp_size[2] + layers[layer]->padding[1]) {

            // pre-synaptic trace for zero-padding nodes
            if (layers[layer]->learning) layers[layer]->d_synapse_pretrace[idx_syn_inp] = 1000.f;
            else layers[layer]->d_synapse_pretrace[idx_syn_inp] = 0.f;

        } else {

            float value;
            int idx_y = idx_ypad - layers[layer]->padding[0];
            int idx_x = idx_xpad - layers[layer]->padding[1];
            int idx_node = idx_x * layers[layer]->inp_size[1] + idx_y;

            // spikes received by the nodes after delay
            if (!layer) {
                int delay_index = channel * layers[layer]->inp_size[1] * layers[layer]->inp_size[2] *
                        layers[layer]->length_delay_inp + idx_node * layers[layer]->length_delay_inp +
                        layers[layer]->d_delay_indices[delay];
                value = (float) inputs[delay_index];
            } else {
                int delay_index = idx_node * layers[layer]->length_delay_inp + layers[layer]->d_delay_indices[delay];
                value = (float) layers[layer-1]->d_d_kernels[channel]->d_node_train[delay_index];
            }

            // update pre-synaptic traces
            layers[layer]->d_synapse_pretrace[idx_syn_inp] += (sim_step[0] / layers[layer]->decay) *
                    (-layers[layer]->d_synapse_pretrace[idx_syn_inp] + layers[layer]->alpha * value);
            if (layers[layer]->d_synapse_pretrace[idx_syn_inp] < 0.f)
                layers[layer]->d_synapse_pretrace[idx_syn_inp] = 0.f;

            // update counters for Shrestha's, Gerstner's, and Kheradpisheh's STDP
            if (layers[layer]->learning_type == 2 ||
                layers[layer]->learning_type == 3 ||
                layers[layer]->learning_type == 4) {
                if (value > 0.f) layers[layer]->d_stdp_precnt[idx_syn_inp] = 0;
                else if (layers[layer]->d_stdp_precnt[idx_syn_inp] >= 0)
                    layers[layer]->d_stdp_precnt[idx_syn_inp]++;
            }
        }
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

        int channel_inp = 0;
        if (layers[layer]->out_maps == 1) channel_inp = channel;

        int idx_x_rf = node / layers[layer]->out_size[1];
        int idx_y_rf = node % layers[layer]->out_size[1];
        int idx_nodesep_channel = channel * layers[layer]->out_node_kernel + node;
        layers[layer]->d_d_kernels[kernel]->d_nodesep_channel_input[idx_nodesep_channel] = 0.f;

        // prevent kernels that didn't converge to accumulate spikes
        if (layers[layer]->learning || (!layers[layer]->enable_learning && layers[layer]->d_kernels_cnvg[kernel])) {

            layers[layer]->active = true;
            for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {

                    float value;
                    int idx_xpad = idx_x_rf * layers[layer]->strides + cols;
                    int idx_ypad = idx_y_rf * layers[layer]->strides + rows;

                    for (int d = 0; d < layers[layer]->d_d_kernels[kernel]->num_delays_active; d++) {

                        if (idx_ypad < layers[layer]->padding[0] ||
                            idx_ypad >= layers[layer]->inp_size[1] + layers[layer]->padding[0] ||
                            idx_xpad < layers[layer]->padding[1] ||
                            idx_xpad >= layers[layer]->inp_size[2] + layers[layer]->padding[1]) value = 0.f;
                        else {

                            int idx_y = idx_ypad - layers[layer]->padding[0];
                            int idx_x = idx_xpad - layers[layer]->padding[1];
                            int idx_node = idx_x * layers[layer]->inp_size[1] + idx_y;

                            // spikes received after synaptic delay
                            if (!layer) {
                                int delay_index = channel * layers[layer]->inp_size[1] * layers[layer]->inp_size[2] *
                                        layers[layer]->length_delay_inp + idx_node * layers[layer]->length_delay_inp +
                                        layers[layer]->d_delay_indices[d];
                                value = (float) inputs[delay_index];
                            } else {
                                int delay_index = idx_node * layers[layer]->length_delay_inp +
                                        layers[layer]->d_delay_indices[d];
                                value = (float) layers[layer-1]->d_d_kernels[channel]->d_node_train[delay_index];
                            }
                        }

                        // propagate input spikes
                        int idx_syn = cols * layers[layer]->rf_side + rows;
                        int idx_syn_weights = channel_inp * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + d;
                        layers[layer]->d_d_kernels[kernel]->d_nodesep_channel_input[idx_nodesep_channel] += value *
                                (layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] +
                                layers[layer]->synapse_inh_scaling *
                                layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights]);
                    }
                }
            }
        }
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

        int idx_nodesep = channel * layers[layer]->out_node_kernel + node;
        layers[layer]->d_d_kernels[kernel]->learning_trigger = false;
        layers[layer]->d_d_kernels[kernel]->d_nodesep_input[idx_nodesep] = 0.f;
        layers[layer]->d_d_kernels[kernel]->d_nodesep_pretrace[idx_nodesep] = 0.f;
        layers[layer]->d_d_kernels[kernel]->d_nodesep_maxpretrace[idx_nodesep] = 0.f;

        // add inputs
        for (int ch = 0; ch < layers[layer]->kernel_channels; ch++) {
            int idx_nodesep_aux = (ch + channel) * layers[layer]->out_node_kernel + node;
            layers[layer]->d_d_kernels[kernel]->d_nodesep_input[idx_nodesep] +=
                    layers[layer]->d_d_kernels[kernel]->d_nodesep_channel_input[idx_nodesep_aux];
        }

        int idx_x_rf = node / layers[layer]->out_size[1];
        int idx_y_rf = node % layers[layer]->out_size[1];
        for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
            for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {

                int idx_xpad = idx_x_rf * layers[layer]->strides + cols;
                int idx_ypad = idx_y_rf * layers[layer]->strides + rows;
                int idx_nodepad = idx_xpad * (layers[layer]->inp_size[1] + layers[layer]->padding_total[0]) + idx_ypad;

                if (idx_ypad >= layers[layer]->padding[0] ||
                    idx_ypad < layers[layer]->inp_size[1] + layers[layer]->padding[0] ||
                    idx_xpad >= layers[layer]->padding[1] ||
                    idx_xpad < layers[layer]->inp_size[2] + layers[layer]->padding[1]) {

                    for (int ch = 0; ch < layers[layer]->kernel_channels; ch++) {
                        for (int d = 0; d < layers[layer]->d_d_kernels[kernel]->num_delays_active; d++) {
                            int idx_syn_inp = (ch + channel) * layers[layer]->inp_node_kernel *
                                    layers[layer]->num_delays + idx_nodepad * layers[layer]->num_delays + d;

                            // node cumulative pretrace
                            layers[layer]->d_d_kernels[kernel]->d_nodesep_pretrace[idx_nodesep] +=
                                    layers[layer]->d_synapse_pretrace[idx_syn_inp];

                            // max pretrace of receptive field
                            if (layers[layer]->d_synapse_pretrace[idx_syn_inp] >
                                layers[layer]->d_d_kernels[kernel]->d_nodesep_maxpretrace[idx_nodesep])
                                layers[layer]->d_d_kernels[kernel]->d_nodesep_maxpretrace[idx_nodesep] =
                                        layers[layer]->d_synapse_pretrace[idx_syn_inp];
                        }
                    }
                }
            }
        }
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

        int idx_x_rf = node / layers[layer]->out_size[1];
        int idx_y_rf = node % layers[layer]->out_size[1];
        int idx_nodesep = channel * layers[layer]->out_node_kernel + node;

        // homeostasis mechanism
        float max_trace = 0.f;
        if (layers[layer]->homeostasis) {
            for (int rows = -1; rows <= 1; rows++) {
                for (int cols = -1; cols <= 1; cols++) {
                    int idx_node = (idx_x_rf + cols) * layers[layer]->out_size[1] + idx_y_rf + rows;
                    int idx_nodesep_aux = channel * layers[layer]->out_node_kernel + idx_node;
                    if (idx_node >= 0 && idx_node < layers[layer]->out_node_kernel) {
                        if (max_trace < layers[layer]->d_d_kernels[kernel]->d_nodesep_pretrace[idx_nodesep_aux])
                            max_trace = layers[layer]->d_d_kernels[kernel]->d_nodesep_pretrace[idx_nodesep_aux];
                    }
                }
            }
        }

        // update refractory counter
        layers[layer]->d_d_kernels[kernel]->d_nodesep_refrac[idx_nodesep]++;

        // update V if node is not in refractory period
        if (layers[layer]->d_d_kernels[kernel]->d_nodesep_refrac[idx_nodesep] * sim_step[0] >= refrac[0])
            layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep] +=
                    (sim_step[0]/layers[layer]->decay) *
                    (-layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep] - max_trace +
                     layers[layer]->d_d_kernels[kernel]->d_nodesep_input[idx_nodesep]);
        if (layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep] < 0.f)
            layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep] = 0.f;

        // firing threshold
        float node_Vth = layers[layer]->node_Vth;
        if (layers[layer]->threshold_diehl)
            node_Vth += layers[layer]->d_d_kernels[kernel]->d_threshold_diehl_nodesep_theta[idx_nodesep];

        // update Diehl's adaptive threshold
        if (layers[layer]->threshold_diehl) {
            layers[layer]->d_d_kernels[kernel]->d_threshold_diehl_nodesep_theta[idx_nodesep] -=
                    sim_step[0] * layers[layer]->d_d_kernels[kernel]->d_threshold_diehl_nodesep_theta[idx_nodesep] /
                    layers[layer]->decay;
            if (layers[layer]->d_d_kernels[kernel]->d_threshold_diehl_nodesep_theta[idx_nodesep] < 0.f)
                layers[layer]->d_d_kernels[kernel]->d_threshold_diehl_nodesep_theta[idx_nodesep] = 0.f;
        }

        // spike generation
        layers[layer]->d_d_kernels[kernel]->d_nodesep_perpendicular[idx_nodesep] = 0;
        if (layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep] >= node_Vth) {
            layers[layer]->d_d_kernels[kernel]->d_nodesep_train[idx_nodesep] = 1;
            layers[layer]->d_d_kernels[kernel]->d_nodesep_perpendicular[idx_nodesep] = 1;
            layers[layer]->firing_node = true;
            if (layers[layer]->learning && !layers[layer]->inhibition)
                layers[layer]->d_d_kernels[kernel]->learning_trigger = true;
            if (layers[layer]->threshold_diehl)
                layers[layer]->d_d_kernels[kernel]->d_threshold_diehl_nodesep_theta[idx_nodesep] +=
                        layers[layer]->threshold_diehl_increase;
        }

        // update counters for Gerstner's and Kheradpisheh's STDP
        if (layers[layer]->learning_type == 3 ||
            layers[layer]->learning_type == 4) {
            if (layers[layer]->d_d_kernels[kernel]->d_nodesep_train[idx_nodesep])
                layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep] = 0;
            else if (layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep] >= 0)
                layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep]++;
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

        layers[layer]->d_d_kernels[kernel]->d_V_max[channel] = 0.f;
        layers[layer]->d_d_kernels[kernel]->d_node_max[channel] = -1;

        if (layers[layer]->active &&
            layers[layer]->firing_node &&
            layers[layer]->inhibition &&
            layers[layer]->learning &&
            layers[layer]->learning_type &&
            layers[layer]->inhibition_spatial) {

            for (int node = 0; node < layers[layer]->out_node_kernel; node++) {
                int idx_nodesep = channel * layers[layer]->out_node_kernel + node;
                if (layers[layer]->d_d_kernels[kernel]->d_nodesep_perpendicular[idx_nodesep] &&
                    layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep] >
                    layers[layer]->d_d_kernels[kernel]->d_V_max[channel]) {

                    layers[layer]->d_d_kernels[kernel]->d_V_max[channel] =
                            layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep];
                    layers[layer]->d_d_kernels[kernel]->d_node_max[channel] = node;
                }
            }
        }
    }
}


// STDP select firing node in each kernel
__global__ void spatial_firing_node_kernel(Layer **layers) {

    int layer = blockIdx.x;
    int kernel = threadIdx.x;

    if (layers[layer]->cnt_kernels > kernel) {

        layers[layer]->d_d_kernels[kernel]->V_max = 0.f;
        layers[layer]->d_d_kernels[kernel]->node_max = -1;
        layers[layer]->d_d_kernels[kernel]->channel_max = -1;

        if (layers[layer]->active &&
            layers[layer]->firing_node &&
            layers[layer]->inhibition &&
            layers[layer]->learning &&
            layers[layer]->learning_type &&
            layers[layer]->inhibition_spatial) {

            for (int ch = 0; ch < layers[layer]->out_maps; ch++) {
                if (layers[layer]->d_d_kernels[kernel]->d_V_max[ch] >
                    layers[layer]->d_d_kernels[kernel]->V_max) {
                    layers[layer]->d_d_kernels[kernel]->V_max = layers[layer]->d_d_kernels[kernel]->d_V_max[ch];
                    layers[layer]->d_d_kernels[kernel]->node_max = layers[layer]->d_d_kernels[kernel]->d_node_max[ch];
                    layers[layer]->d_d_kernels[kernel]->channel_max = ch;
                }
            }
        }
    }
}


// STDP select firing node
__global__ void spatial_firing_node(Layer **layers) {

    int layer = blockIdx.x;

    layers[layer]->kernel_max = -1;
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
                layers[layer]->kernel_max = k;
            }
        }

        if (layers[layer]->kernel_max != -1) {
            int node = layers[layer]->d_d_kernels[layers[layer]->kernel_max]->node_max;
            int channel = layers[layer]->d_d_kernels[layers[layer]->kernel_max]->channel_max;
            int idx_nodesep = channel * layers[layer]->out_node_kernel + node;
            layers[layer]->d_d_kernels[layers[layer]->kernel_max]->d_nodesep_perpendicular[idx_nodesep] = 0;
            layers[layer]->d_d_kernels[layers[layer]->kernel_max]->learning_trigger = true;
        }
    }
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

        int kernel_max = layers[layer]->kernel_max;
        int channel_max = layers[layer]->d_d_kernels[kernel_max]->channel_max;
        int node_max = layers[layer]->d_d_kernels[kernel_max]->node_max;
        int idx_x_rf = node_max / layers[layer]->out_size[1];
        int idx_y_rf = node_max % layers[layer]->out_size[1];

        for (int rows = -layers[layer]->neigh_inh; rows <= layers[layer]->neigh_inh; rows++) {
            for (int cols = -layers[layer]->neigh_inh; cols <= layers[layer]->neigh_inh; cols++) {

                int idx_x_rf2 = idx_x_rf + cols;
                int idx_y_rf2 = idx_y_rf + rows;
                int idx_node = idx_x_rf2 * layers[layer]->out_size[1] + idx_y_rf2;

                if (idx_node >= 0 && idx_node < layers[layer]->out_node_kernel) {
                    if (kernel != kernel_max ||
                        (kernel == kernel_max && channel != channel_max) ||
                        (kernel == kernel_max && channel == channel_max && idx_node != node_max)) {

                        int idx_nodesep = channel * layers[layer]->out_node_kernel + idx_node;
                        layers[layer]->d_d_kernels[kernel]->d_nodesep_train[idx_nodesep] = 0;
                        layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep] = 0.f;
                        layers[layer]->d_d_kernels[kernel]->d_nodesep_refrac[idx_nodesep] = 0.f;

                        // update counters for Gerstner's and Kheradpisheh's STDP
                        if (layers[layer]->learning_type == 3 ||
                            layers[layer]->learning_type == 4)
                            layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep] = -1;
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

        int channel_inp = 0;
        int channel_out = 0;
        if (layers[layer]->out_maps == 1) channel_inp = channel;
        else if (layers[layer]->kernel_channels == 1) channel_out = channel;

        int idx_channel = channel_out * layers[layer]->kernel_channels + channel_inp;
        layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = false;

        if (layers[layer]->d_d_kernels[kernel]->learning_trigger &&
            layers[layer]->d_d_kernels[kernel]->d_delay_active[delay]) {

            for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                    int idx_syn = cols * layers[layer]->rf_side + rows;
                    int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                            layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                    layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] = 0.f;
                    if (layers[layer]->synapse_inh_scaling > 0.f)
                        layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] = 0.f;
                }
            }

            int cnt_nodes = 0;
            for (int node = 0; node < layers[layer]->out_node_kernel; node++) {
                int idx_nodesep = channel_out * layers[layer]->out_node_kernel + node;
                if (layers[layer]->d_d_kernels[kernel]->d_nodesep_train[idx_nodesep]) {

                    cnt_nodes++;
                    layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = true;

                    int idx_x_rf = node / layers[layer]->out_size[1];
                    int idx_y_rf = node % layers[layer]->out_size[1];

                    for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                        for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {

                            int idx_xpad = idx_x_rf * layers[layer]->strides + cols;
                            int idx_ypad = idx_y_rf * layers[layer]->strides + rows;
                            int idx_nodepad = idx_xpad * 
                                    (layers[layer]->inp_size[1] + layers[layer]->padding_total[0]) + idx_ypad;

                            int idx_syn = cols * layers[layer]->rf_side + rows;
                            int idx_syn_weights = channel_inp * layers[layer]->rf_side * layers[layer]->rf_side *
                                    layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                            int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                                    layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                            int idx_syn_inp = channel * layers[layer]->inp_node_kernel *
                                    layers[layer]->num_delays + idx_nodepad * layers[layer]->num_delays + delay;

                            /* Long-Term Potentiation (LTP) and Long-Term Depression (LTD) */
                            // excitatory synapses
                            float diff_weights = layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] -
                                    layers[layer]->synapse_w_init;
                            float trace_norm = layers[layer]->d_synapse_pretrace[idx_syn_inp] /
                                    layers[layer]->d_d_kernels[kernel]->d_nodesep_maxpretrace[idx_nodesep];
                            float LTP = exp(-diff_weights) * (exp(trace_norm) - layers[layer]->stdp_paredes_a);
                            float LTD = exp(diff_weights) * (exp(1.f - trace_norm) - layers[layer]->stdp_paredes_a);
                            layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] +=
                                    layers[layer]->learning_rate * (LTP - LTD);

                            // inhibitory synapses
                            if (layers[layer]->synapse_inh_scaling > 0.f) {
                                diff_weights = layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights] +
                                        layers[layer]->synapse_w_init;
                                LTP = exp(-diff_weights) * (exp(trace_norm) - layers[layer]->stdp_paredes_a);
                                LTD = exp(diff_weights) * (exp(1.f - trace_norm) - layers[layer]->stdp_paredes_a);
                                layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] +=
                                        layers[layer]->learning_rate * (LTP - LTD);
                            }
                        }
                    }
                }
            }

            // average weight update
            if (cnt_nodes > 0) {
                for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                        int idx_syn = cols * layers[layer]->rf_side + rows;
                        int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                        layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] /= (float) cnt_nodes;
                        if (layers[layer]->synapse_inh_scaling > 0.f)
                            layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] /= (float) cnt_nodes;
                    }
                }
            }
        }
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

        if (layers[layer]->d_d_kernels[kernel]->learning_trigger &&
            layers[layer]->d_d_kernels[kernel]->d_delay_active[delay]) {

            // number of output maps contributing in this weight update
            int cnt_maps = 0;
            for (int m = 0; m < layers[layer]->out_maps; m++) {
                for (int ch = 0; ch < layers[layer]->kernel_channels; ch++) {
                    int idx_channel = m * layers[layer]->kernel_channels + ch;
                    if (layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel]) {
                        cnt_maps++;
                        break;
                    }
                }
            }

            // weight update
            if (cnt_maps > 0) {
                for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                        float accum_exc = 0.f, accum_inh = 0.f;
                        int idx_syn = cols * layers[layer]->rf_side + rows;
                        for (int ch = 0; ch < layers[layer]->out_maps; ch++) {
                            int idx_syn_delta = (ch + channel) * layers[layer]->rf_side * layers[layer]->rf_side *
                                    layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                            accum_exc += layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta];
                            if (layers[layer]->synapse_inh_scaling > 0.f)
                                accum_inh += layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta];
                        }
                        int idx_syn_weights = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                        layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] += accum_exc /
                                (float) cnt_maps;
                        if (layers[layer]->synapse_inh_scaling > 0.f)
                            layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights] += accum_inh /
                                    (float) cnt_maps;
                    }
                }
            }

            // prevent weights from exploding (if needed)
            if (layers[layer]->learning_type == 2 || layers[layer]->learning_type == 3) {
                if (layers[layer]->stdp_shrestha_gerstner_weight_boundaries) {
                    for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                        for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                            int idx_syn = cols * layers[layer]->rf_side + rows;
                            int idx_syn_weights = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                                    layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;

                            if (layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] < 0.f)
                                layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] = 0.f;
                            else if (layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] >
                                     layers[layer]->stdp_shrestha_gerstner_weight_max)
                                layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] =
                                        layers[layer]->stdp_shrestha_gerstner_weight_max;

                            if (layers[layer]->synapse_inh_scaling > 0.f) {
                                if (layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights] > 0.f)
                                    layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights] = 0.f;
                                else if (layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights] <
                                         -layers[layer]->stdp_shrestha_gerstner_weight_max)
                                    layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights] =
                                            -layers[layer]->stdp_shrestha_gerstner_weight_max;
                            }
                        }
                    }
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

        int channel_inp = 0;
        int channel_out = 0;
        if (layers[layer]->out_maps == 1) channel_inp = channel;
        else if (layers[layer]->kernel_channels == 1) channel_out = channel;

        int idx_channel = channel_out * layers[layer]->kernel_channels + channel_inp;
        layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = false;

        if (layers[layer]->d_d_kernels[kernel]->learning_trigger &&
            layers[layer]->d_d_kernels[kernel]->d_delay_active[delay]) {

            for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                    int idx_syn = cols * layers[layer]->rf_side + rows;
                    int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                            layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                    layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] = 0.f;
                    if (layers[layer]->synapse_inh_scaling > 0.f)
                        layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] = 0.f;
                }
            }

            int cnt_nodes = 0;
            for (int node = 0; node < layers[layer]->out_node_kernel; node++) {

                int idx_x_rf = node / layers[layer]->out_size[1];
                int idx_y_rf = node % layers[layer]->out_size[1];
                int idx_nodesep = channel_out * layers[layer]->out_node_kernel + node;

                if (!layers[layer]->d_d_kernels[kernel]->d_nodesep_train[idx_nodesep])
                    continue;

                layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = true;
                cnt_nodes++;

                for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                        int idx_xpad = idx_x_rf * layers[layer]->strides + cols;
                        int idx_ypad = idx_y_rf * layers[layer]->strides + rows;
                        int idx_nodepad = idx_xpad *
                                (layers[layer]->inp_size[1] + layers[layer]->padding_total[0]) + idx_ypad;
                        int idx_syn_inp = channel * layers[layer]->inp_node_kernel *
                                layers[layer]->num_delays + idx_nodepad * layers[layer]->num_delays + delay;

                        int idx_syn = cols * layers[layer]->rf_side + rows;
                        int idx_syn_weights = channel_inp * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                        int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;

                        float diff_weights;
                        int delta_T = layers[layer]->d_stdp_precnt[idx_syn_inp];

                        /* LTP */
                        if (delta_T >= 0 && delta_T < layers[layer]->stdp_shrestha_gerstner_window_LTP) {

                            // excitatory synapses
                            diff_weights = layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] -
                                    layers[layer]->synapse_w_init;
                            layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] +=
                                    layers[layer]->learning_rate * exp(-diff_weights);

                            // inhibitory synapses
                            if (layers[layer]->synapse_inh_scaling > 0.f) {
                                diff_weights = layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights] +
                                        layers[layer]->synapse_w_init;
                                layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] +=
                                        layers[layer]->learning_rate * exp(-diff_weights);
                            }

                        /* LTD */
                        } else {

                            // excitatory synapses
                            diff_weights = layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] -
                                    layers[layer]->synapse_w_init;
                            layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] -=
                                    layers[layer]->learning_rate * exp(diff_weights);

                            // inhibitory synapses
                            if (layers[layer]->synapse_inh_scaling > 0.f) {
                                diff_weights = layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights] +
                                        layers[layer]->synapse_w_init;
                                layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] -=
                                        layers[layer]->learning_rate * exp(diff_weights);
                            }
                        }

                        // reset counters
                        layers[layer]->d_stdp_precnt[idx_syn_inp] = -1;
                    }
                }
            }

            // average weight update
            if (cnt_nodes > 0) {
                for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                        int idx_syn = cols * layers[layer]->rf_side + rows;
                        int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                        layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] /= (float) cnt_nodes;
                        if (layers[layer]->synapse_inh_scaling > 0.f)
                            layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] /= (float) cnt_nodes;
                    }
                }
            }
        }
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

        int channel_inp = 0;
        int channel_out = 0;
        if (layers[layer]->out_maps == 1) channel_inp = channel;
        else if (layers[layer]->kernel_channels == 1) channel_out = channel;

        int idx_channel = channel_out * layers[layer]->kernel_channels + channel_inp;
        layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = false;

        if (layers[layer]->d_d_kernels[kernel]->d_delay_active[delay]) {

            for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                    int idx_syn = cols * layers[layer]->rf_side + rows;
                    int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                            layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                    layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] = 0.f;
                    if (layers[layer]->synapse_inh_scaling > 0.f)
                        layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] = 0.f;
                }
            }

            int cnt_nodes = 0;
            for (int node = 0; node < layers[layer]->out_node_kernel; node++) {

                bool node_contrib = false;
                int idx_x_rf = node / layers[layer]->out_size[1];
                int idx_y_rf = node % layers[layer]->out_size[1];
                int idx_nodesep = channel_out * layers[layer]->out_node_kernel + node;

                for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                        int idx_xpad = idx_x_rf * layers[layer]->strides + cols;
                        int idx_ypad = idx_y_rf * layers[layer]->strides + rows;
                        int idx_nodepad = idx_xpad *
                                (layers[layer]->inp_size[1] + layers[layer]->padding_total[0]) + idx_ypad;
                        int idx_syn_inp = channel * layers[layer]->inp_node_kernel *
                                layers[layer]->num_delays + idx_nodepad * layers[layer]->num_delays + delay;

                        if (layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep] == -1)
                            continue;

                        int idx_syn = cols * layers[layer]->rf_side + rows;
                        int idx_syn_weights = channel_inp * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                        int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;

                        int delta_T = layers[layer]->d_stdp_precnt[idx_syn_inp] -
                                layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep];

                        /* LTP */
                        if (delta_T >= 0 && delta_T <= layers[layer]->stdp_shrestha_gerstner_window_LTP &&
                            !layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep]) {

                            if (!node_contrib) {
                                cnt_nodes++;
                                node_contrib = true;
                                layers[layer]->d_d_kernels[kernel]->learning_trigger = true;
                                layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = true;
                            }

                            float update = layers[layer]->learning_rate *
                                    exp(-(float) delta_T / (float) layers[layer]->stdp_shrestha_gerstner_window_LTP);

                            // excitatory synapses
                            if (layers[layer]->stdp_gerstner_weight_dependence)
                                layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] +=
                                        update * (layers[layer]->stdp_shrestha_gerstner_weight_max -
                                        layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights]);
                            else layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] +=
                                    update;

                            // inhibitory synapses
                            if (layers[layer]->synapse_inh_scaling > 0.f) {
                                if (layers[layer]->stdp_gerstner_weight_dependence)
                                    layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] -=
                                            update * abs(-layers[layer]->stdp_shrestha_gerstner_weight_max -
                                            layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights]);
                                else layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] -= update;
                            }

                        /* LTD */
                        } else if (delta_T > layers[layer]->stdp_shrestha_gerstner_window_LTP &&
                                   !layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep]) {

                            if (!node_contrib) {
                                cnt_nodes++;
                                node_contrib = true;
                                layers[layer]->d_d_kernels[kernel]->learning_trigger = true;
                                layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = true;
                            }

                            float update = -layers[layer]->learning_rate;

                            // excitatory synapses
                            layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] += update;

                            // inhibitory synapses
                            if (layers[layer]->synapse_inh_scaling > 0.f)
                                layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] -= update;

                        /* LTD */
                        } else if (delta_T < 0 &&
                                   !layers[layer]->d_stdp_precnt[idx_syn_inp]) {

                            if (!node_contrib) {
                                cnt_nodes++;
                                node_contrib = true;
                                layers[layer]->d_d_kernels[kernel]->learning_trigger = true;
                                layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = true;
                            }

                            float update = -layers[layer]->learning_rate *
                                    exp((float) delta_T / (float) layers[layer]->stdp_shrestha_gerstner_window_LTD);

                            // excitatory synapses
                            if (layers[layer]->stdp_gerstner_weight_dependence)
                                layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] +=
                                        update * layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights];
                            else layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] += update;

                            // inhibitory synapses
                            if (layers[layer]->synapse_inh_scaling > 0.f) {
                                if (layers[layer]->stdp_gerstner_weight_dependence)
                                    layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] -= update *
                                            abs(layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights]);
                                else layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] -= update;
                            }
                        }
                    }
                }
            }

            // average weight update
            if (cnt_nodes > 0) {
                for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                        int idx_syn = cols * layers[layer]->rf_side + rows;
                        int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                        layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] /= (float) cnt_nodes;
                        if (layers[layer]->synapse_inh_scaling > 0.f)
                            layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] /= (float) cnt_nodes;
                    }
                }
            }
        }
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

        int channel_inp = 0;
        int channel_out = 0;
        if (layers[layer]->out_maps == 1) channel_inp = channel;
        else if (layers[layer]->kernel_channels == 1) channel_out = channel;

        int idx_channel = channel_out * layers[layer]->kernel_channels + channel_inp;
        layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = false;

        if (layers[layer]->d_d_kernels[kernel]->d_delay_active[delay]) {

            for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                    int idx_syn = cols * layers[layer]->rf_side + rows;
                    int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                            layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                    layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] = 0.f;
                    if (layers[layer]->synapse_inh_scaling > 0.f)
                        layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] = 0.f;
                }
            }

            int cnt_nodes = 0;
            for (int node = 0; node < layers[layer]->out_node_kernel; node++) {

                bool node_contrib = false;
                int idx_x_rf = node / layers[layer]->out_size[1];
                int idx_y_rf = node % layers[layer]->out_size[1];
                int idx_nodesep = channel_out * layers[layer]->out_node_kernel + node;

                for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                        int idx_xpad = idx_x_rf * layers[layer]->strides + cols;
                        int idx_ypad = idx_y_rf * layers[layer]->strides + rows;
                        int idx_nodepad = idx_xpad *
                                (layers[layer]->inp_size[1] + layers[layer]->padding_total[0]) + idx_ypad;
                        int idx_syn_inp = channel * layers[layer]->inp_node_kernel *
                                layers[layer]->num_delays + idx_nodepad * layers[layer]->num_delays + delay;

                        if (layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep] == -1)
                            continue;

                        int idx_syn = cols * layers[layer]->rf_side + rows;
                        int idx_syn_weights = channel_inp * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                        int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;

                        int delta_T = layers[layer]->d_stdp_precnt[idx_syn_inp] -
                                layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep];

                        /* LTP */
                        if (delta_T >= 0 &&
                            !layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep]) {

                            if (!node_contrib) {
                                cnt_nodes++;
                                node_contrib = true;
                                layers[layer]->d_d_kernels[kernel]->learning_trigger = true;
                                layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = true;
                            }

                            // excitatory synapses
                            layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] +=
                                    layers[layer]->learning_rate *
                                    layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] *
                                    (1.f - layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights]);

                            // inhibitory synapses
                            if (layers[layer]->synapse_inh_scaling > 0.f)
                                layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] -=
                                        layers[layer]->learning_rate *
                                        abs(layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights]) *
                                        (1.f - abs(layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights]));

                        /* LTD */
                        } else if (delta_T < 0) {

                            if (!node_contrib) {
                                cnt_nodes++;
                                node_contrib = true;
                                layers[layer]->d_d_kernels[kernel]->learning_trigger = true;
                                layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps_channels[idx_channel] = true;
                            }

                            // excitatory synapses
                            layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] -=
                                    layers[layer]->learning_rate *
                                    layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] *
                                    (1.f - layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights]);

                            // inhibitory synapses
                            if (layers[layer]->synapse_inh_scaling > 0.f)
                                layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] +=
                                        layers[layer]->learning_rate *
                                        abs(layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights]) *
                                        (1.f - abs(layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights]));
                        }
                    }
                }
            }

            // average weight update
            if (cnt_nodes > 0) {
                for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                        int idx_syn = cols * layers[layer]->rf_side + rows;
                        int idx_syn_delta = channel * layers[layer]->rf_side * layers[layer]->rf_side *
                                            layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + delay;
                        layers[layer]->d_d_kernels[kernel]->d_weights_exc_delta[idx_syn_delta] /= (float) cnt_nodes;
                        if (layers[layer]->synapse_inh_scaling > 0.f)
                            layers[layer]->d_d_kernels[kernel]->d_weights_inh_delta[idx_syn_delta] /= (float) cnt_nodes;
                    }
                }
            }
        }
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

            int delay_max_weights = -1;
            float max_weights = 0.f;
            for (int d = 0; d < layers[layer]->d_d_kernels[kernel]->num_delays_active; d++) {
                layers[layer]->d_d_kernels[kernel]->d_sum_exc_weights[d] = 0.f;
                for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                        for (int ch = 0; ch < layers[layer]->kernel_channels; ch++) {
                            int idx_syn = cols * layers[layer]->rf_side + rows;
                            int idx_syn_weights = ch * layers[layer]->rf_side * layers[layer]->rf_side *
                                    layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + d;
                            layers[layer]->d_d_kernels[kernel]->d_sum_exc_weights[d] +=
                                    layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights];
                        }
                    }
                }
                if (layers[layer]->d_d_kernels[kernel]->d_sum_exc_weights[d] >= max_weights) {
                    delay_max_weights = d;
                    max_weights = layers[layer]->d_d_kernels[kernel]->d_sum_exc_weights[d];
                }
            }

            bool drop_delays = false;
            for (int d = delay_max_weights + 1; d < layers[layer]->d_d_kernels[kernel]->num_delays_active; d++) {
                if (drop_delays ||
                    layers[layer]->d_d_kernels[kernel]->d_sum_exc_weights[d] <
                    layers[layer]->d_d_kernels[kernel]->d_sum_exc_weights[delay_max_weights] * drop_delays_th[0]) {
                    layers[layer]->d_d_kernels[kernel]->d_delay_active[d] = false;
                    drop_delays = true;
                }
            }

            layers[layer]->d_d_kernels[kernel]->num_delays_active = 0;
            for (int d = 0; d < layers[layer]->num_delays; d++) {
                if (layers[layer]->d_d_kernels[kernel]->d_delay_active[d])
                    layers[layer]->d_d_kernels[kernel]->num_delays_active++;
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

        layers[layer]->d_d_kernels[kernel]->d_max_channel[node] = -1;
        if (layers[layer]->firing_node) {

            float V_max = 0.f;
            for (int ch = 0; ch < layers[layer]->out_maps; ch++) {
                int idx_nodesep = ch * layers[layer]->out_node_kernel + node;
                if (layers[layer]->d_d_kernels[kernel]->d_nodesep_train[idx_nodesep] &&
                    layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep] > V_max) {
                    V_max = layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep];
                    layers[layer]->d_d_kernels[kernel]->d_max_channel[node] = ch;
                }
            }
        }
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

        layers[layer]->d_max_kernel[node] = -1;
        if (layers[layer]->firing_node) {

            float V_max = 0.f;
            for (int k = 0; k < layers[layer]->cnt_kernels; k++) {
                if (layers[layer]->d_d_kernels[k]->d_max_channel[node] != -1) {
                    int idx_nodesep = layers[layer]->d_d_kernels[k]->d_max_channel[node] *
                            layers[layer]->out_node_kernel + node;
                    if (layers[layer]->d_d_kernels[k]->d_nodesep_V[idx_nodesep] > V_max) {
                        V_max = layers[layer]->d_d_kernels[k]->d_nodesep_V[idx_nodesep];
                        layers[layer]->d_max_kernel[node] = k;
                    }
                }
            }
        }
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
        if (kernel_max != -1) {
            int channel_max = layers[layer]->d_d_kernels[kernel_max]->d_max_channel[node];
            if (layers[layer]->learning) layers[layer]->d_d_kernels[kernel_max]->learning_trigger = true;
            if (kernel != kernel_max || (kernel == kernel_max && channel != channel_max)) {
                int idx_nodesep = channel * layers[layer]->out_node_kernel + node;
                layers[layer]->d_d_kernels[kernel]->d_nodesep_train[idx_nodesep] = 0;
                layers[layer]->d_d_kernels[kernel]->d_nodesep_V[idx_nodesep] = 0.f;
                layers[layer]->d_d_kernels[kernel]->d_nodesep_refrac[idx_nodesep] = 0.f;

                // update counters for Gerstner's and Kheradpisheh's STDP
                if (layers[layer]->learning_type == 3 ||
                    layers[layer]->learning_type == 4)
                    layers[layer]->d_d_kernels[kernel]->d_stdp_postcnt[idx_nodesep] = -1;
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

        // max weights in a kernel
        float max_weight_exc = 0.f;
        float max_weight_inh = 0.f;
        for (int ch = 0; ch < layers[layer]->kernel_channels; ch++) {
            for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                    for (int d = 0; d < layers[layer]->d_d_kernels[kernel]->num_delays_active; d++) {
                        int idx_syn = cols * layers[layer]->rf_side + rows;
                        int idx_syn_weights = ch * layers[layer]->rf_side * layers[layer]->rf_side *
                                layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + d;
                        if (max_weight_exc < layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights])
                            max_weight_exc = layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights];
                        if (layers[layer]->synapse_inh_scaling > 0.f &&
                            max_weight_inh < layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights] +
                            2.f * layers[layer]->synapse_w_init)
                            max_weight_inh = layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights] +
                                    2.f * layers[layer]->synapse_w_init;
                    }
                }
            }
        }

        // objective function computation (MSE trace-weight)
        int stdp_objective_cnt = 0;
        layers[layer]->d_d_kernels[kernel]->d_stdp_paredes_objective_avg[channel] = 0.f;
        for (int node = 0; node < layers[layer]->out_node_kernel; node++) {

            int idx_nodesep = channel * layers[layer]->out_node_kernel + node;
            if (layers[layer]->d_d_kernels[kernel]->d_nodesep_train[idx_nodesep]) {
                layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps[channel] = true;
                int idx_x_rf = node / layers[layer]->out_size[1];
                int idx_y_rf = node % layers[layer]->out_size[1];

                for (int ch = 0; ch < layers[layer]->kernel_channels; ch++) {
                    for (int rows = 0; rows < layers[layer]->rf_side - layers[layer]->rf_side_limits[0]; rows++) {
                        for (int cols = 0; cols < layers[layer]->rf_side - layers[layer]->rf_side_limits[1]; cols++) {
                            int idx_syn = cols * layers[layer]->rf_side + rows;
                            int idx_xpad = idx_x_rf * layers[layer]->strides + cols;
                            int idx_ypad = idx_y_rf * layers[layer]->strides + rows;
                            int idx_nodepad = idx_xpad *
                                    (layers[layer]->inp_size[1] + layers[layer]->padding_total[0]) + idx_ypad;

                            for (int d = 0; d < layers[layer]->d_d_kernels[kernel]->num_delays_active; d++) {
                                int idx_syn_weights = ch * layers[layer]->rf_side * layers[layer]->rf_side *
                                        layers[layer]->num_delays + idx_syn * layers[layer]->num_delays + d;
                                int idx_syn_inp = (ch + channel) * layers[layer]->inp_node_kernel *
                                        layers[layer]->num_delays + idx_nodepad * layers[layer]->num_delays + d;

                                // excitatory synapses
                                layers[layer]->d_d_kernels[kernel]->d_stdp_paredes_objective_avg[channel] +=
                                        pow(layers[layer]->d_synapse_pretrace[idx_syn_inp] /
                                            layers[layer]->d_d_kernels[kernel]->d_nodesep_maxpretrace[idx_nodesep] -
                                            layers[layer]->d_d_kernels[kernel]->d_weights_exc[idx_syn_weights] /
                                            max_weight_exc, 2.f);

                                // inhibitory synapses
                                if (layers[layer]->synapse_inh_scaling > 0.f)
                                    layers[layer]->d_d_kernels[kernel]->d_stdp_paredes_objective_avg[channel] +=
                                            pow(layers[layer]->d_synapse_pretrace[idx_syn_inp] /
                                                layers[layer]->d_d_kernels[kernel]->d_nodesep_maxpretrace[idx_nodesep] -
                                                (layers[layer]->d_d_kernels[kernel]->d_weights_inh[idx_syn_weights] +
                                                 2.f * layers[layer]->synapse_w_init) / max_weight_inh, 2.f);
                            }
                        }
                    }
                }
                stdp_objective_cnt++;
            }
        }

        // update convergence data
        if (stdp_objective_cnt > 0) {
            layers[layer]->d_d_kernels[kernel]->d_stdp_paredes_objective_avg[channel] /=
                    (float) (stdp_objective_cnt * layers[layer]->rf_side * layers[layer]->rf_side *
                    layers[layer]->d_d_kernels[kernel]->num_delays_active);
            if (layers[layer]->synapse_inh_scaling > 0.f)
                layers[layer]->d_d_kernels[kernel]->d_stdp_paredes_objective_avg[channel] /= 2.f;
        }
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

        // compute objective function
        float accum = 0.f;
        int stdp_objective_cnt = 0;
        for (int ch = 0; ch < layers[layer]->out_maps; ch++) {
            if (layers[layer]->d_d_kernels[kernel]->d_weights_delta_maps[ch]) {
                accum += layers[layer]->d_d_kernels[kernel]->d_stdp_paredes_objective_avg[ch];
                stdp_objective_cnt++;
            }
        }

        // compute moving average of objective function
        if (stdp_objective_cnt > 0) {

            for (int i = layers[layer]->stdp_paredes_stats_window; i > 0; i--)
                layers[layer]->d_d_kernels[kernel]->d_stdp_paredes_objective[i] =
                        layers[layer]->d_d_kernels[kernel]->d_stdp_paredes_objective[i-1];
            layers[layer]->d_d_kernels[kernel]->d_stdp_paredes_objective[0] = accum / (float) stdp_objective_cnt;

            layers[layer]->d_d_kernels[kernel]->stdp_paredes_objective_avg = 0.f;
            for (int i = 0; i < layers[layer]->stdp_paredes_stats_window; i++)
                layers[layer]->d_d_kernels[kernel]->stdp_paredes_objective_avg +=
                        layers[layer]->d_d_kernels[kernel]->d_stdp_paredes_objective[i];
            layers[layer]->d_d_kernels[kernel]->stdp_paredes_objective_avg /=
                    (float) layers[layer]->stdp_paredes_stats_window;

            if (layers[layer]->d_d_kernels[kernel]->stdp_paredes_objective_avg < layers[layer]->stdp_paredes_convg_th)
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
__global__ void update_output(Layer **layers, float *sim_step) {

    int layer = blockIdx.x;
    int node = blockIdx.y;
    int kernel = threadIdx.x;

    if (layers[layer]->active &&
        layers[layer]->cnt_kernels > kernel &&
        layers[layer]->out_node_kernel > node) {

        layers[layer]->firing_node = false;

        int begin_vector = node * layers[layer]->length_delay_out;
        int end_vector = (node + 1) * layers[layer]->length_delay_out;
        for (int i = end_vector-1; i > begin_vector; i--)
            layers[layer]->d_d_kernels[kernel]->d_node_train[i] =
                    layers[layer]->d_d_kernels[kernel]->d_node_train[i-1];
        layers[layer]->d_d_kernels[kernel]->d_node_train[begin_vector] = 0;

        for (int ch = 0; ch < layers[layer]->out_maps; ch++) {
            int idx_nodesep = ch * layers[layer]->out_node_kernel + node;
            if (layers[layer]->d_d_kernels[kernel]->d_nodesep_train[idx_nodesep]) {
                layers[layer]->d_d_kernels[kernel]->d_node_train[begin_vector] = 1;
                layers[layer]->d_d_kernels[kernel]->d_nodesep_train[idx_nodesep] = 0;
            }
        }

        layers[layer]->d_d_kernels[kernel]->d_node_posttrace[node] +=
                (sim_step[0]/layers[layer]->decay) *
                (-layers[layer]->d_d_kernels[kernel]->d_node_posttrace[node] +
                 (float) layers[layer]->d_d_kernels[kernel]->d_node_train[begin_vector]);
        if (layers[layer]->d_d_kernels[kernel]->d_node_posttrace[node] < 0.f)
            layers[layer]->d_d_kernels[kernel]->d_node_posttrace[node] = 0.f;
    }
}


// limit the number of STDP updates
__global__ void learning_limit_updates(Layer **layers) {

    int layer = blockIdx.x;

    if (layers[layer]->active &&
        layers[layer]->learning &&
        layers[layer]->learning_type) {

        bool spike = false;
        for (int k = 0; k < layers[layer]->cnt_kernels; k++) {
            for (int i = 0; i < layers[layer]->out_node_kernel; i++) {
                if (layers[layer]->d_d_kernels[k]->d_node_train[i * layers[layer]->length_delay_out]) {
                    layers[layer]->learning_updates_cnt++;
                    spike = true;
                    break;
                }
                if (spike) break;
            }
        }

        if (layers[layer]->learning_updates_cnt > layers[layer]->learning_limit_updates &&
            layers[layer]->learning_limit_updates > 0) {
            layers[layer]->learning_updates_cnt = 0;
            layers[layer]->learning = false;
            layers[layer]->limit_learning = true;
        }
    }
}