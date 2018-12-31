#ifndef H_CUSNN
#define H_CUSNN


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


#include <stdio.h>
#include <string>
#include <vector>
#include <vector_types.h>


/* KERNEL CLASS */
class Kernel {
    public:

        // output map data
        int *h_node_train;
        float *h_node_posttrace;
        int *d_node_train;
        float *d_node_posttrace;

        int *h_nodesep_train;
        float *h_nodesep_V;
        float *h_nodesep_refrac;
        int *d_nodesep_train;
        int *d_nodesep_perpendicular;
        float *d_nodesep_V;
        float *d_nodesep_input;
        float *d_nodesep_channel_input;
        float *d_nodesep_refrac;
        float *d_nodesep_pretrace;
        float *d_nodesep_maxpretrace;

        // delay flags
        bool *h_delay_active;
        bool *d_delay_active;
        float *d_sum_exc_weights;
        int num_delays_active;

        // inhibition params
        int channel_max;
        int node_max;
        int *d_node_max;
        int *d_max_channel;
        float V_max;
        float *d_V_max;

        // excitatory weights
        float *h_weights_exc;
        float *d_weights_exc;
        float *d_weights_exc_delta;

        // inhibitory weights
        float *h_weights_inh;
        float *d_weights_inh;
        float *d_weights_inh_delta;

        // learning params
        bool learning_trigger;
        bool *d_weights_delta_maps;
        bool *d_weights_delta_maps_channels;

        // STDP params
        int *h_stdp_postcnt;
        int *d_stdp_postcnt;

        // Paredes' STDP params
        float stdp_paredes_objective_avg;
        float *h_stdp_paredes_objective;
        float *d_stdp_paredes_objective;
        float *d_stdp_paredes_objective_avg;

        // Diehl's adaptive threshold
        float *h_threshold_diehl_nodesep_theta;
        float *d_threshold_diehl_nodesep_theta;

        /* FUNCTIONS */
        Kernel(int out_node_kernel, int out_nodesep_kernel, int out_maps, int length_delay_out, float node_refrac,
               int rf_side, int num_delays, float w_init, int kernel_channels, float sim_step);
        ~Kernel();
};


/* LAYER CLASS */
class Layer {
    public:

        // layer type
        int layer_type;
        std::string layer_type_str;

        // kernel counter
        int cnt_kernels;

        // params
        bool load_weights;
        bool homeostasis;
        bool active;
        bool firing_node;
        int num_delays;
        float synapse_inh_scaling;
        int rf_side;
        int rf_side_limits[2];
        int strides;
        int padding[2];
        int padding_total[2];
        int out_channels;
        int out_maps;
        int kernel_channels;
        float node_Vth;
        float decay;
        float alpha;
        float max_delay;
        float synapse_w_init;
        std::string padding_type;

        // synaptic delay
        int length_delay_inp;
        int length_delay_out;
        int *h_delay_indices;
        int *d_delay_indices;

        // input and output sizes
        int inp_size[3];
        int out_size[3];
        int inp_node_kernel;
        int inp_node_total;
        int inp_synapses_total;
        int out_node_kernel;
        int out_nodesep_kernel;

        // perpendicular inhibition
        int kernel_max;
        int neigh_inh;
        int *d_max_kernel;
        bool inhibition;

        // pre-synaptic trace
        float *h_synapse_pretrace;
        float *d_synapse_pretrace;

        // learning params
        bool learning;
        bool limit_learning;
        bool enable_learning;
        bool inhibition_spatial;
        bool *h_kernels_cnvg;
        bool *d_kernels_cnvg;
        int learning_type;
        int enable_learning_cnt;
        int learning_warmup_time;
        int learning_updates_cnt;
        int learning_limit_updates;
        float learning_rate;

        // STDP params
        int *h_stdp_precnt;
        int *d_stdp_precnt;

        // Paredes' STDP params
        int stdp_paredes_stats_window;
        float stdp_paredes_convg_th;
        float stdp_paredes_a;

        // Shrestha's and Gerstner's STDP params
        bool stdp_shrestha_gerstner_weight_boundaries;
        int stdp_shrestha_gerstner_window_LTP;
        int stdp_shrestha_gerstner_window_LTD;
        float stdp_shrestha_gerstner_weight_max;

        // Gerstner's STDP params
        bool stdp_gerstner_weight_dependence;

        // Diehl's adaptive threshold
        bool threshold_diehl;
        float threshold_diehl_increase;

        // layers
        Kernel** h_kernels;
        Kernel** h_d_kernels;
        Kernel** d_d_kernels = NULL;

        /* FUNCTIONS */
        Layer(std::string layer_type, bool learning, bool load_weights, bool homeostasis, float Vth, float decay,
              float alpha, float max_delay, int num_delays, float synapse_inh_scaling, int rf_side, int out_channels,
              std::string padding, float w_init, float sim_step);
        ~Layer();

        void add_kernel(int out_node_kernel, int out_nodesep_kernel, int out_maps, int length_delay_out,
                        float node_refrac, int rf_side, int num_delays, float w_init, int kernel_channels,
                        float sim_step);
};


/* NETWORK CLASS */
class Network {
    public:

        // number of layers
        int cnt_layers;

        // input size
        int *h_inp_size;
        int *d_inp_size;
        float inp_scale[2];

        // input data
        int *h_inputs;
        int *d_inputs;

        // simulation timestep
        float *h_sim_step;
        float *d_sim_step;

        // neuron and synapse params
        int *h_length_delay_inp;
        int *d_length_delay_inp;
        float *h_node_refrac;
        float *d_node_refrac;
        float *h_synapse_trace_init;
        bool inhibition;

        // drop-delays mechanism
        bool drop_delays;
        float *h_drop_delays_th;
        float *d_drop_delays_th;

        // learning params
        bool learning;
        int learning_type;

        // CUDA blocks and threads dimensions
        int max_inputs;
        int max_channels;
        int max_kernels;
        int max_outputs;
        int max_delays;
        dim3 block_0;
        dim3 block_1;
        dim3 block_2;
        dim3 block_3;
        dim3 block_4;
        dim3 block_5;
        dim3 block_6;
        dim3 thread_0;
        dim3 thread_1;

        // layers
        Layer** h_layers;
        Layer** h_d_layers;
        Layer** d_d_layers = NULL;

        /* FUNCTIONS */
        Network(const int inp_size[3], const float inp_scale[2], const float sim_step, float node_refrac,
                float synapse_trace_init, bool inhibition, bool drop_delays, float drop_delays_th);
        ~Network();

        void add_layer(std::string layer_type, bool learning, bool load_weights, bool homeostasis, float Vth,
                       float decay, float alpha, float max_delay = 1.f, int num_delays = 1,
                       float synapse_inh_scaling = 0.f, int rf_side = 7, int out_channels = 8,
                       std::string padding = "none", float w_init = 0.5f);
        void create_network(bool& break_fun);
        void enable_adaptive_threshold_diehl(float threshold_delta, bool& break_fun);
        void enable_stdp_paredes(float learning_rate, float scale_a, float convg_th, int limit_updates,
                                 bool inhibition_spatial, int stats_window, bool& break_fun);
        void enable_stdp_shrestha(float learning_rate, float window_LTP, bool weight_boundaries, float weight_max,
                                  int limit_updates, bool inhibition_spatial, bool& break_fun);
        void enable_stdp_gerstner(float learning_rate_exc, float window_LTP, float window_LTD, bool weight_dependence,
                                  bool weight_boundaries, float weight_max, int limit_updates, bool inhibition_spatial,
                                  bool& break_fun);
        void enable_stdp_kheradpisheh(float learning_rate_exc, int limit_updates, bool inhibition_spatial,
                                      bool& break_fun);

        void modify_conv2dsep(int l);
        void modify_dense(int l);
        void modify_pooling(int l);
        void modify_merge(int l);
        void modify_kernels_pooling(int l);
        void modify_kernels_merge(int l);

        void feed(bool& break_fun);
        void update_input();
        void copy_to_host();
        void summary();
        void init();
        void weights_to_device();
};


#endif
