#ifndef H_CUSNN_KERNELS
#define H_CUSNN_KERNELS


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


#include <host_defines.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>


__global__ void update_input_trains(int *inputs, int *input_size, int *delay);
__global__ void enable_learning(Layer **layers);
__global__ void update_input_channels(Layer **layers, float *sim_step, int *inputs);
__global__ void propagation(Layer **layers, int *inputs);
__global__ void add_input(Layer **layers);
__global__ void update_V(Layer **layers, float *sim_step, float *refrac);
__global__ void spatial_firing_node_kernel_channel(Layer **layers);
__global__ void spatial_firing_node_kernel(Layer **layers);
__global__ void spatial_firing_node(Layer **layers);
__global__ void spatial_perpendicular_inhibition(Layer **layers);
__global__ void stdp_paredes_kernel_channel(Layer **layers);
__global__ void stdp_shrestha_kernel_channel(Layer **layers);
__global__ void stdp_gerstner_kernel_channel(Layer **layers);
__global__ void stdp_kheradpisheh_kernel_channel(Layer **layers);
__global__ void learning_update_weights(Layer **layers);
__global__ void drop_delays_kernel(Layer **layers, float *drop_delays_th);
__global__ void firing_node_kernel(Layer **layers);
__global__ void firing_node(Layer **layers);
__global__ void perpendicular_inhibition(Layer **layers);
__global__ void stdp_paredes_track_convergence_channel(Layer **layers);
__global__ void stdp_paredes_track_convergence(Layer **layers);
__global__ void update_output_channels(Layer **layers);
__global__ void update_output(Layer **layers, float *sim_step);
__global__ void learning_limit_updates(Layer **layers);


#endif