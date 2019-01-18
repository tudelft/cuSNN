# cuSNN
cuSNN is a C++ library that enables GPU-accelerated simulations of large-scale Spiking Neural Networks (SNNs).

* [More about cuSNN](#more-about-cusnn)
  * [Layer Types](#layer-types)
  * [Neuron Models](#neuron-models)
  * [Synapse Models](#synapse-models)
  * [Learning Rules](#learning-rules)
* [Installation](#installation)
* [cuSNN samples](#cusnn-samples)
* [The Team](#the-team)

This project was created for the work ["*Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow 
Estimation: From Events to Global Motion Perception*" (Paredes-Vallés, F., Scheper, K.Y., and de Croon, G.C.H.E., 2018)](https://arxiv.org/abs/1807.10936).

If you use this library in an academic publication, please cite our work:

```
@article{paredes2018unsupervised,
  title={Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception},
  author={Paredes-Vall{\'e}s, Federico and Scheper, Kirk YW and de Croon, Guido CHE},
  url={https://arxiv.org/abs/1807.10936},
  year={2018}
}
```

##  More about cuSNN
cuSNN is a library that consists of the following files:

| File                  	| Description 	|
|----------------------------	|-------------	|
| **cusnn.cu/.cuh**         	|  CUDA GPU engine           	|
| **cusnn_kernels.cu/.cuh** 	|  CUDA device functions for GPU engine           	|

In cuSNN, a SNN is defined using the following host/device classes:

| Classes                  	| Content 	|
|----------------------------	|-------------	|
| **Network**         	|  *Layer* objects and architecture parameters.      	|
| Network -> **Layer** 	|  *Kernel* object and layer parameters.        	|
| Network -> Layer -> **Kernel**    |  Neural and synaptic parameters. 	|

The cuSNN library has been tested on Linux systems only.

#### Layer Types

* **Conv2d**: 2D convolution over an input neural map composed of several input channels.
* **Conv2dSep**: 2D separable convolution over an input neural map composed of several input channels.
* **Pooling**: 2D pooling over an input neural map composed of several input channels.
* **Merge**: 1x1 convolution with unitary weights over an input neural map composed of several input channels.
* **Dense**: Full connectivity to an input neural map composed of several input channels.

#### Synapse and Neuron Models

At the moment, only the models proposed in our work are implemented. These are:

* Leaky Integrate-and-Fire (LIF) neuron model
* Trace-based adaptive LIF neuron model
* Static synapse model 

#### Learning Rules

At the moment, only the following learning rules are implemented:

* Unsupervised Learning:
    * [Our trace-based STDP formulation](https://arxiv.org/abs/1807.10936)
    * [Gerstner's STDP formulation](https://www.emeraldinsight.com/doi/full/10.1108/k.2003.06732gae.003)
    * [Kheradpisheh *et al.* STDP formulation](https://www.sciencedirect.com/science/article/pii/S0893608017302903)
    * [Shrestha *et al.* STDP formulation](https://ieeexplore.ieee.org/abstract/document/7966096)

##  Installation

**Requirements**:

* [**CMake**](https://cmake.org/) **3.8** (or higher)
* [**NVIDIA CUDA Toolkit**](https://developer.nvidia.com/cuda-toolkit) **8.0** (or higher)

**Get the cuSNN source**:
```
git clone https://github.com/fedepare/cuSNN.git
cd cuSNN
```

**Install cuSNN**:

Run the following Python script to install the cuSNN library:

```
python setup.py <install_dir>   # Python 2.*
python3 setup.py <install_dir>  # Python 3.*
```

By default, cuSNN installation directory is `/usr/local`. Hence, administrative privileges may be required. A different 
installation directory can be specified using the `<install_dir>` argument, as indicated above.

To uninstall all files, run:

```
xargs rm < build/install_manifest.txt
```

Again, administrative privileges may be required depending on your instllation directory.

##  cuSNN samples

Several samples are available to demonstrate the main features provided by the cuSNN library. These are stored in the
submodule [**cuSNN-samples**](https://github.com/fedepare/cuSNN-samples). To incorporate them to your current directory, 
run:

```
git submodule update --init --recursive
```

P.S.: This process can be a bit slow since a small dataset is included.

#### Building cuSNN samples

In Linux, the samples are built using makefiles. For this, go to the sample directory you wish and run:

```
make clean
make
```

In the makefile of each sample, there are compilation flags to adapt the simulation to your needs. Depending on their 
value, some of the following libraries may be required:

* PLOTTER: **OpenGL** and **FreeGLUT** for visualization
* SNAPSHOT: [**cnpy library**](https://github.com/rogersce/cnpy "cnpy library (C++ arrays to Numpy)") for converting C++ arrays to
 Numpy
 
In case your installation directory differs from `/usr/local`, `-lcuSNN` needs to be removed from 
the definition of the `LIBSUSR*` compilation flags, and `LIBSCUSTOM` needs to be defined as follows:

```
LIBSCUSTOM = -I<install_dir>/include -L<install_dir>/lib -lcuSNN
```

Further, if your installation directory differs from `/usr/local`, you may need to include the installation 
directory to your `LD_LIBRARY_PATH` environment variable so the executable can find the library at runtime. You can do 
this by adding the following line to your `~/.bashrc` file:

```
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:<install_dir>/lib"
export LD_LIBRARY_PATH
```
or by running the following command in every new terminal window in which a cuSNN sample wants to be run:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<install_dir>/lib
```
 

#### Running cuSNN samples

Once the sample of your choice is built as explained above, simply run the executable as follows:

```
./build/main
```

## The Team

cuSNN is currently maintained by [Fede Paredes-Vallés](https://github.com/fedepare), 
[Kirk Scheper](https://github.com/kirkscheper), and [Guido de Croon](https://github.com/guidoAI).
