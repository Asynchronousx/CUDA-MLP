## CUDA-MLP
This project demonstrates a Multilayer Perceptron (MLP) implementation using C++ and CUDA, designed for academic purposes. By leveraging the parallel computing capabilities of CUDA, this MLP efficiently trains and evaluates using forward and backward propagation algorithms.

## Project Scope
The main goal of this project is to provide a simple yet effective implementation of an MLP to model and fit a trigonometric function. The dataset consists of a linear space of points, which is used for both training and testing the network.

## Performance 
The models using CUDA presents fairly advantages with the increase of the layer of the network, thus the heavy algebric operation needed to perform the operations. <br>
The function is fitted nicely using almost every configuration of layers both with the CUDA and the simple CPU one. Below some graphs that shows the performance and the 
time needed for the model. The configuration used is composed by the following dense layer: `[1,512,1]`

<div style="display: flex; justify-content: space-between;">
  <img src="images/fit.png" width="380"/>
  <img src="images/train_time.png" width="380"/>
</div>


## Comparison
How does the models compare? Simply put, using CUDA does make sense when operating with huge layers and thus huge algebric operation of various sorts. 
That because using CUDA comes with a cost, namely that of CUDA operations (such as CUDA Malloc, CUDA Memcpy, CUDA Launch and so on) performed on data. 
It does have sense then using such technology within the scope of large matrices and operations (such as images), otherwise the CPU it's just better for simple problems (and faster). 
In the image below, a simple comparison between a model composed of `[1,8,8,8,1]` neuron layers and a `[1,512,512,512,1]` one.

<div style="display: flex; justify-content: space-between;">
  <img src="images/comparison_8hu.png" width="380"/>
  <img src="images/comparison_512hu.png" width="380"/>
</div>
