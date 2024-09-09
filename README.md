## CUDA-MLP
A simple Multilayer Perceptron implementation using C++ and CUDA for academical purposes. <br> 
This project leverages the power of CUDA to train a Multi Layer Perceptron (MLP) using the forward and backward algorithms.
The scope of this project is fairly simple: it is a small, efficient implementation of an MLP 
for the task of fitting a trigonometric function using a linear space of points as the main train/test dataset <br><br>
Note that, this repository contain both the source code for the CUDA and plain CPU Multi Layer Perceptron.

## Performance 
The models using CUDA presents fairly advantages with the increase of the layer of the network, thus the heavy algebric operation needed to perform the operations. <br>
The function is fitted nicely using almost every configuration of layers both with the CUDA and the simple CPU one. Below some graphs that shows the performance and the 
time needed for the model. The configuration used is composed by the following dense layer: `[1,512,1]`

<div style="display: flex; justify-content: space-between;">
  <img src="images/fit.png" width="350"/>
  <img src="images/train_time.png" width="350"/>
</div>


## Comparison
How does the models compare? Simply put, using CUDA does make sense when operating with huge layers and thus huge algebric operation of various sorts. 
That because using CUDA comes with a cost, namely that of CUDA operations (such as CUDA Malloc, CUDA Memcpy, CUDA Launch and so on) performed on data. 
It does have sense then using such technology within the scope of large matrices and operations (such as images), otherwise the CPU it's just better for simple problems (and faster). 
In the image below, a simple comparison between a model composed of `[1,8,8,8,1]` neuron layers and a `[1,512,512,512,1]` one.
