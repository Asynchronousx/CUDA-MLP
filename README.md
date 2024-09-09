## CUDA-MLP
This project demonstrates a Multilayer Perceptron (MLP) implementation using C++ and CUDA, designed for academic purposes. By leveraging the parallel computing capabilities of CUDA, this MLP efficiently trains and evaluates using forward and backward propagation algorithms.

## Project Scope
The main goal of this project is to provide a simple yet effective implementation of an MLP to model and fit various trigonometric function. The dataset consists of a linear space of points, which is used for both training and testing the network.

## Performance 
he models using CUDA demonstrate significant advantages in computation times as the number of layers in the network increases, due to the heavy algebraic operations required.
The function used to measure the network's performance is `sin^⁡2`, and it is effectively fitted with nearly every configuration of layers, both with the CUDA-accelerated version and the standard CPU implementation. Below are some graphs illustrating the model's performance and the computation time. The network configuration used consists of the following dense layers: `[1, 512, 1]`.

<div style="display: flex; justify-content: space-between;">
  <img src="images/fit.png" width="380"/>
  <img src="images/train_time.png" width="380"/>
</div>

## Comparison
How does the CUDA/CPU models compare? Simply put, using CUDA does make sense when operating with huge layers and thus huge algebric operation of various sorts. 
That because using CUDA comes with a cost, namely that of CUDA operations (such as CUDA Malloc, CUDA Memcpy, CUDA Launch and so on) performed on data. 
It does have sense then using such technology within the scope of large matrices and operations (such as images), otherwise the CPU it's just better for simple problems (and faster). 
In the image below, a simple comparison between a model composed of `[1,8,8,8,1]` neuron layers and a `[1,512,512,512,1]` one.

<div style="display: flex; justify-content: space-between;">
  <img src="images/comparison_8hu.png" width="380"/>
  <img src="images/comparison_512hu.png" width="380"/>
</div>

## Usage
Training, evaluating and plotting the results of the network is super simple. The entire network was written with efficiency and readability in mind and it's less than 150 lines of actual code, leveraging a custom matrix framework written from scratch. You can create a network, generating a dataset and evaluating the predictions in less than 10 lines of code!

```
#include "net.h"

int main() {

// INIT PHASE //
// Init a vector containing the layers of the MLP
std::vector<size_t> layers = {1, 8, 8, 8, 1};

// Init an mlp object using the layers vector and a fixed lr and iterations
Net<float> mlp(layers, 0.5, 20000);

// Load the net on the gpu
mlp.cuda();

// init a data obj
Data<float> data;

// DATAGEN PHASE //
// Generate a linspace of n points between a and b 
Matrix<float> X = data.linspace(0, 1, 100);

// Generate a target matrix using the sin^2(x) toy function
Matrix<float> Y = data.generate(X, "sin^2");

// TRAINING PHASE //
// Train the mlp using the generated data
mlp.train(X, Y);

// SAMPLE GENERATION AND PREDICT //
// Generate n random sample to predict 
Matrix<float> sample = data.sample(5);

// Generate the actual target for the sample
Matrix<float> target = data.generate(sample, "sin^2");

// Predict the output of the sample
Matrix<float> prediction = mlp.predict(sample);
```
