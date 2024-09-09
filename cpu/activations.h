// File in which the activation functions and their derivatives are defined. 
#pragma once
#include <iostream>

// Sigmoid activation function using template
template <typename T> 
T sigmoid(T x) {
    return 1 / (1 + exp(-x));
}

// ReLU activation function using template
template <typename T>
T relu(T x) {
    return x > 0 ? x : 0;
}

// Tanh activation function using template
template <typename T>
T tanh(T x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

// Derivative of the ReLU activation function using template
template <typename T>
T relu_derivative(T x) {

    // Basically deriving a ReLU function is just checking if the input is greater than 0 or not.
    // That because the derivate of x is always 1 and the derivative of 0 is 0.
    return x > 0 ? 1 : 0;
}

// Derivative of the sigmoid activation function using template 
template <typename T>
T sigmoid_derivative(T x) {

    // Using the chain rule we can derive the sigmoid function as follows
    // f'(x) = f(x) * (1 - f(x)). Where f(x) is the sigmoid function.
    return sigmoid(x) * (1 - sigmoid(x));
}

// derivative of the tanh activation function using template
template <typename T>
T tanh_derivative(T x) {

    // The derivative of the tanh function is 1 - tanh^2(x)s
    return 1 - pow(tanh(x), 2);
}