// Include net.h file
#include "net.h"

// Main entry point of the program. This will initialize an mlp, generate a dataset with his target, 
// train the mlp using the dataset and predict the output of the mlp using some random sample, 
// printing the input, prediction, actual target and the error of the prediction.
// Modify the values of your choice to have different results (layers, lr, iterations, linspace, function etc).
int main() {

// INIT PHASE //
// Init a vector containing the layers of the MLP
std::vector<size_t> layers = {1, 8, 8, 8, 1};

// Init an mlp object using the layers vector and a fixed lr and iterations
Net<float> mlp(layers, 0.5, 10000);

// init a data obj
Data<float> data;

// TRAINING/TARGET DATA GENERATION PHASE //
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

// PRINT PHASE //
// Print input 
std::cout << "Input: " << std::endl;
sample.print();

// Print the prediction
std::cout << "Prediction: " << std::endl;
prediction.print();

// Print the actual target
std::cout << "Actual: " << std::endl;
target.print();

}

// Compile the program using the following command:
// g++ main.cpp -o main