// Include net.h file
#include "net.h"
#include <iostream>

// CPP file to generate the necessary output for the python plot scripts. 
// It will generate an X input dataset matrix composed by a linspace of N points between 0 and 1.
// Then it will generate the Y target matrix using the sin^2(x) toy function.
// It will predict the X input dataset against the untrained and trained mlp to have a comparison.
// Then, it will dump the X, Y, untrained_Yhat, and trained_Yhat to a csv file to be used in the python plot scripts.
// Modify the values of your choice to have different results (layers, lr, iterations, linspace, function etc).
int main() {

// INIT PHASE //
// Init a vector containing the layers of the MLP
std::vector<size_t> layers = {1, 8, 8, 8, 1};

// Init an mlp object using the layers vector and a fixed lr and iterations
Net<float> mlp(layers, 0.5, 10000);

// init a data obj
Data<float> data;

// DATAGEN PHASE //
// Generate a linspace of n points between a and b 
Matrix<float> X = data.linspace(0, 1, 100);

// Generate a target matrix using the sin^2(x) toy function
Matrix<float> Y = data.generate(X, "sin^2");


// UNTRAINED PREDICTION PHASE //
// Before training, we predict the output of the target matrix.
Matrix<float> untrained_prediction = mlp.predict(X);

// TRAINING PHASE //
// Train the mlp using the generated data
mlp.train(X, Y);

// TRAINED PREDICTION PHASE //
// Predict the output of the sample
Matrix<float> prediction = mlp.predict(X);

// DUMP PHASE //

// We save the model 
mlp.save();

// We also save X, Y, and the prediction vectors to a csv file using the ID of the mlp object
// as identifier for the folder name
data.save(X.rawdata(), "X.csv", mlp.ID);
data.save(Y.rawdata(), "Y.csv", mlp.ID);
data.save(untrained_prediction.rawdata(), "untrained_Yhat.csv", mlp.ID);
data.save(prediction.rawdata(), "trained_Yhat.csv", mlp.ID);

// End of the program
std::cout << "Dumped the data to csv files!" << std::endl;

}

// Compile the program using the following command:
// g++ main.cpp -o main
