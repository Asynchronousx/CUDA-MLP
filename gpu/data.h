// Library which contains function to create and manipulate data for the network and to/from the disk.
#pragma once 
#include "matrix.h"
#include <cstddef>
#include <cstring>
#include <fstream>
#include <vector>

// Class manager that contains the data manipulation functions.
template <typename T>
class Data {

    public:

        // Default constructor
        Data() {}

        // Function to generate a linspace of n points between a and b
        Matrix<T> linspace(T a, T b, size_t n) {

            // Create a copy of the current matrix to avoid modifying the current instance of the matrix
            Matrix<T> result(1, n);

            // Compute the step size: this is done by dividing the difference between b and a by n - 1.
            // n - 1 because we want to include the last element in the linspace.
            T step = (b-a)/(n-1);

            // Loop through the columns of the matrix
            for (size_t i=0; i<n; i++) {

                // Fill the element at the index (0, i) with the value of a + i * step
                result(0, i) = a+i * step;
            }

            // Return the current instance of the matrix
            return result;

        }

        // Function to generate a target matrix using a specific toy function:
        // The function are: sin^2(x), cos^2(x), tanh(x), sin(x), cos(x).
        Matrix<T> generate(Matrix<T> X, std::string function) {

            // Assign the target function to the targetfun attribute
            this->targetfun = function;
            
            // Some if and else to check the correct function to use. 
            // We gonna them apply pointwise the function required to the matrix X.
            if (!std::strcmp(function.c_str(), "sin")) 
                return X.apply([](T x) { return sin(x); });

            if (!std::strcmp(function.c_str(), "cos")) 
                return X.apply([](T x) { return cos(x); });
            
            if (!std::strcmp(function.c_str(), "tanh"))
                return X.apply([](T x) { return tanh(x); });

            if (!std::strcmp(function.c_str(), "sin^2"))
                return X.apply([](T x) { return sin(x) * sin(x); });

            if (!std::strcmp(function.c_str(), "cos^2"))    
                return X.apply([](T x) { return cos(x) * cos(x); });

            // If the function is not recognized, we reset the targetfun 
            this->targetfun = "";

            // If the function is not recognized, we return an empty matrix
            std::cout << "Toy Function not recognized, returning an empty matrix" << std::endl;
            return Matrix<T>(0, 0);
            
        }


        // Function to generate some sample data to use for the network predictions.
        // The function takes three arguments: a, b and n, where a and b are the range of the random values
        // and n is the number of random values to generate.
        Matrix<T> sample(size_t n = 1, T a = 0, T b = 1) {

            // Create a matrix (or row vector) of size 1 x n filled with random values between a and b
            Matrix<T> result(1, n);

            // We init the random seed using the current time
            srand(time(NULL));

            // Loop through the columns of the matrix
            for (size_t i=0; i<n; i++) {

                // Fill the element at the index (0, i) with a random value between a and b.
                // Note that, since we don't want any specific distribution for the data, 
                // we can simply use the rand() function to generate random values between a and b.
                // The purpose of those values is only to test the network with some predictions.
                result(0, i) = a + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (b-a)));

            }

            // Return the matrix
            return result;

        }

        // Function to save hyperparameters to a CSV file. The function takes three arguments:
        // Layers: the vector containing the layers of the network, 
        // lr: the learning rate of the network,
        // iterations: the number of iterations to train the network, 
        // ID: the ID of the model,
        // fun: the function used to generate the target matrix, 
        // file: the name of the file to save the hyperparameters.
        void savehyper(const std::vector<size_t>& layers, T lr, size_t iterations, size_t ID, const std::string activation, const std::string& file) {

            // Creating the model directory
            std::string dirname = "model_" + std::to_string(ID);

            // We create the folder using the system command
            system(("mkdir -p " + dirname).c_str());

            // We concatenate the file to the directory name
            std::string combined_path = dirname + "/" + file;

            // Open the file using the computer filename
            std::ofstream f(combined_path, std::ios::out);

            // Check if the file is open
            if (!f.is_open()) {

                // If the file is not open, throw an exception
                throw std::runtime_error("Could not open file");
            }

            // Write layers to the file
            f << "Layers: ";

            // Write the layers of the network to the file
            for (size_t i = 0; i < layers.size(); ++i) {

                // Write the element at the index i to the file
                f << layers[i];

                // If the index is not the last element, write a comma
                if (i != layers.size() - 1) {
                    f << ",";
                }
            }

            // Write a new line character to the file
            f << std::endl;

            // Write layers to the file
            f << "LR: ";

            // Write the learning rate to the file
            f << lr << std::endl;

            // Write iterations to the file
            f << "Iterations: ";

            // Write the number of iterations to the file
            f << iterations << std::endl;

            // Write the activation function to the file
            f << "Activation: ";

            // Write the activation function to the file
            f << activation << std::endl;

            // Close the file
            f.close();

        }

        // Function overload to save a dataset to a CSV file. The function takes three arguments:
        // V: the input vector, Y: the target matrix, and dirname: the name of the directory to save the files.
        void save(std::vector<T> V, const std::string& name, size_t ID) {

            // Creating the model directory
            std::string dirname = "model_" + std::to_string(ID);

            // We check if a folder exists, if not we create it
            system(("mkdir -p " + dirname).c_str());

            // Save the matrix data to a CSV file
            this->vec2csv(V, dirname, name);

        }

        // Function to retrieve the target function used to generate the target matrix
        // in the last generate call.
        std::string genfun() {
            return this->targetfun;
        }

    private:

        // Attribute: target function used to generate the target matrix.
        std::string targetfun;

        // Function to save a vector to a CSV file. The function takes three arguments:
        // data: the vector to save, filename: the name of the file, and delimiter: type of delimiter to use.
        // Note: we pass const reference to the object to avoid copying the objects.
        void vec2csv(const std::vector<T>& data, const std::string& dir, const std::string& file, char del = ',') {

            // We check if a folder exists, if not we create it
            system(("mkdir -p " + dir).c_str());

            // After, we create the full path to the file using fs::path
            std::string combined_path = dir + "/" + file;
            
            // Open the file using the computer filename
            std::ofstream f(combined_path, std::ios::out);

            // Check if the file is open
            if (!f.is_open()) {

                // If the file is not open, throw an exception
                throw std::runtime_error("Could not open file");
            }

            // Loop through the data vector
            for (size_t i = 0; i < data.size(); ++i) {

                // Write the element at the index i to the file
                f << data[i];

                // If the index is not the last element, write the delimiter
                if (i != data.size() - 1) {
                    f << del;
                }
            }

            // Write the end new line character to the file
            f << std::endl;

            // Close the file
            f.close();

        }

};