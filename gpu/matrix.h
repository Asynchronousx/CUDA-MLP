// matrix.h file containing all the necessary libraries and headers for the matrix class. 
// Since we're using template programming, we need to include the implementation in the header file.
#pragma once
#include <cstddef>
#include <sys/types.h>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <tuple>
#include <functional>
#include <random>
#include <cuda_runtime.h>

// PI CONST
const float PI {3.14159};

// Class that represent a matrix and the linear algebra operations that can be performed on it.
// Given rows, columns and a value, we can create a matrix of size rows * cols filled with the value v (default 0).
template <typename T> class Matrix {

    public:

        // Number of rows and columns in the matrix
        size_t rows, cols;

        // Vector to store the data of the matrix: we use 1D vector of size rows * cols of type T
        // to avoid useless space complexity of 2D vectors and to make the matrix class more efficient.
        std::vector<T> data;
        
        // Shape of the matrix as a tuple of size_t of rows, cols
        std::tuple<size_t, size_t> tshape;

        // Standard constructor
        Matrix() {}
        Matrix(size_t rows, size_t cols, T v=0) {
            
            // Set rows, cols, shape
            this->rows = rows;
            this->cols = cols;
            this->tshape = std::make_tuple(rows, cols);

            // We init an empty matrix of values v of size rows * cols
            this->data = std::vector<T>(rows * cols, v);

        }

        // Destructor
        ~Matrix() {}

        /////////// OVERLOADING OPERATORS ///////////

        // Overload the () operator to both access and modify the elements of the matrix.
        T& operator()(size_t rows, size_t cols) {

            // We return a reference to the element at the index (rows, cols) of the matrix
            // of type T so we can read or modify the element at that index.
            // We simply multiply rows by the number of columns to dispatch the 2D index to a 1D index, 
            // then add the current cols index.
            return this->data[rows * this->cols + cols];
        
        }

        // Overloading the = operator to assign the values of a vector to the inner data 
        Matrix operator=(const std::vector<T>& v) {
            
            // Check if the size of the vector is the same as the size of the matrix
            assert(v.size() == this->rows * this->cols);

            // Create a copy of the current matrix to avoid modifying the current instance of the matrix.
            // This instruction, specifically, will call EXPLICITLY the copy constructor.
            Matrix result(this->rows, this->cols);

            // Copy the data of the vector to the inner data of the matrix
            result.data = v;

            // Return the current instance of the matrix
            return result;

        }

        /////////// MATRIX UTILS OPERATIONS ////////

        // Function to retrieve the data vector of the matrix 
        std::vector<T> rawdata() {
            return this->data;
        }

        // Function to print the shape of the matrix
        void shape() {
            std::cout << "Shape (r,c): " << std::get<0>(this->tshape) << " " << std::get<1>(this->tshape) << std::endl;
        }

        // Function to get the shape of the matrix
        std::tuple<size_t, size_t> get_shape() {
            return this->tshape;
        }

        // Function to print the matrix
        void print() {

            // Loop through the rows and columns of the matrix
            for (size_t i = 0; i < this->rows; i++) {
                for (size_t j = 0; j < this->cols; j++) {

                    // Print the element at the current index:
                    std::cout << (*this)(i,j) << " ";
                }

                // Print a new line after each row
                std::cout << std::endl;
            }
        }

        // Function to reset the matrix to zeros
        void zeros() {

            // Loop through the rows and columns of the matrix
            for (size_t i = 0; i < this->rows; i++) {
                for (size_t j = 0; j < this->cols; j++) {

                    // Set the element at the current index to 0
                    (*this)(i, j) = 0;
                }
            }
        }

        // Function to generate a random number using the normal distribution
        T random_normal(T mean=0, T std=1) {
            
            // Using the normal distribution to generate random numbers 
            // with a mean and standard deviation, using C++11 random library.
            static std::random_device __randomDevice;
            static std::mt19937 __randomGen(__randomDevice());
            static std::normal_distribution<T> __normalDistribution(mean, std);

            // After generating our distribution, we return a random number.
            return __normalDistribution(__randomGen);

        }

        // Function to init weights of the matrix using HE initialization
        void init_weights() {

            // The He initialization is computed as follows:
            // We initialize the weights of the neural network by drawing them from a normal distribution
            // with a mean of 0 and a standard deviation of sqrt(2 / input_units), where fan_in is the number of input units.
            int input_units = this->rows;

            // Compute the standard deviation
            T std = std::sqrt(2.0 / input_units);

            // Loop through the rows and columns of the matrix
            for (int i = 0; i < this->rows; ++i) {
                for (int j = 0; j < this->cols; ++j) {

                    // Fill the element at the current index with a random value.
                    (*this)(i, j) = random_normal(0.0, std);

                }
            }
        }

        // Function to fill the matrix with random values between a and b (default 0, 1])
        void fill_random(T a, T b) {

            // Loop through the rows and columns of the matrix
            for (size_t i = 0; i < this->rows; i++) {
                for (size_t j = 0; j < this->cols; j++) {

                    // Fill the element at the current index with a random value.
                    // NBTE: Since we use the overloaded () operator to access the element at the index (i, j)
                    // we need to use (*this) to deferenciate & access the current instance of the matrix class.
                    (*this)(i, j) = this->random_normal(a, b);
                }
            }
        }

        // Function to compute the mean of the matrix
        T mean() {

            // Init a variable to store the sum of the elements of the matrix
            T sum = 0;

            // Loop through the rows and columns of the matrix
            for (size_t i = 0; i < this->rows; i++) {
                for (size_t j = 0; j < this->cols; j++) {

                    // Add the element at the index (i, j) to the sum
                    sum += (*this)(i, j);
                }
            }

            // Return the mean of the matrix
            return sum / (this->rows * this->cols);

        }

        // Function to apply a function element-wise to each element of the matrix.
        // This will be useful to apply activation functions to the elements of the matrix.
        Matrix apply(std::function<T(T)> f) {

            // Create a copy of the current matrix to avoid modifying the current instance of the matrix
            Matrix result(this->rows, this->cols);

            // Loop through the rows and columns of the matrix
            for (size_t i = 0; i < this->rows; i++) {
                for (size_t j = 0; j < this->cols; j++) {

                    // Apply the function f to the element at the index (i, j) of the matrix
                    result(i, j) = f((*this)(i, j));
                }
            }

            // Return the current instance of the matrix
            return result;

        }
        
        /////////// LINEAR ALGEBRA OPERATIONS ///////////

        // Those are the standard linear algebra operations that can be performed on matrices.
        // Later, we will implement those functions using CUDA to leverage the GPU for faster computations.
        // But leaving the CPU implementation here for reference or for anyone that wants to try 
        // them in combination with CUDA functions.

        // Function to add two matrix element-wise
        Matrix add(Matrix m) {
            
            // Check if the shape of the two matrices are the same
            assert(this->tshape == m.tshape);

            // Create a copy of the current matrix to avoid modifying the current instance of the matrix
            Matrix result(this->rows, this->cols);

            // For the rows of the matrix
            for (size_t i = 0; i < this->rows; i++) {

                // For the columns of the matrix
                for (size_t j = 0; j < this->cols; j++) {

                    // add the element at the index (i, j) of the current matrix to the element at the index (i, j) of the matrix m
                    result(i, j) = (*this)(i, j) + m(i, j);
                }
            }

            // Return the current instance of the matrix
            return result;

        }

        // Function to subtract two matrices element-wise
        Matrix subtract(Matrix m) {
            
            // Check if the shape of the two matrices are the same
            assert(this->tshape == m.tshape);

            // Create a copy of the current matrix to avoid modifying the current instance of the matrix
            Matrix result(this->rows, this->cols);

            // For the rows of the matrix
            for (size_t i = 0; i < this->rows; i++) {

                // For the columns of the matrix
                for (size_t j = 0; j < this->cols; j++) {

                    // Subtract the element at the index (i, j) of the current matrix to the element at the index (i, j) of the matrix m
                    result(i, j) = (*this)(i, j) - m(i, j);
                }
            }

            // Return the current instance of the matrix
            return result;

        }

        // Function to multiply a matrix by a scalar
        Matrix multiply(T scalar) {
            
            // Create a copy of the current matrix to avoid modifying the current instance of the matrix
            Matrix result(this->rows, this->cols);
            
            // For the rows of the matrix 
            for (size_t i = 0; i < this->rows; i++) {

                // For the columns of the matrix 
                for (size_t j = 0; j < this->cols; j++) {

                    // Multiply the element at the index (i, j) by the scalar
                    result(i, j) = (*this)(i, j) * scalar;
                                    
                }
            }

            // Return the current instance of the matrix
            return result;

        }

        // Function to multiply two matrices element-wise
        Matrix multiply(Matrix m) {
            
            // Check if the shape of the two matrices are the compatible
            // Print the shape of the two matrices
            assert(this->tshape == m.tshape);

            // Create a copy of the current matrix to avoid modifying the current instance of the matrix
            Matrix result((*this));

            // For the rows of the matrix
            for (size_t i = 0; i < this->rows; i++) {

                // For the columns of the matrix
                for (size_t j = 0; j < this->cols; j++) {

                    // Multiply the element at the index (i, j) of the current matrix to the element at the index (i, j) of the matrix m
                    result(i, j) *= m(i, j);
                }
            }

            // Return the current instance of the matrix
            return result;

        }

        // Function to multiply two matrices by performing matrix multiplication
        Matrix matmul(Matrix m) {
            
            // Check if the shape of the two matrices are the compatible
            assert(this->cols == m.rows);

            // Create a new matrix of size (rows, m.cols)
            Matrix result(this->rows, m.cols);

            // Loop through the rows of the first matrix
            for (size_t i = 0; i < this->rows; i++) {

                // Loop through the columns of the second matrix
                for (size_t j = 0; j < m.cols; j++) {

                    // Loop through the columns of the first matrix
                    for (size_t k = 0; k < this->cols; k++) {

                        // Multiply the element at the index (i, k) of the first matrix to the element at the index (k, j) of the second matrix
                        // and add the result to the element at the index (i, j) of the result matrix
                        result(i, j) += (*this)(i, k) * m(k, j);
                    }
                }
            }

            // Return the current instance of the matrix
            return result;
            
        }

        // Functio to square a matrix 
        Matrix square() {
            
            // Create a copy of the current matrix to avoid modifying the current instance of the matrix
            Matrix result(this->rows, this->cols);

            // Loop through the rows and columns of the matrix
            for (size_t i = 0; i < this->rows; i++) {
                for (size_t j = 0; j < this->cols; j++) {

                    // Square the element at the index (i, j) of the matrix
                    result(i, j) = std::pow((*this)(i, j), 2);
                }
            }

            // Return the current instance of the matrix
            return result;

        }

        // Function to get a transposed matrix
        Matrix transpose() {
            
            // A transposed matrix it's a matrix where the rows and columns are swapped.
            // Createa matrix of size (cols, rows)
            Matrix result(this->cols, this->rows);

            // Loop through the rows of the matrix
            for (size_t i = 0; i < this->rows; i++) {

                // Loop through the columns of the matrix
                for (size_t j = 0; j < this->cols; j++) {

                    // We then assign the element at the index (j, i) of the current matrix 
                    // to the element at the index (i, j) of the result matrix, effectively swapping 
                    // the rows and columns of the matrix.
                    result(j, i) = (*this)(i, j);
                }
            }

            // Return the current instance of the matrix
            return result;

        }


        /////////// CUDA OVERLOAD FUNCTIONS ///////////

        // The linear algebra operations are implemented in the matrix.cu file using CUDA.
        // We will declare the functions here and implement them in the matrix.cu file.
        // CUDA Functions declaration 
        Matrix add(Matrix m, T* d_data);
        Matrix subtract(Matrix m, T* d_data);
        Matrix multiply(T scalar, T* d_data);
        Matrix multiply(Matrix m, T* d_data);
        Matrix matmul(Matrix m, T* d_data);
        Matrix square(T* d_data);
        Matrix transpose(T* d_data);
        Matrix activate(const std::string, char* f, T* d_data);
        Matrix derivate(const std::string, char* f, T* d_data);

};

