#include "matrix.h"

////// DEVICE FUNCTIONS //////

// Brief explanation about device functions: since we are working with CUDA, we need to define some functions
// that will be executed on the GPU. These functions are called device functions, and they are defined with the
// __device__ keyword. We will define those function to avoid writing the same code multiple times in the CUDA
// kernels, and to make the code more readable and modular by splitting the code into smaller functions.
// In particular, the device function will represent the activation function of the network, so we will give the 
// user the possibility to choose between different ones.

// Sigmoid activation function using template
template <typename T> 
__device__ T sigmoid(T x) {
    return 1 / (1 + exp(-x));
}

// ReLU activation function using template
template <typename T>
__device__ T relu(T x) {
    return x > 0 ? x : 0;
}

// Tanh activation function using template
// ttanh is used to avoid conflicts with the tanh function defined in the cmath library.
template <typename T>
__device__ T ttanh(T x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

// Derivative of the ReLU activation function using template
template <typename T>
__device__ T relu_derivative(T x) {

    // Basically deriving a ReLU function is just checking if the input is greater than 0 or not.
    // That because the derivate of x is always 1 and the derivative of 0 is 0.
    return x > 0 ? 1 : 0;
}

// Derivative of the sigmoid activation function using template 
template <typename T>
__device__ T sigmoid_derivative(T x) {

    // Using the chain rule we can derive the sigmoid function as follows
    // f'(x) = f(x) * (1 - f(x)). Where f(x) is the sigmoid function.
    return sigmoid(x) * (1 - sigmoid(x));
}

// derivative of the tanh activation function using template
template <typename T>
__device__ T tanh_derivative(T x) {

    // The derivative of the tanh function is 1 - tanh^2(x)s
    return 1 - pow(tanh(x), 2);
}

////// CUDA KERNELS //////

// Brief explanation: CUDA kernels are functions that are executed on the GPU. They are defined with the __global__ keyword,
// and they are called by the CPU to perform some operations on the GPU. The kernels are executed in parallel by multiple
// threads, and they are organized in blocks and grids.

// CUDA kernel for matrix multiplication.
// This function takes in input some pre-allocated memory arrays on the GPU and performs the matrix multiplication.
// The input matrices are A and B, and the output matrix is C.
// The dimensions of the matrices are rowsA x colsA and colsA x colsB, and the output matrix is rowsA x colsB.
template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, size_t rowsA, size_t colsA, size_t colsB) {

    // Brief explanation of the indexing of the matrix elements:
    // We need to calculate the row and column of the output matrix based on various factor: 
    // 1. The Block index in the grid 
    // 2. The size of the block 
    // 3. The thread index in the block
    // We need to displace the index by using the grid index and the block dimension to calculate 
    // the correct matrix index to work on. Since a kernel function is ran parallely on multiple threads,
    // each thread will have different values of blockIdx and blockDim. We use then the threadIdx to displace 
    // the correct element of the matrix to work on.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // If the calculated row and column are within the bounds of the output matrix, we calculate
    // the value of the output matrix at that position. We do this check to avoid out of bounds access,
    // since the number of threads in a block may be more than the number of elements in the matrix.
    if (row < rowsA && col < colsB) {

        // Value is zero initially
        T value = 0;

        // For the lenght of the columns of A (that are the same as the rows of B), we calculate the dot product
        // of the row of A and the column of B to get the value of the output matrix at that position.
        for (size_t k = 0; k < colsA; ++k) {

            // Dot product calculation
            value += A[row * colsA + k] * B[k * colsB + col];
        }

        // Store the value in the output matrix
        C[row * colsB + col] = value;

    }
}

// CUDA kernel for matrix addition.
// This function takes in input some pre-allocated memory arrays on the GPU and performs the matrix addition.
// The input matrices are A and B, and the output matrix is C.
// The dimensions of the matrices are rows x cols, and the output matrix is rows x cols: that because
// the addition output the same size of the input matrices.
template <typename T>
__global__ void add_kernel(const T* A, const T* B, T* C, size_t rows, size_t cols) {

    // Calculate the row and column of the output matrix based on the block index and thread index
    // as we did above for the matrix multiplication.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // If the calculated row and column are within the bounds of the output matrix, we calculate
    // the value of the output matrix at that position. We do this check to avoid out of bounds access,
    // since the number of threads in a block may be more than the number of elements in the matrix.
    if (row < rows && col < cols) {

        // Calculate the index of the element in the matrix
        size_t index = row * cols + col;

        // Add the elements of the input matrices and store the result in the output matrix
        C[index] = A[index] + B[index];

    }
}

// CUDA Kernel for matrix subtraction 
// This function takes in input some pre-allocated memory arrays on the GPU and performs the matrix subtraction.
// The input matrices are A and B, and the output matrix is C.
// The dimensions of the matrices are rows x cols, and the output matrix is rows x cols: that because
// the subtraction output the same size of the input matrices.
template <typename T>
__global__ void sub_kernel(const T* A, const T* B, T* C, size_t rows, size_t cols) {

    // Calculate the row and column of the output matrix based on the block index and thread index
    // as we did above for the matrix multiplication.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // If the calculated row and column are within the bounds of the output matrix, we calculate
    // the value of the output matrix at that position. We do this check to avoid out of bounds access,
    // since the number of threads in a block may be more than the number of elements in the matrix.
    if (row < rows && col < cols) {

        // Calculate the index of the element in the matrix
        size_t index = row * cols + col;

        // Subtract the elements of the input matrices and store the result in the output matrix
        C[index] = A[index] - B[index];

    }
}

// CUDA kernel that perform the scalar multiplication between a matrix and a scalar.
// The input matrix is A, the scalar is s, and the output matrix is C.
// The dimensions of the matrices are rows x cols, and the output matrix is rows x cols: that because
// the scalar multiplication output the same size of the input matrix.
template <typename T>
__global__ void scalmul_kernel(const T* A, const T* s, T* C, size_t rows, size_t cols) {

    // Calculate the row and column of the output matrix based on the block index and thread index
    // as we did above for the matrix multiplication.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // If the calculated row and column are within the bounds of the output matrix, we calculate
    // the value of the output matrix at that position. We do this check to avoid out of bounds access,
    // since the number of threads in a block may be more than the number of elements in the matrix.
    if (row < rows && col < cols) {

        // Calculate the index of the element in the matrix
        size_t index = row * cols + col;

        // Multiply the element of the input matrix by the scalar and store the result in the output matrix
        // Since the scalar is a pointer, we need to access the first element of the array to get the scalar value.
        C[index] = A[index] * s[0];

    }
}

// CUDA kernel for pointwise matrix multiplication (Hadamard product).
// This function takes in input some pre-allocated memory arrays on the GPU and performs the pointwise matrix multiplication.
// The input matrices are A and B, and the output matrix is C.
// The dimensions of the matrices are rows x cols, and the output matrix is rows x cols: that because
// the pointwise multiplication output the same size of the input matrices.
template <typename T>
__global__ void multiply_kernel(const T* A, const T* B, T* C, size_t rows, size_t cols) {

    // Calculate the row and column of the output matrix based on the block index and thread index
    // as we did above for the matrix multiplication.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // If the calculated row and column are within the bounds of the output matrix, we calculate
    // the value of the output matrix at that position. We do this check to avoid out of bounds access,
    // since the number of threads in a block may be more than the number of elements in the matrix.
    if (row < rows && col < cols) {

        // Calculate the index of the element in the matrix
        size_t index = row * cols + col;

        // Multiply the elements of the input matrices and store the result in the output matrix
        C[index] = A[index] * B[index];

    }
}

// CUDA Kernel for squaring a matrix element-wise.
// This function takes in input some pre-allocated memory arrays on the GPU and squares the elements of a matrix.
// The input matrix is A, and the output matrix is C.
// The dimensions of the matrices are rows x cols, and the output matrix is rows x cols: that because
// the squaring operation output the same size of the input matrix.
template <typename T>
__global__ void square_kernel(const T* A, T* C, size_t rows, size_t cols) {

    // Calculate the row and column of the output matrix based on the block index and thread index
    // as we did above for the matrix multiplication.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // If the calculated row and column are within the bounds of the output matrix, we calculate
    // the value of the output matrix at that position. We do this check to avoid out of bounds access,
    // since the number of threads in a block may be more than the number of elements in the matrix.
    if (row < rows && col < cols) {

        // Calculate the index of the element in the matrix
        size_t index = row * cols + col;

        // Square the element of the input matrix and store the result in the output matrix
        C[index] = A[index] * A[index];

    }
}

// CUDA Kernel for transposing a matrix.
// This function takes in input some pre-allocated memory arrays on the GPU and performs the matrix transposition.
// The input matrix is A, and the output matrix is C.
// The dimensions of the matrices are rows x cols, and the output matrix is cols x rows: that because
// the transposition of a matrix is a change of the rows and columns.
template <typename T>
__global__ void transpose_kernel(const T* A, T* C, size_t rows, size_t cols) {

    // Calculate the row and column of the output matrix based on the block index and thread index
    // as we did above for the matrix multiplication.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // If the calculated row and column are within the bounds of the output matrix, we calculate
    // the value of the output matrix at that position. We do this check to avoid out of bounds access,
    // since the number of threads in a block may be more than the number of elements in the matrix.
    if (row < cols && col < rows) {

        // Calculate the index of the element in the matrix
        size_t index = row * rows + col;

        // Transpose the matrix and store the result in the output matrix
        // To swap the rows and columns, we need to swap the row and column indexes.
        // So instead of doing row*rows + col, we do col*cols + row, effectively displacing 
        // the element to the correct position in the output matrix in a 1D manner.
        C[index] = A[col * cols + row];

    }
}

// CUDA Kernel to apply a sigmoid function to a matrix element-wise.
// This function takes in input some pre-allocated memory arrays on the GPU and applies the sigmoid function to a matrix.
// The input matrix is A, and the output matrix is C.
// The dimensions of the matrices are rows x cols, and the output matrix is rows x cols: that because
// the sigmoid function output the same size of the input matrix.
template <typename T>
__global__ void activation_kernel(const T* A, T* C, size_t rows, size_t cols, char* f) {

    // Calculate the row and column of the output matrix based on the block index and thread index
    // as we did above for the matrix multiplication.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // If the calculated row and column are within the bounds of the output matrix, we calculate
    // the value of the output matrix at that position. We do this check to avoid out of bounds access,
    // since the number of threads in a block may be more than the number of elements in the matrix.
    if (row < rows && col < cols) {

        // Calculate the index of the element in the matrix
        size_t index = row * cols + col;

        // Apply the activation function to the element of the input matrix and store the result in the output matrix.
        // We use a switch statement to choose the activation function based on the input parameter.
        // Add your activation function here if you want to use a different one.
        switch (f[0]) {
            case 's':
                C[index] = sigmoid(A[index]);
                break;
            case 'r':
                C[index] = relu(A[index]);
                break;
            case 't':
                C[index] = ttanh(A[index]);
                break;
            default:
                C[index] = sigmoid(A[index]);
        }

    }

}

// CUDA Kernel to copmute the derivative of the sigmoid function.
// This function takes in input some pre-allocated memory arrays on the GPU and applies the derivative of the sigmoid function to a matrix.
// The input matrix is A, and the output matrix is C.
// The dimensions of the matrices are rows x cols, and the output matrix is rows x cols: that because
// the sigmoid function output the same size of the input matrix.
template <typename T>
__global__ void derivative_kernel(const T* A, T* C, size_t rows, size_t cols, char* f) {

    // Calculate the row and column of the output matrix based on the block index and thread index
    // as we did above for the matrix multiplication.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // If the calculated row and column are within the bounds of the output matrix, we calculate
    // the value of the output matrix at that position. We do this check to avoid out of bounds access,
    // since the number of threads in a block may be more than the number of elements in the matrix.
    if (row < rows && col < cols) {

        // Calculate the index of the element in the matrix
        size_t index = row * cols + col;

        // Apply the derivative of the activation function to the element of the input matrix and store the result in 
        // the output matrix. We use a switch statement to choose the derivative function based on the input parameter.
        // Add your derivative function here if you want to use a different one.
        switch (f[0]) {
            case 's':
                C[index] = sigmoid_derivative(A[index]);
                break;
            case 'r':
                C[index] = relu_derivative(A[index]);
                break;
            case 't':
                C[index] = tanh_derivative(A[index]);
                break;
            default:
                C[index] = sigmoid_derivative(A[index]);
        }

    }

}

// Explicit template instantiation for the CUDA functions.
// We have to do this because the CUDA kernels are defined in a separate file, and the compiler
// needs to know the types that will be used in the kernels at compile time.
template __global__ void matmul_kernel<float>(const float*, const float*, float*, size_t, size_t, size_t);
template __global__ void matmul_kernel<double>(const double*, const double*, double*, size_t, size_t, size_t);
template __global__ void add_kernel<float>(const float*, const float*, float*, size_t, size_t);
template __global__ void add_kernel<double>(const double*, const double*, double*, size_t, size_t);
template __global__ void sub_kernel<float>(const float*, const float*, float*, size_t, size_t);
template __global__ void sub_kernel<double>(const double*, const double*, double*, size_t, size_t);
template __global__ void scalmul_kernel<float>(const float*, const float*, float*, size_t, size_t);
template __global__ void scalmul_kernel<double>(const double*, const double*, double*, size_t, size_t);
template __global__ void multiply_kernel<float>(const float*, const float*, float*, size_t, size_t);
template __global__ void multiply_kernel<double>(const double*, const double*, double*, size_t, size_t);
template __global__ void square_kernel<float>(const float*, float*, size_t, size_t);
template __global__ void square_kernel<double>(const double*, double*, size_t, size_t);
template __global__ void transpose_kernel<float>(const float*, float*, size_t, size_t);
template __global__ void transpose_kernel<double>(const double*, double*, size_t, size_t);
template __global__ void activation_kernel<float>(const float*, float*, size_t, size_t, char*);
template __global__ void activation_kernel<double>(const double*, double*, size_t, size_t, char*);
template __global__ void derivative_kernel<float>(const float*, float*, size_t, size_t, char*);
template __global__ void derivative_kernel<double>(const double*, double*, size_t, size_t, char*);


////// WRAPPER FUNCTION FOR CUDA KERNELS //////

// Defining a wrapper function for performing different operations on the matrix, based on the caller function.
// This function is used to call the correct CUDA kernel based on the operation that needs to be performed,
// without having to write the same code multiple times.
template <typename T>
Matrix<T> cuda_kernel_wrapper(Matrix<T> A, Matrix<T> B, T* d_data, std::string caller, size_t r, size_t c) {

    // Create the output matrix: We use r rows and c cols.
    // Now, if the op is a matmul, it will have, i.e: 1x3 * 3x2 = 1x2. 
    // But if the op is another one (add, sub, multiply, scalar) it will literally have the same shape as the 
    // input matrices (i.e: 3x3 + 3x3 = 3x3, 3x3 * 3x3 = 3x3, 3x3 * 3 = 3x3, etc.)
    Matrix<T> result(r, c);

    // SInce we have in input a big chunk of memory allocated in the Net class, 
    // we're going to displace the address of this chunk to the block of memory that we need.
    // The initial block points to the starting area of the d_data memory.
    T* d_A = d_data;

    // The second block starts at d_data + the size of the first matrix
    T* d_B = d_data + A.rows * A.cols;

    // The third block starts at d_data + the size of the first matrix + the size of the second matrix
    T* d_C = d_data + A.rows * A.cols + B.rows * B.cols;

    // Copy data to the GPU: we copy the raw data of the matrices to the GPU memory
    // We need to copy the data to the GPU memory since the CUDA kernels will be executed on the GPU,
    // and we do actually copy the data to the memory blocks that we defined just above.
    // We use the cudaMemcpy function, specifying the direction of the copy (from host to device), 
    // the pointer to the data on the host, the pointer to the data on the device, and the size
    // of the data to copy.
    cudaMemcpy(d_A, A.rawdata().data(), A.rows * A.cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.rawdata().data(), B.rows * B.cols * sizeof(T), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions: we use the max number of threads per block (1024) for our 
    // device (Quadro K5000). Change based on your needs.
    dim3 blockDim(32, 32);

    // The grid dimensions are calculated based on the size of the matrices: we need to have enough blocks
    // to cover all the elements of the output matrix. We divide the number of elements by the number of threads
    // in a block to get the number of blocks needed.
    dim3 gridDim((B.cols + blockDim.x - 1) / blockDim.x, (A.rows + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel based on the caller function
    if (caller == "matmul") {
        matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, A.rows, A.cols, B.cols);
    }
    else if (caller == "add") {
        add_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, A.rows, A.cols);
    }
    else if (caller == "sub") {
        sub_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, A.rows, A.cols);
    }
    else if (caller == "scalmul") {
        scalmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, A.rows, A.cols);
    }
    else if (caller == "multiply") {
        multiply_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, A.rows, A.cols);
    }
    else if (caller == "square") {
        square_kernel<<<gridDim, blockDim>>>(d_A, d_C, A.rows, A.cols);
    }
    else if (caller == "transpose") {
        transpose_kernel<<<gridDim, blockDim>>>(d_A, d_C, A.rows, A.cols);
    }
    else {
        std::cerr << "Invalid caller function" << std::endl;
    }

    // Wait for GPU to finish before accessing on the data to transfer to the host
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(result.data.data(), d_C, result.rows * result.cols * sizeof(T), cudaMemcpyDeviceToHost);

    // NOTE: we do not free the memory since it will be needed for some operations.
    // The free operation will be done in the destructor of the Net class.

    return result;

}

// Wrapper function for activations 
template <typename T>
Matrix<T> cuda_kernel_activation_wrapper(Matrix<T> A, T* d_data, const std::string f, char* d_f, std::string type) {

    // Create the output matrix: we don't need to specify the shape of the output matrix since it will be the same
    // as the input matrix.
    Matrix<T> result(A.rows, A.cols);

    // SInce we have in input a big chunk of memory allocated in the Net class, 
    // we're going to displace the address of this chunk to the block of memory that we need.
    // The initial block points to the starting area of the d_data memory.
    T* d_A = d_data;

    // The second block starts at d_data + the size of the first matrix
    T* d_C = d_data + A.rows * A.cols;

    // Copy data to the GPU
    cudaMemcpy(d_A, A.rawdata().data(), A.rows * A.cols * sizeof(T), cudaMemcpyHostToDevice);

    // Copy the activation function to the GPU
    cudaMemcpy(d_f, &f[0], 1 * sizeof(char), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions: we use the max number of threads per block (1024) for our 
    // device (Quadro K5000). Change based on your needs.
    dim3 blockDim(32, 32);

    // The grid dimensions are calculated based on the size of the matrices: we need to have enough blocks
    // to cover all the elements of the output matrix. We divide the number of elements by the number of threads
    // in a block to get the number of blocks needed.
    dim3 gridDim((A.cols + blockDim.x - 1) / blockDim.x, (A.rows + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel based on the caller function
    if (type == "activate") {
        activation_kernel<<<gridDim, blockDim>>>(d_A, d_C, A.rows, A.cols, d_f);
    }
    else if (type == "derivate") {
        derivative_kernel<<<gridDim, blockDim>>>(d_A, d_C, A.rows, A.cols, d_f);
    }
    else {
        std::cerr << "Invalid caller function" << std::endl;
    }

    // Wait for GPU to finish before accessing on the data to transfer to the host
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(result.data.data(), d_C, result.rows * result.cols * sizeof(T), cudaMemcpyDeviceToHost);

    // NOTE: we do not free the memory since it will be needed for some operations.
    // The free operation will be done in the

    // Return the result
    return result;

}

////// MATRIX CLASS METHODS //////

// MATMUL function for the Matrix class.
// This function performs the matrix multiplication between two matrices.
// The input matrix is m, and the output matrix is the result of the multiplication.
template <typename T>
Matrix<T> Matrix<T>::matmul(Matrix m, T* d_data) {

    // Check if the dimensions of the matrices are compatible for the matmul
    assert(this->cols == m.rows);

    // Call the wrapper function for matrix multiplication
    return cuda_kernel_wrapper((*this), m, d_data, "matmul", this->rows, m.cols);

}

// ADD function for the Matrix class.
// This function performs the matrix addition between two matrices.
// The input matrix is m, and the output matrix is the result of the addition.
template <typename T>
Matrix<T> Matrix<T>::add(Matrix m, T* d_data) {

    // Check if the shape of the two matrices are the same
    assert(this->tshape == m.tshape);

    // Call the wrapper function for matrix addition
    return cuda_kernel_wrapper((*this), m, d_data, "add", this->rows, this->cols);

}

// SUB function for the Matrix class.
// This function performs the matrix subtraction between two matrices.
// The input matrix is m, and the output matrix is the result of the subtraction.
template <typename T>
Matrix<T> Matrix<T>::subtract(Matrix m, T* d_data) {

    // Check if the shape of the two matrices are the same
    assert(this->tshape == m.tshape);

    // Call the wrapper function for matrix subtraction
    return cuda_kernel_wrapper((*this), m, d_data, "sub", this->rows, this->cols);

}

// MULTIPLY (Scalar) function for the Matrix class.
// This function performs the scalar multiplication between a matrix and a scalar.
// The input scalar is s, and the output matrix is the result of the scalar multiplication.
template <typename T>
Matrix<T> Matrix<T>::multiply(T s, T* d_data) {

    // We transform the scalar into a 1,1 matrix and then we call the scalar multiplication
    Matrix<T> scalar(1, 1);
    scalar(0, 0) = s;

    // Call the wrapper function for scalar multiplication
    return cuda_kernel_wrapper((*this), scalar, d_data, "scalmul", this->rows, this->cols);

}

// MULTIPLY (Hadamard Product) function for the Matrix class.
// This function performs the pointwise matrix multiplication (Hadamard product) between two matrices.
// The input matrix is m, and the output matrix is the result of the pointwise multiplication.
template <typename T>
Matrix<T> Matrix<T>::multiply(Matrix m, T* d_data) {

    // Check if the shape of the two matrices are the same
    assert(this->tshape == m.tshape);

    // Call the wrapper function for pointwise matrix multiplication
    return cuda_kernel_wrapper((*this), m, d_data, "multiply", this->rows, this->cols);

}

// SQUARE function for the Matrix class.
// This function squares the elements of a matrix element-wise.
// The output matrix is the result of the squaring operation.
template <typename T>
Matrix<T> Matrix<T>::square(T* d_data) {

    // Call the wrapper function for squaring a matrix element-wise
    return cuda_kernel_wrapper((*this), (*this), d_data, "square", this->rows, this->cols);

}

// TRANSPOSE function for the Matrix class.
// This function transposes a matrix.
// The output matrix is the transposed matrix.
template <typename T>
Matrix<T> Matrix<T>::transpose(T* d_data) {

    // Call the wrapper function for transposing a matrix
    return cuda_kernel_wrapper((*this), (*this), d_data, "transpose", this->cols, this->rows);

}

// Activation function for the Matrix class.
// This function applies an activation function to a matrix element-wise.
// The output matrix is the result of the activation function applied to the input matrix.
template <typename T>
Matrix<T> Matrix<T>::activate(std::string f, char* d_f, T* d_data) {

    // Call the wrapper function for applying an activation function to a matrix element-wise
    return cuda_kernel_activation_wrapper((*this), d_data, f, d_f, "activate");

}

// Derivative function for the Matrix class.
// This function computes the derivative of an activation function applied to a matrix element-wise.
// The output matrix is the result of the derivative of the activation function applied to the input matrix.
template <typename T>
Matrix<T> Matrix<T>::derivate(std::string f, char* d_f,  T* d_data) {

    // Call the wrapper function for computing the derivative of an activation function applied to a matrix element-wise
    return cuda_kernel_activation_wrapper((*this), d_data, f, d_f,  "derivate");

}



// Explicit instantiation of template function for matmul: we do this for the same exact 
// reason of the above 
template Matrix<float> cuda_kernel_wrapper(Matrix<float> A, Matrix<float> B, float* d_data, std::string caller, size_t r, size_t c);
template Matrix<double> cuda_kernel_wrapper(Matrix<double> A, Matrix<double> B, double* d_data, std::string caller, size_t r, size_t c);
template Matrix<float> Matrix<float>::matmul(Matrix<float> m, float* d_data);
template Matrix<double> Matrix<double>::matmul(Matrix<double>, double* d_data);
template Matrix<float> Matrix<float>::add(Matrix<float> m, float* d_data);
template Matrix<double> Matrix<double>::add(Matrix<double>, double* d_data);
template Matrix<float> Matrix<float>::subtract(Matrix<float> m, float* d_data);
template Matrix<double> Matrix<double>::subtract(Matrix<double>, double* d_data);
template Matrix<float> Matrix<float>::multiply(float s, float* d_data);
template Matrix<double> Matrix<double>::multiply(double s, double* d_data);
template Matrix<float> Matrix<float>::multiply(Matrix<float> m, float* d_data);
template Matrix<double> Matrix<double>::multiply(Matrix<double> m, double* d_data);
template Matrix<float> Matrix<float>::square(float* d_data);
template Matrix<double> Matrix<double>::square(double* d_data);
template Matrix<float> Matrix<float>::transpose(float* d_data);
template Matrix<double> Matrix<double>::transpose(double* d_data);
template Matrix<float> Matrix<float>::activate(std::string f, char* d_f, float* d_data);
template Matrix<double> Matrix<double>::activate(std::string f, char* d_f, double* d_data);
template Matrix<float> Matrix<float>::derivate(std::string f, char* d_f, float* d_data);
template Matrix<double> Matrix<double>::derivate(std::string f, char* d_f, double* d_data);


