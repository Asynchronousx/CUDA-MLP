// Actual implementation of our simple MLP
#include "data.h"
#include "activations.h"
#include <chrono>

template <typename T> class Net {

    public:

        // Simple ID for the network: a random 4 digit number
        size_t ID;

        // TEST: TWO VECTOR FOR CALCULATING FORWARD AVERAGE TIME AND 
        // BACKWARD AVERAGE TIME
        std::vector<float> forward_times;
        std::vector<float> backward_times;

        // Constructor
        Net(std::vector<size_t> layers, float lr=0.1, int iterations=1000, std::string activation="sigmoid") {

            // Initialize the attributes
            // Layer it's a vector that contains the number of units of each layer.
            this->layers = layers;

            // Lr and iterations are just hyperparameters of the network.
            this->lr = lr;
            this->loss = 0.;
            this->iterations = iterations;

            // Activation function of the network: sigmoid default
            this->afunc = activation;

            // Assign a random 4 digit number to the ID of the network
            srand(time(NULL));
            this->ID = rand() % 9000 + 1000;

            // Initialize the weights and biases: We have a weight and a bias matrix for each layer.
            // For a network with n layers, we'll have n-1 weights and n-1 biases matrices.
            // Example: a network with 3 layers (one input, one hidden and one output) will have 
            // 2 weights and 2 biases matrices.
            for (int i = 0; i < this->layers.size() - 1; i++) {

                // Getting the input and output units of the current layer 
                size_t input_units = this->layers[i];
                size_t output_units = this->layers[i + 1];

                // We initialize the weights using as rows the number of units of the current layer and 
                // as columns the number of units of the next layer. This is because the weights matrix
                // need to have the same number of columns as the number of units of the next layer to perform 
                // the matrix multiplication in the forward pass. We fill the matrix with random values
                // using the He initialization method. We can also use Xavier initialization or any other.
                Matrix<T> w(input_units, output_units);
                w.he();

                // We push the initialized weights to the weights vector of the network.
                this->weights.push_back(w);

                // We then init a bias matrix containing many nodes as the output units of the current layer.
                // Basically a one row matrix with the number of columns equal to the number of output units.
                // That because, for each unit of the current layer, we have a bias unit (TF standard).
                // Note that initializing a matrix like that will fill it with zeros by default of our matrix lib.
                this->biases.push_back(Matrix<T>(1, output_units));

            }


        }

        // Function to train the network using the input and target matrices X and Y.
        // It does also have a default value of 100 for the print_each_iter parameter.
        // and a default value of false for the saveinfo parameter which will save the loss and accuracy.
        void train(Matrix<T> X, Matrix<T> Y, size_t print_each_iter=100, bool saveinfo=false) {

            // The training process is the process of updating the weights and biases of the network
            // using the forward and backward passes. The training process is done for a number of iterations
            // specified by the iterations attribute of the network.

            // We init an avg loss variable to keep track of the loss of the network over n iterations.
            // We do this also for the network accuracy.
            float avg_loss = 0.;
            float avg_accuracy = 0.;

            // We also init a variable to store the time needed for each epoch, so we can 
            // estimate an ETA.
            float time = 0.;

            // We also need some vectors to store the loss and accuracy of the network over the iterations.
            // This will be helpful to plot the loss and accuracy of the network over the iterations
            // We already have the losses and accuracies vectors in the attributes of the network, but we need
            // to clear them before starting the training process.
            this->losses.clear();
            this->accuracies.clear();


            // We'll also need a threshold delta to calculate the accuracy of the network since we're
            // dealing with a regression problem.
            // The step delta it's equal to twice the step size in which the linear space is generated.
            // I.e: if we generate a linear space between 0 and 1 with 100 points, the delta will be 0.01,
            // hence the delta will be 0.02 to give some margin of error.
            float delta = 0.02;

            // Print the success of the network training
            std::cout << "Starting training.." << std::endl;

            // Loop over the number of iterations
            for (int i=0; i<this->iterations; i++) {

                // Start taking the time of the epoch
                auto start = std::chrono::high_resolution_clock::now();

                // We are going to use a super simple method to sample the data: we are going to pick 
                // a value from the dataset based on the iteration number. This is a very simple way to train
                // the network. In a real-world scenario, we would use a more sophisticated method to sample,
                // But for this task, sampling the data from the X array using circular indexing is enough.
                // I.E: we will sample the data from 0 to N using the iteration number MODULO the number of X cols.
                // In this way we will assure that every point will be seen by the network.
                // We asign a new matrix containing the input value of the corresponding iteration.
                Matrix<T> x = Matrix<T>(1, 1, X(0, i % X.cols));

                // Instantiate a matrix containing the target value of the corresponding input
                // Also here, we create a new matrix of 1,1 containing the target value of the corresponding input.
                // Refer to the matrix class constructor to see how this works.
                Matrix<T> y = Matrix<T>(1, 1, Y(0, i % Y.cols));

                // Forward pass: compute the activations of the network given an input X, and 
                // retrieve the output of the network.
                Matrix<T> yhat = this->forward(x);

                // Backward pass: compute the gradients of the weights and biases of the network.
                // We don't need to pass the output of the network to the backward pass because it's already
                // stored in back of the activations list.
                this->backward(y);

                // Sum the loss of the network
                avg_loss += this->loss;

                // Calculate the accuracy of the network
                // We need to subtract the output of the network from the target value and then
                // check if the absolute value of the difference is less than a threshold delta.
                // If the difference is less than the threshold, we increment the accuracy counter.
                // This is a very simple way to calculate the accuracy of the network.
                // We then divide the accuracy by the number of samples to get the average accuracy.
                if (std::abs(y(0,0) - yhat(0,0)) < delta) {
                    avg_accuracy += 1;
                }

                // End taking the time of the epoch
                auto end = std::chrono::high_resolution_clock::now();

                // Add the time of the epoch to the time variable
                time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                // Print the loss and iteration of the network
                if (i % print_each_iter == 0 and i != 0) {

                    // We calculate an ETA based on the time of the epoch and the number of iterations left.
                    // Dividing time by print_each_iter gives us the time needed for each iteration.
                    // Multiplying the time needed for each iteration by the number of iterations left gives us the ETA.
                    // We also divide eta by 1000000 to convert microseconds to seconds.
                    float eta = (time / print_each_iter) * (this->iterations - i);
                    eta /= 1000000;

                    // We then convert eta to minutes and seconds.
                    // Since we have the time in seconds, we can divide it by 60 to get the minutes
                    // and get the remainder of the division to get the seconds.
                    int minutes = eta / 60;
                    int seconds = (int)eta % 60;

                    // We divide the avg_accuracy by the number of iteration done to get the average accuracy.
                    // This is because we sum 1 to the accuracy counter if the difference between the output of the
                    // network and the target value is less than the threshold delta, so we need to understand, dividing
                    // by the number of iterations, how many times the network has predicted the output correctly.
                    avg_accuracy /= print_each_iter;
                    
                    // Cout the loss and the iteration of the network.
                    // We clear the console using the ANSI escape codes to print the loss and iteration 
                    // on the same line of the console at each iteration.
                    std::cout << "\e[A\r\e[0K"
                    << "Iteration: " << i 
                    << " ------ Loss: " << avg_loss 
                    << " ------ Accuracy: " << avg_accuracy
                    << " ------ ETA: " << minutes << "m:" << seconds << "s"
                    << std::endl;

                    // Push the loss and accuracy to the vectors, time in seconds too.
                    losses.push_back(avg_loss);
                    accuracies.push_back(avg_accuracy/print_each_iter);

                    // We divide the time variable (that contains the sum of the time needed for print_each_iter iterations)
                    // by 1000000 to convert microseconds to seconds. We then push the time to the times vector.
                    times.push_back(time/1000000);

                    // Reset the loss for the next print
                    avg_loss = 0.;
                    avg_accuracy = 0.;
                    time = 0.;

                } 

            }

            // Print the success of the network training
            std::cout << "Training completed!" << std::endl;

        }

        // Predict method: Given an input X, predict the output of the network.
        Matrix<T> predict(Matrix<T> X) {

            // The predict method is used to predict the output of the network given an input X.
            // Since x can contains multiple samples, we loop over the columns of the matrix and
            // forward each sample to the network. The output of the network is then stored in a matrix
            // and returned to the user.

            // Init a matrix to store the predictions of the network
            Matrix<T> predictions(1, X.cols);

            // Loop over the columns of the input matrix
            for (int i=0; i<X.cols; i++) {

                // Sample values from the sample set using the index i
                Matrix<T> x = Matrix<T>(1, 1, X(0, i));

                // Forward pass: compute the activations of the network given an input X, and 
                // retrieve the output of the network.
                Matrix<T> yhat = this->forward(x);

                // Store the output of the network in the predictions matrix
                predictions(0, i) = yhat(0,0);

            }

            // Return the predictions of the network
            return predictions;

        }

        // Save losses, accuracies, hyperparameters and weights/bias of the network to a file
        void save() {

            // Save the hyperparameters of the network to a file
            this->data.savehyper(
                this->layers, 
                this->lr, 
                this->iterations, 
                this->ID, 
                this->afunc,
                "hyperparameters.bin"
            );

            // Save the losses to a csv file
            this->data.save(this->losses, "losses.csv", this->ID);

            // Save the accuracies to a csv file
            this->data.save(this->accuracies, "accuracies.csv", this->ID);

            // Save the times to a csv file
            this->data.save(this->times, "times.csv", this->ID);

            // Save the weights and biases of the network to a file
            // TODO

        }


    private:


        //// ATTRIBUTES ////

        // Those vector of matrices represents the actual attributes of each layers: weights, biases and activations.
        // If this may be confusing, remember that in an efficient implementation of a neural network
        // we don't actually have nodes but just the weights and biases that will be used for computations, 
        // Along with the activations that will be computed during the forward pass, making the use of 
        // an actual node structure unnecessary and inefficient.
        std::vector<Matrix<T>> weights;
        std::vector<Matrix<T>> biases;
        std::vector<Matrix<T>> activations;

        // Vectors of losses and accuracies and the time of N epochs.
        std::vector<float> losses;
        std::vector<float> accuracies;
        std::vector<float> times;

        // Hyperparameters
        float lr;
        int iterations;
        std::vector<size_t> layers;
        
        // Activation function
        std::string afunc;

        // Current loss per epoch
        float loss;

        // Data object to perform some operation on data such as load, save etc
        Data<T> data;

        //// METHODS ////

        // Forward pass: compute the activations of the network given an input X.
        Matrix<T> forward(Matrix<T> X) {

            // At each forward, we need to clear the activations list, because we want to store the activations
            // of the current input.
            this->activations.clear();

            // The forward pass is the process of computing the activations of the network given an input X.
            // Since we do feed-forward the input from the start of the network to the end, 
            // we push the input to the activations list. 
            // Note that the input X it's pushed as the first element of the list, so it will be used as 
            // the activation input to forward to the first layer.
            this->activations.push_back(X);

            // Loop over the layers of the network minus one: this is because we don't count the input layer.
            // I.E: In a network with 3 layers (input, hidden, output), it's obvious that we will forward our 
            // input two times: to the hidden and the output layer!
            for (int i=0; i<this->layers.size()-1; i++) {

                // Now, we proceed to forward the input to the next layer of the network. 
                // What we do is to multiply the input (initially x) by the weights of the current layer
                // and then add the biases. This is the a = Wx + b operation.
                // Note: activation[i] is the input (x) of the current layer.
                Matrix<T> Z = this->activations[i].matmul(this->weights[i]).add(this->biases[i]);

                // After that, we do apply the non linearity to the just calculated Z matrix
                // (apply applies a function to each element of the matrix)
                Matrix<T> A = Z.apply(sigmoid<T>);

                // Then we push back the new matrix containing the layer activation to the vector, 
                // So it will be used in the next iteration as the input, effectively feeding forward the input.
                this->activations.push_back(A);

            }

            // After the loop has ended, we return the activation of the last layer, containing 
            // the output of the network. 
            return this->activations.back();

        }

        // Backward pass: compute the gradients of the weights and biases of the network.
        void backward(Matrix<T> y) {

            // The backward pass is the process of computing the gradients of the weights and biases of the network
            // using the chain rule of calculus. The gradients are used to update the weights and biases of the network
            // using the gradient descent algorithm.

            // Compute the error of the network: the error is the difference between the output of the network
            // and the target output.
            Matrix<T> error = this->activations.back().subtract(y);
            
            // The loss it's simply the mean of the square error for each output unit of the network.
            this->loss = error.square().mean();

            // Now we loop over the layers of the network in reverse order, starting from the last layer.
            // What we want to achieve here, is to compute the gradients of the weights and biases of the network
            // using the chain rule of calculus.
            // For example: Considering the loss function (i.e: Etotal = (1/2(y - yhat)^2) and its derivative, 
            // we can compute the gradient of the weights of the last layer as follows:
            // dEtotal/dW = dEtotal/dyhat * dyhat/dZ * dZ/dW, where:
            // - dEtotal/dyhat is the derivative of the loss function with respect to the output of the network.
            // - dyhat/dZ is the derivative of the activation (output) with respect to the net input of the network.
            // - dZ/dW is the derivative of the net input of the network with respect to the weights of the network.
            for (int i=this->layers.size()-1; i>0; i--) {

                // We compute the gradient with respect to the output of the current layer.
                // The gradient is the derivative of the loss function with respect to the output of the current layer.
                // This operation right here is the ENTIRE chain rule of calculus.
                // That because we are computing the derivative of the loss function with respect to the output of the network.
                // That is already in the form dEtotal/dyhat * dyhat/dZ * dZ/dW! (look above in the forward method
                // what activation[i] is). Note that we're multiplying the error element-wise by the derivative of the
                // activation function of the current layer. That because we want to address each of the output units 
                // for the total error (the error matrix), scaling the error for each output unit.
                Matrix<T> gradient = error.multiply(activations[i].apply(sigmoid_derivative<T>));

                // Scale the gradient by the learning rate.
                gradient = gradient.multiply(this->lr);

                // We also need to explicitly compute the gradient of the weights that connects to the current layer.
                // This is done by multiplying the transpose of the activations of the previous layer by the gradient.
                // Basically, the activations of the previous layer are the net inputs of the current layer.
                // This is the dZ/dW part of the chain rule of calculus, and it's used to update the weights.
                // Activation from previous layer * gradient gives weight update
                Matrix<T> weight_update = this->activations[i-1].transpose().matmul(gradient);

                // We then update the weights of the network by subtracting the computed gradient.
                // We subtract the gradient because we want to minimize the loss function, hence 
                // we need to go in the opposite direction of the gradient to minimize it.
                this->weights[i-1] = this->weights[i-1].subtract(weight_update);

                // We do this also for the bias 
                this->biases[i-1] = this->biases[i-1].subtract(gradient);

                // We then update the error for the next layer by multiplying the gradient of the current layer
                // by the weights that connect the current layer to the previous layer.
                // Note that the gradient is the derivative of the loss function with respect to the output of the 
                // current layer, hence it's the "error" of the current layer.
                // By multiplying the gradient by the previous weights, we are backpropagating the error.
                // print weight rows and cols 
                error = gradient.matmul(this->weights[i-1].transpose());

            }
            
        }

};   




