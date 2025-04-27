#include "fully_connected_layer.h"
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// Constructor - initializes the layer with specified dimensions and Xavier weight initialization
FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size)
{
    // Initialize weights and biases
    this->weights = Tensor({input_size, output_size});
    this->biases = Tensor({1, output_size});

    // Initialize weight gradients and bias gradients
    this->weight_gradients = Tensor({input_size, output_size});
    this->bias_gradients = Tensor({1, output_size});

    // Initialize weights using Xavier initialization
    double xavier_limit = sqrt(6.0 / (input_size + output_size));

    // Initialize weights with random values
    for (int i = 0; i < input_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            double rand_val = ((double)rand() / RAND_MAX) * 2 * xavier_limit - xavier_limit;
            weights[{i, j}] = rand_val;
        }
    }

    // Initialize biases to zero
    for (int i = 0; i < output_size; ++i)
    {
        biases[{0, i}] = 0.0;
    }
}

// Forward pass - computes output = input * weights + biases
Tensor FullyConnectedLayer::forward(const Tensor &input)
{
    // Store a deep copy of the input for backward pass
    this->input = input.deep_copy();

    // Ensure input is contiguous for efficient matrix operations

    // Get input dimensions to determine batch size
    std::vector<int> input_dims = input.get_dimensions();
    int batch_size = input_dims[0]; // First dimension is batch size

    // Create output tensor with appropriate shape
    Tensor output({batch_size, output_size});

    // Perform matrix multiplication
    output = input.matrix_multiplication(weights, enable_parallelization);

    // Add biases to each row of the output
    #pragma omp parallel for collapse(2) if(enable_parallelization)
    for (int b = 0; b < batch_size; b++)
    {
        for (int j = 0; j < output_size; j++)
        {
            output[{b, j}] += biases[{0, j}];
        }
    }

    this->output = output;
    return output;
}

// Backward pass - computes gradients for weights, biases, and inputs
Tensor FullyConnectedLayer::backward(Tensor &output_gradients)
{
    // Store output gradients (create a deep copy)
    this->output_gradients = output_gradients;

    // Calculate weight gradients: input_transposed * output_gradients
    this->weight_gradients = input.transpose().matrix_multiplication(output_gradients, enable_parallelization);

    // Calculate input gradients: output_gradients * weights_transposed
    this->input_gradients = output_gradients.matrix_multiplication(weights.transpose(), enable_parallelization);

    // Copy bias gradients directly from output gradients
    // Bias gradients are the sum of output gradients across the batch
    #pragma omp parallel for if(enable_parallelization)
    for (int j = 0; j < output_size; j++)
    {
        double sum = 0.0;
        for (int b = 0; b < output_gradients.get_dimensions()[0]; b++)
        {
            sum += output_gradients[{b, j}];
        }
        bias_gradients[{0, j}] = sum;
    }

    return input_gradients;
}

// Updates weights and biases using calculated gradients and learning rate
void FullyConnectedLayer::update_weights(double learning_rate)
{
    // Update weights using gradient descent
    #pragma omp parallel for collapse(2) if(enable_parallelization)
    for (int i = 0; i < input_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            weights[{i, j}] -= learning_rate * weight_gradients[{i, j}];
        }
    }

    // Update biases using gradient descent
    #pragma omp parallel for if(enable_parallelization)
    for (int i = 0; i < output_size; ++i)
    {
        biases[{0, i}] -= learning_rate * bias_gradients[{0, i}];
    }
}

// Resets all gradients to zero for next iteration
void FullyConnectedLayer::reset_gradients()
{
    // Reset gradients to zeros
    weight_gradients = 0.0;
    bias_gradients = 0.0;

    // Create input gradients tensor with appropriate shape if input dimensions are known
    if (input.get_n_values() > 0)
    {
        input_gradients = Tensor(input.get_dimensions());
    }
}