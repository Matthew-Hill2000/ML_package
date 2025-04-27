#include "relu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

// Forward pass - applies the ReLU function element-wise
Tensor ReLU::forward(const Tensor &input)
{
    // Store the input using deep_copy to ensure it's preserved for backprop
    this->input = input;
    this->input_shape = input.get_dimensions();
    this->output_shape = input_shape;

    // Create a new tensor for output with the same dimensions
    this->output = Tensor(output_shape);

    // Apply the ReLU function (max(0,x)) to each element of the input tensor
    #pragma omp parallel for if (enable_parallelization)
    for (int i =0; i < this->input.get_n_values(); i++)
    {
        double val = input.get_value_direct(i);
        this->output.set_value_direct(i, std::max(0.0, val));
    }

    // Store and return the output
    return output;
}

// Backward pass - computes gradients for backpropagation
Tensor ReLU::backward(Tensor &output_gradients)
{
    // Make a deep copy of output gradients to prevent shared reference issues

    // Create a new tensor for gradient propagation
    this->input_gradients = Tensor(input_shape);

    // Apply ReLU gradient: derivative is 1 for positive inputs, 0 otherwise
    #pragma omp parallel for if (enable_parallelization)
    for (int i = 0; i < this->input.get_n_values(); i++)
    {    

        double input_val{input.get_value_direct(i)};
        double grad_val{output_gradients.get_value_direct(i)};

        this->input_gradients.set_value_direct(i, grad_val * (input_val > 0 ? 1.0 : 0.0));
    }

    // Store and return the input gradients
    return input_gradients;
}

// Resets gradients to ensure clean state for next iteration
void ReLU::reset_gradients()
{
    // Reset gradients only if we have valid input dimensions
    this->input_gradients = Tensor(this->input_shape);
}

// Sets the enable_parallelization flag
void ReLU::set_enable_parallelization(bool enable_parallelization)
{
    this->enable_parallelization = enable_parallelization;
}