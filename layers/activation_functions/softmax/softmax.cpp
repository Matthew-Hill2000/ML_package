#include "softmax.h"
#include <omp.h>

// Constructor - initializes layer with specified input shape
SoftmaxLayer::SoftmaxLayer(const std::vector<int> &input_shape)
{
    this->input_shape = input_shape;
}

// Forward pass - converts raw scores to normalized probabilities
Tensor SoftmaxLayer::forward(const Tensor &input)
{

    // Get input dimensions to determine batch size
    std::vector<int> input_dims = input.get_dimensions();
    int batch_size = input_dims[0]; // First dimension is batch size

    // Create output tensor with same dimensions as input (including batch)
    this->input = input;
    this->output= Tensor(input_dims);

    // Process each sample in the batch separately
    #pragma omp parallel for if (enable_parallelization)
    for (int b = 0; b < batch_size; b++)
    {
        // Get this batch sample
        Tensor sample = this->input[b];
        int sample_size = sample.get_n_values();

        // Find the maximum value for numerical stability in this sample
        double max_val = -std::numeric_limits<double>::max();
        for (int i = 0; i < sample_size; i++)
        {
            max_val = std::max(max_val, sample.get_value_direct(i));
        }

        // Calculate sum of exponentials for normalization for this sample
        double exponent_sum = 0.0;
        for (int i = 0; i < sample_size; i++)
        {
            // Subtract max_val for numerical stability (prevents overflow)
            exponent_sum += std::exp(sample.get_value_direct(i) - max_val);
        }

        // Calculate softmax probabilities for this sample
        for (int i = 0; i < sample_size; i++)
        {
            // Safe exponential calculation with numerical stability
            double exp_val = std::exp(sample.get_value_direct(i) - max_val);
            this->output[b].set_value_direct(i, exp_val / exponent_sum);
        }
    }

    return output;
}

// Backward pass - computes gradients for backpropagation
Tensor SoftmaxLayer::backward(Tensor &output_gradients)
{
    // When softmax is followed by cross-entropy loss, the gradient passes directly
    // This simplification only works because cross-entropy's gradient
    // already incorporates the softmax derivative (output - target)

    // Make sure input_gradients has proper dimensions
    if (input_gradients.get_n_values() == 0 ||
        input_gradients.get_dimensions() != input_shape)
    {
        input_gradients = Tensor(input_shape);
    }

    // Pass gradients through (simplified when paired with cross-entropy loss)
    return output_gradients;
}

// Resets gradients to ensure clean state for next iteration
void SoftmaxLayer::reset_gradients()
{
    // Reset gradients only if we have valid input dimensions
    if (!input_shape.empty())
    {
        input_gradients = Tensor(input_shape);
    }
}

// Sets the enable_parallelization flag
void SoftmaxLayer::set_enable_parallelization(bool enable_parallelization)
{
    this->enable_parallelization = enable_parallelization;
}