#include "fully_connected_layer.h"

// Prints weight matrix to console for debugging
void FullyConnectedLayer::print_weights() const
{
    std::cout << "Weights:" << std::endl;
    weights.print();
}

// Prints bias values to console for debugging
void FullyConnectedLayer::print_biases() const
{
    std::cout << "Biases:" << std::endl;
    biases.print();
}

// Prints weight gradient values to console for debugging
void FullyConnectedLayer::print_weight_gradient() const
{
    std::cout << "Weight Gradients:" << std::endl;
    weight_gradients.print();
}

// Prints bias gradient values to console for debugging
void FullyConnectedLayer::print_bias_gradient() const
{
    std::cout << "Bias Gradients:" << std::endl;
    bias_gradients.print();
}

// Returns a copy of the weight matrix
Tensor FullyConnectedLayer::get_weights() const
{
    return weights;
}

// Returns a copy of the bias vector
Tensor FullyConnectedLayer::get_biases() const
{
    return biases;
}

// Sets a single weight value at specified indices
void FullyConnectedLayer::set_weight(int input_idx, int output_idx, double value)
{
    // Check if the indices are valid
    if (input_idx < 0 || input_idx >= input_size)
    {
        throw std::out_of_range("Input index out of range");
    }

    if (output_idx < 0 || output_idx >= output_size)
    {
        throw std::out_of_range("Output index out of range");
    }

    // Set the weight value
    weights[{input_idx, output_idx}] = value;
}

// Replaces entire weight matrix with provided tensor
void FullyConnectedLayer::set_all_weights(const Tensor &new_weights)
{
    // Check if the new weights tensor has the correct dimensions
    auto weight_dims = new_weights.get_dimensions();

    if (weight_dims.size() != 2 ||
        weight_dims[0] != input_size ||
        weight_dims[1] != output_size)
    {
        throw std::invalid_argument(
            "Weights dimensions must be [input_size, output_size]");
    }

    // Replace the entire weights tensor
    this->weights = new_weights.deep_copy();
}

// Sets a single bias value at specified index
void FullyConnectedLayer::set_bias(int output_idx, double value)
{
    // Check if the index is valid
    if (output_idx < 0 || output_idx >= output_size)
    {
        throw std::out_of_range("Output index out of range");
    }

    // Set the bias value
    biases[{0, output_idx}] = value;
}

// Replaces entire bias vector with provided tensor
void FullyConnectedLayer::set_all_biases(const Tensor &new_biases)
{
    // Check if the new biases tensor has the correct dimensions
    auto bias_dims = new_biases.get_dimensions();

    if (bias_dims.size() != 2 ||
        bias_dims[0] != 1 ||
        bias_dims[1] != output_size)
    {
        throw std::invalid_argument(
            "Biases dimensions must be [1, output_size]");
    }

    // Replace the entire biases tensor
    this->biases = new_biases.deep_copy();
}

// Sets the flag to enable or disable parallelization
void FullyConnectedLayer::set_enable_parallelization(bool enable_parallelization)
{
    this->enable_parallelization = enable_parallelization;
}