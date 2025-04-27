#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cmath>
#include "../../layer.h"
#include "../../../tensor/tensor_view.h"

// The SoftmaxLayer class implements the softmax activation function
// Converts raw scores to normalized probabilities that sum to 1
class SoftmaxLayer : public Layer
{
private:
    std::vector<int> input_shape;  // Dimensions of the input tensor
    Tensor input;                 // Input tensor from forward pass
    Tensor output;                // Output tensor from forward pass
    bool enable_parallelization; // Flag for parallelization

public:
    // Constructor with input shape
    SoftmaxLayer(const std::vector<int> &input_shape);

    // Forward pass - applies softmax normalization
    Tensor forward(const Tensor &input) override;
    
    // Backward pass - computes gradients for backpropagation
    Tensor backward(Tensor &output_gradients) override;
    
    // No weights to update in softmax layer
    void update_weights(double learning_rate) override {};
    
    // Resets gradients to ensure clean state
    void reset_gradients() override;

    // Sets the enable_parallelization flag
    void set_enable_parallelization(bool enable_parallelization);
};

#endif