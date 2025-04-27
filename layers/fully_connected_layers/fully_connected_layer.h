#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "../layer.h"
#include <vector>
#include <random>

// The FullyConnectedLayer class implements a standard fully connected neural network layer
// Performs matrix multiplication between inputs and weights, then adds biases
class FullyConnectedLayer : public Layer
{
private:
    int input_size;            // Number of input features
    int output_size;           // Number of output neurons

    Tensor weights;            // Weight matrix (input_size x output_size)
    Tensor biases;             // Bias vector (1 x output_size)

    Tensor weight_gradients;   // Gradients for weight updates
    Tensor bias_gradients;     // Gradients for bias updates
    bool enable_parallelization; // Flag to enable parallelization

public:
    // Constructor with input and output dimensions
    FullyConnectedLayer(int input_size, int output_size);

    // Forward pass - computes output = input * weights + biases
    Tensor forward(const Tensor &input) override;
    
    // Backward pass - computes gradients for weights, biases, and inputs
    Tensor backward(Tensor &output_gradients) override;
    
    // Updates weights and biases using calculated gradients
    void update_weights(double learning_rate) override;
    
    // Resets all gradients to zero
    void reset_gradients() override;

    // Debugging methods to display layer parameters
    void print_weights() const;
    void print_biases() const;
    void print_weight_gradient() const;
    void print_bias_gradient() const;

    // Accessors for layer parameters
    Tensor get_weights() const;
    Tensor get_biases() const;

    // Setters for layer parameters
    void set_weight(int input_idx, int output_idx, double value);
    void set_all_weights(const Tensor &new_weights);
    void set_bias(int output_idx, double value);
    void set_all_biases(const Tensor &new_biases);

    void set_enable_parallelization(bool enable_parallelization);
};

#endif