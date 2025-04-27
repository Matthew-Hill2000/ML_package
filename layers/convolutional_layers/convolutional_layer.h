#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "../layer.h"

// The ConvolutionalLayer class implements a standard convolutional neural network layer
// Performs cross-correlation between input and filters, then adds biases
class ConvolutionalLayer : public Layer
{
private:
  Tensor input;            // Input tensor from forward pass
  Tensor output;           // Output tensor from forward pass
  Tensor output_gradients; // Gradients from output layer
  Tensor input_gradients;  // Gradients to pass to previous
  
  int num_filters;           // Number of filters in the layer
  std::vector<int> input_shape;  // Shape: [input_channels, input_height, input_width]
  std::vector<int> output_shape; // Shape: [num_filters, output_height, output_width]
  int kernel_size;           // Size of the kernel (assumes square kernels)
  
  Tensor kernels;            // Shape: [num_filters, input_channels, kernel_size, kernel_size]
  Tensor biases;             // Shape: output_shape

  Tensor kernel_gradients;   // Gradients for filter weights
  Tensor bias_gradients;     // Gradients for biases

  bool enable_parallelization; // Flag to enable parallelization

public:
  // Constructor with input shape, number of filters, and kernel size
  ConvolutionalLayer(const std::vector<int> &input_shape, int output_channels,
                     int kernel_size);
  
  // Forward pass - applies filters to input and adds biases
  Tensor forward(const Tensor &input) override;
  
  // Backward pass - computes gradients for kernels, biases, and inputs
  Tensor backward(Tensor &output_gradients) override;
  
  // Updates kernels and biases using calculated gradients
  void update_weights(double learning_rate) override;
  
  // Resets all gradients to zero
  void reset_gradients() override;

  // Debugging methods to display layer parameters
  void print_kernels();
  void print_biases();
  void print_weight_gadient();
  void print_bias_gradient();

  // Setters and getters for layer parameters
  void set_kernel(int filter_idx, int channel_idx, const Tensor &new_kernel);
  void set_all_kernels(const Tensor &new_kernels);
  Tensor get_kernels() const;

  void set_enable_parallelization(bool enable_parallelization);

  // Virtual destructor
  ~ConvolutionalLayer() override {}
};

#endif