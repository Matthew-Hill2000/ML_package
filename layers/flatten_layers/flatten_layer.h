#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "../layer.h"
#include <vector>

// The FlattenLayer class converts multi-dimensional inputs into a 1D vector
// Used to transition from convolutional layers to fully connected layers
class FlattenLayer : public Layer
{
private:
  Tensor input;                // Input tensor from forward pass
  Tensor output;               // Output tensor from forward pass
  Tensor output_gradients;     // Gradients from output layer
  Tensor input_gradients;      // Gradients to pass to previous
  
  std::vector<int> input_shape;  // Original input shape (excluding batch)
  int total_elements;            // Total number of elements in flattened output

  bool enable_parallelization; // Flag for parallelization
public:
  // Constructor that takes the input shape to be flattened
  FlattenLayer(const std::vector<int> &input_shape);

  // Converts multi-dimensional input to flattened 2D output (batch_size x total_elements)
  Tensor forward(const Tensor &input) override;
  
  // Reshapes gradients back to original input dimensions
  Tensor backward(Tensor &output_gradients) override;
  
  // No weights to update in this layer
  void update_weights(double learning_rate) override {};
  
  // Resets gradient storage
  void reset_gradients() override;

  void set_enable_parallelization(bool enable_parallelization);
  

  // Virtual destructor
  ~FlattenLayer() override {}
};

#endif