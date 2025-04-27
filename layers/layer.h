#ifndef LAYER_H
#define LAYER_H

#include "../tensor/tensor_view.h"
#include <memory>

class Layer
{
protected:
  Tensor input;            // Input tensor from forward pass
  Tensor output;           // Output tensor from forward pass
  Tensor output_gradients; // Gradients from output layer
  Tensor input_gradients;  // Gradients to pass to previous layer
  bool enable_parallelization; // Flag for parallelization

public:
  // Forward pass
  virtual Tensor forward(const Tensor &input) = 0;

  // Backward pass
  virtual Tensor backward(Tensor &output_gradients) = 0;

  // Update parameters using calculated gradients
  virtual void update_weights(double learning_rate) = 0;

  // Reset gradients between epochs/iterations
  virtual void reset_gradients() = 0;

  // Get output from the last forward pass
  virtual const Tensor &get_output() const
  {
    return output;
  }

  virtual void set_enable_parallelization(bool enable_parallelization) = 0;


  virtual ~Layer() {}
};

#endif