#include "flatten_layer.h"
#include <functional>
#include <numeric>
#include <stdexcept>
#include <omp.h>

// Constructor - stores input shape and calculates total elements
FlattenLayer::FlattenLayer(const std::vector<int> &input_shape)
    : input_shape(input_shape)
{
  // Calculate total number of elements in the input (excluding batch dimension)
  int total_elements = 1;
  for (int dim : input_shape)
  {
    total_elements *= dim;
  }

  // Store the total elements for output dimensioning
  this->total_elements = total_elements;
}

// Forward pass - flattens multi-dimensional input to 2D output
Tensor FlattenLayer::forward(const Tensor &input)
{
  // Store input for backward pass (using view, no copy needed)
  this->input = input;

  // Get input dimensions for batch size
  auto input_dims = input.get_dimensions();
  int batch_size = input_dims[0];

  // Create output tensor with shape [batch_size, total_elements]
  this->output = Tensor({batch_size, total_elements});

  // Copy input tensor values to output tensor in flattened form
  #pragma omp parallel for collapse(2) if (enable_parallelization)
  for (int b = 0; b < batch_size; ++b)
  {
    for (int i = 0; i < total_elements; ++i)
    {
      this->output[{b, i}] = this->input[b].get_value_direct(i);
    }
  }

  return output;
}

// Backward pass - reshapes gradients back to original input dimensions
Tensor FlattenLayer::backward(Tensor &output_gradients)
{
  // Store output gradients
  this->output_gradients = output_gradients;

  // Get dimensions for input
  auto input_dims = input.get_dimensions();
  int batch_size = input_dims[0];

  // Create input gradients tensor with original input shape
  this->input_gradients = Tensor(input_dims);

  std::vector<int> strides{this->input_gradients[0].get_strides()};
  std::vector<int> sample_dims(input_dims.begin() + 1, input_dims.end());

  // For batched data, we can directly map from flattened to multidimensional
  #pragma omp parallel for if (enable_parallelization)
  for (int b = 0; b < batch_size; b++)
  {
    // Get flattened gradients for this batch
    TensorView batch_grads = output_gradients[b];

    // Process each element in the flattened tensor
    for (int i = 0; i < total_elements; i++)
    {
      // Calculate multi-dimensional indices from flat index using pre-calculated strides
      std::vector<int> indices = {b}; // Start with batch index
      int remaining = i;
      
      for (size_t dim = 0; dim < sample_dims.size() - 1; dim++) {
        int idx = remaining / strides[dim];
        indices.push_back(idx);
        remaining %= strides[dim];
      }
      
      // Add the last dimension index
      indices.push_back(remaining);
      
      // Copy the gradient value
      input_gradients.set_value(indices, batch_grads.get_value_direct(i));
    }
  }

  return input_gradients;
}

// Resets gradients to ensure clean state for next forward/backward pass
void FlattenLayer::reset_gradients()
{
  // Reset gradients if input shape is known
  if (!input_shape.empty())
  {
    input_gradients = Tensor(input_shape);
  }
}

void FlattenLayer::set_enable_parallelization(bool enable_parallelization)
{
  this->enable_parallelization = enable_parallelization;
}