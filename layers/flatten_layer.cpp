#include "flatten_layer.h"
#include <functional>
#include <numeric>
#include <stdexcept>

FlattenLayer::FlattenLayer(const std::vector<int> &input_shape)
    : input_shape(input_shape) {

  // Calculate total number of elements
  int total_elements = std::accumulate(input_shape.begin(), input_shape.end(),
                                       1, std::multiplies<int>());

  // Output shape is a 1D vector with total_elements
  output_shape = {total_elements};
}

Tensor FlattenLayer::forward(const Tensor &input) {
  auto input_dims = input.get_dimensions();
  if (input_dims != input_shape) {
    throw std::invalid_argument(
        "Input tensor dimensions do not match expected shape");
  }

  this->input = input;

  // Create output tensor with flattened shape
  Tensor output(output_shape);

  // Calculate strides for each dimension
  std::vector<int> strides(input_shape.size());
  strides.back() = 1;
  for (int i = input_shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * input_shape[i + 1];
  }

  // Flatten the input tensor
  int output_idx = 0;
  std::vector<int> indices(input_shape.size(), 0);

  // Iterate through all elements using nested loops
  bool done = false;
  while (!done) {
    // Calculate input index and copy value
    int flat_idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      flat_idx += indices[i] * strides[i];
    }
    output[{output_idx}] = input[flat_idx];
    output_idx++;

    // Increment indices
    for (int i = indices.size() - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < input_shape[i]) {
        break;
      }
      indices[i] = 0;
      if (i == 0) {
        done = true;
      }
    }
  }

  this->output = output;
  return output;
}

Tensor FlattenLayer::backward(const Tensor &output_gradients) {
  // Create gradient tensor with original input shape
  Tensor input_gradients(input_shape);

  // Calculate strides for each dimension
  std::vector<int> strides(input_shape.size());
  strides.back() = 1;
  for (int i = input_shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * input_shape[i + 1];
  }

  // Unflatten the gradients
  int output_idx = 0;
  std::vector<int> indices(input_shape.size(), 0);

  // Iterate through all elements using nested loops
  bool done = false;
  while (!done) {
    // Calculate input index and copy gradient
    int flat_idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      flat_idx += indices[i] * strides[i];
    }
    input_gradients[flat_idx] = output_gradients[{output_idx}];
    output_idx++;

    // Increment indices
    for (int i = indices.size() - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < input_shape[i]) {
        break;
      }
      indices[i] = 0;
      if (i == 0) {
        done = true;
      }
    }
  }

  return input_gradients;
}

void FlattenLayer::update_parameters(double learning_rate) {
  // Flatten layer has no learnable parameters
}

void FlattenLayer::set_training_mode(bool mode) { training_mode = mode; }

const Tensor &FlattenLayer::get_output() const { return output; }

const Tensor &FlattenLayer::get_input() const { return input; }

const Tensor &FlattenLayer::get_gradients() const { return gradients; }

const Tensor &FlattenLayer::get_training_mode() const {
  return Tensor({1}, {training_mode});
}
