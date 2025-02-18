// maxpool_layer.cpp
#include "max_pool_layer.h"
#include <limits>
#include <stdexcept>

MaxPoolLayer::MaxPoolLayer(const std::vector<int> &input_shape, int kernel_size)
    : input_shape(input_shape),
      kernel_size(kernel_size) {

  if (input_shape.size() != 3) {
    throw std::invalid_argument(
        "Input shape must have 3 dimensions (channels, height, width)");
  }

  // Verify input dimensions are divisible by kernel_size
  if (input_shape[1] % kernel_size != 0 || input_shape[2] % kernel_size != 0) {
    throw std::invalid_argument(
        "Input height and width must be divisible by kernel_size");
  }

  // Calculate output dimensions
  output_shape = {
    // maxpool_layer.cpp
#include "maxpool_layer.h"
#include <limits>
#include <stdexcept>

    MaxPoolLayer::MaxPoolLayer(const std::vector<int> &input_shape,
                               int kernel_size) : input_shape(input_shape),
    kernel_size(kernel_size){

        if (input_shape.size() != 3){throw std::invalid_argument(
            "Input shape must have 3 dimensions (channels, height, width)");
}

// Verify input dimensions are divisible by kernel_size
if (input_shape[1] % kernel_size != 0 || input_shape[2] % kernel_size != 0) {
  throw std::invalid_argument(
      "Input height and width must be divisible by kernel_size");
}

// Calculate output dimensions
output_shape = {
    input_shape[0],               // same number of channels
    input_shape[1] / kernel_size, // output height
    input_shape[2] / kernel_size  // output width
};

// Initialize storage for max indices
max_indices.resize(input_shape[0],
                   std::vector<std::vector<std::pair<int, int>>>(
                       output_shape[1],
                       std::vector<std::pair<int, int>>(output_shape[2])));
}

Tensor MaxPoolLayer::forward(const Tensor &input) {
  auto input_dims = input.get_dimensions();
  if (input_dims != input_shape) {
    throw std::invalid_argument(
        "Input tensor dimensions do not match expected shape");
  }

  this->input = input;
  Tensor output(output_shape);

  // For each channel
  for (int c = 0; c < input_shape[0]; c++) {
    // For each output position
    for (int i = 0; i < output_shape[1]; i++) {
      for (int j = 0; j < output_shape[2]; j++) {
        double max_val = -std::numeric_limits<double>::infinity();
        int max_i = -1, max_j = -1;

        // For each element in the kernel window
        for (int ki = 0; ki < kernel_size; ki++) {
          for (int kj = 0; kj < kernel_size; kj++) {
            int input_i = i * kernel_size + ki;
            int input_j = j * kernel_size + kj;

            double val = input[{c, input_i, input_j}];
            if (val > max_val) {
              max_val = val;
              max_i = input_i;
              max_j = input_j;
            }
          }
        }

        output[{c, i, j}] = max_val;
        max_indices[c][i][j] = {max_i, max_j};
      }
    }
  }

  this->output = output;
  return output;
}

Tensor MaxPoolLayer::backward(const Tensor &output_gradients) {
  Tensor input_gradients(input_shape);
  input_gradients = 0.0; // Initialize to zero

  // For each channel
  for (int c = 0; c < input_shape[0]; c++) {
    // For each output position
    for (int i = 0; i < output_shape[1]; i++) {
      for (int j = 0; j < output_shape[2]; j++) {
        // Get the stored indices of the maximum value
        auto [max_i, max_j] = max_indices[c][i][j];

        // Propagate the gradient only to the position that produced the maximum
        input_gradients[{c, max_i, max_j}] = output_gradients[{c, i, j}];
      }
    }
  }

  return input_gradients;
}

void MaxPoolLayer::update_parameters(double learning_rate) {
  // MaxPool has no learnable parameters
}

void MaxPoolLayer::set_training_mode(bool mode) { training_mode = mode; }

const Tensor &MaxPoolLayer::get_output() const { return output; }

const Tensor &MaxPoolLayer::get_input() const { return input; }

const Tensor &MaxPoolLayer::get_gradients() const { return gradients; }

const Tensor &MaxPoolLayer::get_training_mode() const {
  return Tensor({1}, {training_mode});
}
input_shape[0],                   // same number of channels
    input_shape[1] / kernel_size, // output height
    input_shape[2] / kernel_size  // output width
}
;

// Initialize storage for max indices
max_indices.resize(input_shape[0],
                   std::vector<std::vector<std::pair<int, int>>>(
                       output_shape[1],
                       std::vector<std::pair<int, int>>(output_shape[2])));
}

Tensor MaxPoolLayer::forward(const Tensor &input) {
  auto input_dims = input.get_dimensions();
  if (input_dims != input_shape) {
    throw std::invalid_argument(
        "Input tensor dimensions do not match expected shape");
  }

  this->input = input;
  Tensor output(output_shape);

  // For each channel
  for (int c = 0; c < input_shape[0]; c++) {
    // For each output position
    for (int i = 0; i < output_shape[1]; i++) {
      for (int j = 0; j < output_shape[2]; j++) {
        double max_val = -std::numeric_limits<double>::infinity();
        int max_i = -1, max_j = -1;

        // For each element in the kernel window
        for (int ki = 0; ki < kernel_size; ki++) {
          for (int kj = 0; kj < kernel_size; kj++) {
            int input_i = i * kernel_size + ki;
            int input_j = j * kernel_size + kj;

            double val = input[{c, input_i, input_j}];
            if (val > max_val) {
              max_val = val;
              max_i = input_i;
              max_j = input_j;
            }
          }
        }

        output[{c, i, j}] = max_val;
        max_indices[c][i][j] = {max_i, max_j};
      }
    }
  }

  this->output = output;
  return output;
}

Tensor MaxPoolLayer::backward(const Tensor &output_gradients) {
  Tensor input_gradients(input_shape);
  input_gradients = 0.0; // Initialize to zero

  // For each channel
  for (int c = 0; c < input_shape[0]; c++) {
    // For each output position
    for (int i = 0; i < output_shape[1]; i++) {
      for (int j = 0; j < output_shape[2]; j++) {
        // Get the stored indices of the maximum value
        auto [max_i, max_j] = max_indices[c][i][j];

        // Propagate the gradient only to the position that produced the maximum
        input_gradients[{c, max_i, max_j}] = output_gradients[{c, i, j}];
      }
    }
  }

  return input_gradients;
}

void MaxPoolLayer::update_parameters(double learning_rate) {
  // MaxPool has no learnable parameters
}

void MaxPoolLayer::set_training_mode(bool mode) { training_mode = mode; }

const Tensor &MaxPoolLayer::get_output() const { return output; }

const Tensor &MaxPoolLayer::get_input() const { return input; }

const Tensor &MaxPoolLayer::get_gradients() const { return gradients; }

const Tensor &MaxPoolLayer::get_training_mode() const {
  return Tensor({1}, {training_mode});
}
