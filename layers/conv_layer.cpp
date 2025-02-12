#include "conv_layer.h"
#include <cmath>
ConvolutionalLayer::ConvolutionalLayer(const std::vector<int> &input_shape,
                                       int output_channels, int kernel_size)
    : input_shape{input_shape},
      num_filters{output_channels},
      kernel_size{kernel_size} {

  if (input_shape.size() != 3) {
    throw std::invalid_argument(
        "Input shape must have 3 dimensions (channels, height, width)");
  }

  // Calculate output shape
  output_shape = {
      num_filters,
      input_shape[1] - kernel_size + 1, // output height
      input_shape[2] - kernel_size + 1  // output width
  };

  // Initialize kernels with Xavier/Glorot initialization
  kernels = Tensor({num_filters, input_shape[0], kernel_size, kernel_size});
  double xavier_limit =
      sqrt(6.0 / (input_shape[0] * kernel_size * kernel_size +
                  output_channels * kernel_size * kernel_size));

  // Initialize kernels
  for (int f = 0; f < num_filters; f++) {
    for (int c = 0; c < input_shape[0]; c++) {
      for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
          double rand_val =
              ((double)rand() / RAND_MAX) * 2 * xavier_limit - xavier_limit;
          kernels[{f, c, i, j}] = rand_val;
        }
      }
    }
  }

  // Initialize biases to zero
  biases = Tensor(output_shape);
  biases = 0.0;

  // Initialize gradient tensors
  kernel_gradients =
      Tensor({num_filters, input_shape[0], kernel_size, kernel_size});
  bias_gradients = Tensor(output_shape);
}

Tensor ConvolutionalLayer::forward(const Tensor &input) {
  // Validate input dimensions match what we expect
  auto input_dims = input.get_dimensions();
  if (input_dims != input_shape) {
    throw std::invalid_argument(
        "Input tensor dimensions do not match expected shape");
  }

  // Store input for backward pass
  this->input = input;

  // Initialize output tensor with pre-calculated shape
  Tensor output(output_shape);

  // For each filter
  for (int f = 0; f < num_filters; f++) {
    // For each input channel
    for (int c = 0; c < input_shape[0]; c++) {
      // Get the current input channel and kernel
      Tensor input_channel = input[c];
      Tensor kernel = kernels[f][c];

      // Perform cross correlation and add to output
      Tensor conv_result = input_channel.cross_correlate(kernel);

      // Add to existing values (for channel-wise sum)
      for (int i = 0; i < output_shape[1]; i++) {
        for (int j = 0; j < output_shape[2]; j++) {
          output[{f, i, j}] += conv_result[{i, j}];
        }
      }
    }

    // Add bias for this filter
    for (int i = 0; i < output_shape[1]; i++) {
      for (int j = 0; j < output_shape[2]; j++) {
        output[{f, i, j}] += biases[{f, i, j}];
      }
    }
  }

  // Store output for backward pass
  this->output = output;
  return output;
}

Tensor ConvolutionalLayer::backward(const Tensor &output_gradients) {
  // Initialize gradients
  Tensor input_gradients(input_shape);
  kernel_gradients = Tensor(kernels.get_dimensions());
  bias_gradients = output_gradients; // Direct copy since same shape

  // For each filter
  for (int f = 0; f < num_filters; f++) {
    // For each input channel
    for (int c = 0; c < input_shape[0]; c++) {
      // Compute kernel gradients using cross correlation
      kernel_gradients[f][c] = input[c].cross_correlate(output_gradients[f]);

      // Compute input gradients using full convolution
      input_gradients[c] += output_gradients[f].convolve(kernels[f][c]);
    }
  }

  return input_gradients;
}
