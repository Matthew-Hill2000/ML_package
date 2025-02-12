#include "conv_layer.h"

ConvolutionalLayer::ConvolutionalLayer(int input_channels, int output_channels,
                                       int kernel_size)
    : num_filters{output_channels},
      kernel_size{kernel_size} {
  kernels = Tensor({num_filters, input_channels, kernel_size, kernel_size});
  ;
}

ConvolutionalLayer::forward(const Tensor &input) override {

  Tensor output = biases;

  for (int i{0}; i < num_filters; i++) {
    for (int j{0}; j < input.dimensions[0]; j++) {
      output[i] = input.cross_correlate(kernels[i][j])
    }
  }
}
