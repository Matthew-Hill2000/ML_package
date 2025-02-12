#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "layer.h"

class ConvolutionalLayer : public Layer {
  private:
    int kernel_size; // Size of the kernel
    int num_filters; // Number of filters in the layer
    std::vector<int>
        input_shape; // Shape: [input_channels, input_height, input_width]
    std::vector<int>
        output_shape; // Shape: [num_filters, output_height, output_width]

    Tensor kernels; // Shape: [num_filters, input_channels, kernel_size,
                    // kernel_size]
    Tensor biases;  // Shape: output_shape
    Tensor kernel_gradients; // Shape: kernels.shape
    Tensor bias_gradients;   // Shape: biases.shape

    Tensor input;  // Shape: input_shape
    Tensor output; // Shape: output_shape

  public:
    ConvolutionalLayer(const std::vector<int> &input_shape, int output_channels,
                       int kernel_size);
    Tensor forward(const Tensor &input) override;
    Tensor backward(const Tensor &output_gradients) override;
    void update_parameters(double learning_rate) override;
    void set_training_mode(bool mode) override;

    const Tensor &get_output() const override;
    const Tensor &get_input() const override;
    const Tensor &get_gradients() const override;
    const Tensor &get_training_mode() const override;

    ~ConvolutionalLayer() override {}
};

#endif
