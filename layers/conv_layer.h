#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "layer.h"

class ConvolutionalLayer : public Layer {
  private:
    int kernel_size;
    int num_filters;
    std::vector<int> input_shape;
    std::vector<int> output_shape;

    Tensor kernels; // Shape: [num_filters, input_channels, kernel_size,
                    // kernel_size]
    Tensor biases;  // Shape: [num_filters]
    Tensor kernel_gradients;
    Tensor bias_gradients;

  public:
    ConvolutionalLayer(int input_channels, int output_channels,
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
