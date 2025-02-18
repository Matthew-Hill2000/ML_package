#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "layer.h"
#include <vector>

class MaxPoolLayer : public Layer {
  private:
    int kernel_size;
    std::vector<int> input_shape; // [channels, height, width]
    std::vector<int>
        output_shape; // [channels, height/kernel_size, width/kernel_size]

    // Store indices of max elements for backward pass
    std::vector<std::vector<std::vector<std::pair<int, int>>>> max_indices;

  public:
    MaxPoolLayer(const std::vector<int> &input_shape, int kernel_size);

    Tensor forward(const Tensor &input) override;
    Tensor backward(const Tensor &output_gradients) override;
    void update_parameters(double learning_rate) override;
    void set_training_mode(bool mode) override;

    const Tensor &get_output() const override;
    const Tensor &get_input() const override;
    const Tensor &get_gradients() const override;
    const Tensor &get_training_mode() const override;
};

#endif
