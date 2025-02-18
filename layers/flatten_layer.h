#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "layer.h"
#include <vector>

class FlattenLayer : public Layer {
  private:
    std::vector<int> input_shape;  // Original input shape
    std::vector<int> output_shape; // Flattened output shape [total_elements]

  public:
    FlattenLayer(const std::vector<int> &input_shape);

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
