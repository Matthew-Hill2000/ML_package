#ifndef LAYER_H
#define LAYER_H

#include "../tensor/tensor_new.h"
#include <memory>

class Layer {
  protected:
    Tensor input;
    Tensor output;
    Tensor gradients;
    bool training_mode;

  public:
    // Forward pass
    virtual Tensor forward(const Tensor &input) = 0;

    // Backward pass
    virtual Tensor backward(const Tensor &output_gradients) = 0;

    // Update parameters using calculated gradients
    virtual void update_parameters(double learning_rate) = 0;

    // Toggle training/inference mode
    virtual void set_training_mode(bool mode) = 0;

    // Getters
    virtual const Tensor &get_output() const = 0;
    virtual const Tensor &get_input() const = 0;
    virtual const Tensor &get_gradients() const = 0;
    virtual const Tensor &get_training_mode() const = 0;

    virtual ~Layer() {}
};

#endif
