#ifndef NETWORK_H
#define NETWORK_H

#include "../layers/layer.h"
#include "../tensor/tensor_new.h"
#include <memory>
#include <vector>

class Network {
  private:
    std::vector<Layer *> layers;

  public:
    // Default constructor
    Network() = default;

    Network(std::vector<Layer *> layers);
    // Add a layer to the network
    void add_layer(Layer *layer);

    // Forward pass through the network
    Tensor forward(const Tensor &input);

    // Backward pass through the network
    Tensor backward(const Tensor &output_gradients);

    // Update parameters of all layers
    void update_parameters(double learning_rate);

    // Set training mode for all layers
    void set_training_mode(bool mode);

    // Get number of layers
    size_t num_layers() const;

    // Access a specific layer
    std::shared_ptr<Layer> get_layer(size_t index);
};

#endif
