#include "network.h"
#include <stdexcept>

Network::Network(std::vector<Layer> layers)
    : layers{layers} {}

void Network::add_layer(Layer layer) { layers.push_back(layer); }

Tensor Network::forward(const Tensor &input) {
  // No layers case
  if (layers.empty()) {
    return input;
  }

  // Pass through first layer
  Tensor current = layers[0]->forward(input);

  // Pass through remaining layers
  for (size_t i = 1; i < layers.size(); ++i) {
    current = layers[i]->forward(current);
  }

  return current;
}

Tensor Network::backward(const Tensor &output_gradients) {
  if (layers.empty()) {
    return output_gradients;
  }

  // Start from the last layer
  Tensor current_gradients = layers.back()->backward(output_gradients);

  // Propagate through remaining layers in reverse
  for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
    current_gradients = layers[i]->backward(current_gradients);
  }

  return current_gradients;
}

void Network::update_parameters(double learning_rate) {
  for (auto &layer : layers) {
    layer->update_parameters(learning_rate);
  }
}

void Network::set_training_mode(bool mode) {
  for (auto &layer : layers) {
    layer->set_training_mode(mode);
  }
}

size_t Network::num_layers() const { return layers.size(); }

std::shared_ptr<Layer> Network::get_layer(size_t index) {
  if (index >= layers.size()) {
    throw std::out_of_range("Layer index out of range");
  }
  return layers[index];
}
