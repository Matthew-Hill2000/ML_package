#ifndef NETWORK_H
#define NETWORK_H

#include "../layers/layer.h"
#include "../layers/convolutional_layers/convolutional_layer.h"
#include "../layers/flatten_layers/flatten_layer.h"
#include "../layers/fully_connected_layers/fully_connected_layer.h"
#include "../layers/activation_functions/relu/relu.h"
#include "../layers/activation_functions/softmax/softmax.h"
#include "../tensor/tensor_view.h"
#include "../loss_functions/loss.h"
#include "../loss_functions/cross_entropy_loss.h"
#include <memory>
#include <vector>
#include <string>
#include <memory>
#include <vector>

class NetworkBuilder;

// The Network class represents a neural network with layers
// Provides functionality for forward/backward passes and training
class Network
{
private:
  std::vector<Layer *> layers;    // Collection of network layers
  friend class NetworkBuilder;    // Builder has access to private members
  Loss *criterion = nullptr;      // Loss function used for training
  Tensor Loss_gradients;          // Gradients from loss function
  bool enable_parallel = false;   // Flag to enable/disable parallelization

public:
  // Default constructor - creates an empty network
  Network() = default;

  // Constructor with an initial set of layers
  Network(std::vector<Layer *> layers);
  
  // Add a single layer to the network
  void add_layer(Layer *layer);
  
  // Get all layers in the network
  std::vector<Layer *> get_layers() const;

  // Perform forward propagation through all layers
  Tensor forward(const Tensor &input);

  // Perform backward propagation to calculate gradients
  Tensor backward();

  // Update weights using computed gradients
  void update_weights(double learning_rate);

  // Reset all gradients in the network to zero
  void reset_gradients(); 

  // Enable parallelization across all layers
  void enable_parallelization(bool enable_parallelization);

  // Get summary of the network structure
  std::string summary() const;

  // Set the loss function for the network
  void set_criterion(Loss *loss_function);

  // Calculate loss and initialize gradients for backward pass
  double calculate_loss(const Tensor &output, const Tensor &target);
};

// Builder class for constructing networks in a fluent style
class NetworkBuilder
{
private:
  std::vector<Layer *> layers;    // Temporary storage for layers

public:
  // Default constructor
  NetworkBuilder() = default;

  // Destructor to clean up unused layers
  ~NetworkBuilder();

  // Layer addition methods with fluent interface
  NetworkBuilder &addConvLayer(const std::vector<int> &input_shape, int filters, int kernel_size);
  NetworkBuilder &addReluLayer();
  NetworkBuilder &addFlattenLayer(const std::vector<int> &input_shape);
  NetworkBuilder &addFullyConnectedLayer(int input_size, int output_size);
  NetworkBuilder &addSoftmaxLayer(const std::vector<int> &input_shape);

  // Generic method to add any layer type
  NetworkBuilder &addLayer(Layer *layer);

  // Build and return the configured network
  Network build();
};

#endif