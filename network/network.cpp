#include "../layers/layer.h"
#include "../layers/convolutional_layers/convolutional_layer.h"
#include "../layers/flatten_layers/flatten_layer.h"
#include "../layers/fully_connected_layers/fully_connected_layer.h"
#include "../layers/activation_functions/relu/relu.h"
#include "../layers/activation_functions/softmax/softmax.h"
#include "../tensor/tensor_view.h"
#include "network.h"
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

// Constructor that initializes network with a vector of layers
Network::Network(std::vector<Layer *> layers)
    : layers{layers} {}

// Adds a single layer to the network
void Network::add_layer(Layer *layer) { layers.push_back(layer); }

std::vector<Layer *> Network::get_layers() const { return layers; }

// Performs forward propagation through all network layers
Tensor Network::forward(const Tensor &input)
{
  // No layers case
  if (layers.empty())
  {
    return input;
  }

  // Reset gradients before forward pass 
  reset_gradients();

  // Pass through first layer
  Tensor current = layers[0]->forward(input);
  // Pass through remaining layers
  for (int layer{1}; layer < static_cast<int>(layers.size()); layer++)
  {
    current = layers[layer]->forward(current);
  }

  return current;
}

// Performs backward propagation to calculate gradients
Tensor Network::backward()
{
  if (layers.empty())
  {
    return Loss_gradients;
  }

  // Start from the last layer
  Tensor current_gradients = Loss_gradients;

  // Propagate through layers in reverse
  for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--)
  {
    current_gradients = layers[i]->backward(current_gradients);
  }

  return current_gradients;
}

// Updates network weights using calcuated gradients
void Network::update_weights(double learning_rate)
{
  for (int layer{0}; layer < static_cast<int>(layers.size()); layer++)
  {
    layers[layer]->update_weights(learning_rate);
  }
}

void Network::reset_gradients()
{
  for (int layer{0}; layer < static_cast<int>(layers.size()); layer++)
  {
    layers[layer]->reset_gradients();
  }
}


// Sets the loss function for the network
void Network::set_criterion(Loss *loss_function)
{
  criterion = loss_function;
}

double Network::calculate_loss(const Tensor &output, const Tensor &target)
{
  double loss = criterion->forward(output, target);
  Loss_gradients = criterion->backward(output, target);
  return loss;
}

// Destructor - cleans up any layers not transferred to a Network
NetworkBuilder::~NetworkBuilder()
{
  // Clean up any layers that weren't transferred to a Network
  for (auto layer : layers)
  {
    delete layer;
  }
  layers.clear();
}

// Adds a convolutional layer to the network
NetworkBuilder &NetworkBuilder::addConvLayer(const std::vector<int> &input_shape, int filters, int kernel_size)
{
  layers.push_back(new ConvolutionalLayer(input_shape, filters, kernel_size));
  return *this;
}

// Adds a ReLU activation layer to the network
NetworkBuilder &NetworkBuilder::addReluLayer()
{
  layers.push_back(new ReLU());
  return *this;
}

// Adds a flatten layer to the network
NetworkBuilder &NetworkBuilder::addFlattenLayer(const std::vector<int> &input_shape)
{
  layers.push_back(new FlattenLayer(input_shape));
  return *this;
}

// Adds a fully connected layer to the network
NetworkBuilder &NetworkBuilder::addFullyConnectedLayer(int input_size, int output_size)
{
  layers.push_back(new FullyConnectedLayer(input_size, output_size));
  return *this;
}

// Adds a softmax layer to the network
NetworkBuilder &NetworkBuilder::addSoftmaxLayer(const std::vector<int> &input_shape)
{
  layers.push_back(new SoftmaxLayer(input_shape));
  return *this;
}

// Adds any custom layer to the network
NetworkBuilder &NetworkBuilder::addLayer(Layer *layer)
{
  if (layer == nullptr)
  {
    throw std::invalid_argument("Cannot add nullptr as a layer");
  }
  layers.push_back(layer);
  return *this;
}

// Builds and returns the complete network
Network NetworkBuilder::build()
{
  Network network;

  // Transfer ownership of layers to the network
  for (auto layer : layers)
  {
    network.add_layer(layer);
  }

  // Clear our vector without deleting the layers (network has them now)
  layers.clear();

  return network;
}

// Creates a text summary of the network structure
std::string Network::summary() const
{
  std::stringstream ss;
  ss << "Network Summary:" << std::endl;
  ss << "----------------" << std::endl;
  ss << "Total layers: " << layers.size() << std::endl;

  for (int i{0}; i < static_cast<int>(layers.size()); ++i)
  {
    ss << "Layer " << i + 1 << ": ";

    // Using RTTI to determine layer type
    const auto &layer = layers[i];

    if (dynamic_cast<ConvolutionalLayer *>(layer))
    {
      ss << "Convolutional Layer";
    }
    else if (dynamic_cast<FullyConnectedLayer *>(layer))
    {
      ss << "Fully Connected Layer";
    }
    else if (dynamic_cast<ReLU *>(layer))
    {
      ss << "ReLU Activation";
    }
    else if (dynamic_cast<FlattenLayer *>(layer))
    {
      ss << "Flatten Layer";
    }
    else if (dynamic_cast<SoftmaxLayer *>(layer))
    {
      ss << "Softmax Layer";
    }
    else
    {
      ss << "Unknown Layer Type";
    }

    ss << std::endl;
  }

  return ss.str();
}

// Enables parallelization for all layers in the network
void Network::enable_parallelization(bool enable_parallelization)
{
  for (int i{0}; i < static_cast<int>(layers.size()); ++i)
  {
    layers[i]->set_enable_parallelization(enable_parallelization);
  }
  criterion->set_enable_parallelization(enable_parallelization);
}