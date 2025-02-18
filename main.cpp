#include <iostream>

#include "layers/conv_layer.h"
#include "layers/layer.h"
#include "network/network.h"
#include "tensor/tensor_new.h"

int main() {

  double LEARNING_RATE{0.001};
  int BATCH_SIZE{1};
  int NUM_EPOCHS{10};

  Network net;
  ConvolutionalLayer Conv_1(std::vector<int>{1, 28, 28}, 6, 5);
  MaxPool Max_Pool_1(2);
  ConvolutionalLayer Conv_2(std::vector<int>{6, 12, 12}, 16, 5);
  MaxPoool Max_Pool_2(2);
  Flatten flatten;
  FullyConnected Fully_Connected_1(16 * 4 * 4, 120);
  FullyConnected Fully_Connected_2(120, 84);
  FullyConnected Fully_Connected_3(84, 10);
  Softmax Softmax_1;

  // Add layers to the Network
  net.add_layer(*Conv_1);
  net.add_layer(*Max_Pool_1);
  net.add_layer(*Conv_2);
  net.add_layer(*Max_Pool_2);
  net.add_layer(*flatten());
  net.add_layer(*Fully_Connected_1);
  net.add_layer(*Fully_Connected_2);
  net.add_layer(*Fully_Connected_3);
  net.add_layer(*Softmax_1);

  net.set_training_mode(true);

  for (int epoch{0}; epoch < NUM_EPOCHS; epoch++) {
    double epoch_loss{0.0};

    Tensor input({1, 1, 28, 28});
    Tensor target({10});

    Tensor output = net.forward(input);

    double loss = cross_entropy_loss(output, target);
    epoch_loss += loss;

    Tensor loss_grad = cross_entropy_loss_gradient(output, target);
    net.backward(loss_grad);

    net.update_parameters(LEARNING_RATE);

    std::cout << "Epoch " << epoch + 1 << "/" << NUM_EPOCHS
              << ", loss: " << epoch_loss << std::endl;
  }

  return 0;
}
