#include <iostream>

#include "layers/conv_layer.h"
#include "layers/layer.h"
#include "network/network.h"
#include "tensor/tensor_new.h"

int main() {

  double LEARNING_RATE;
  int BATCH_SIZE;
  int NUM_EPOCHS;

  Network net(ConvolutionalLayer(3, 6, 5), MaxPool(2),
              ConvolutionalLayer(6, 16, 5), MaxPool(2), Flatten(),
              FullyConnected(120), FullyConnected(84), FullyConnected(10));

  int N;
  // Dataset is a tensor of shape (N, channels, image_x, image_y)
  Tensor dataset_examples;
  // Dataset labels is a tensor of shape ()
  Tensor dataset_labels;

  for (int epoch{0}; epoch < NUM_EPOCHS; epoch++) {

    int epoch_loss{0};

    for (int example{0}; example < N; example++) {

      Tensor outputs;
      outputs = net.forward(dataset[example])

                    epoch_loss += CrossEntropyLoss(output, datset_labels[i])

                                      net.backward()
    }
  }
}
