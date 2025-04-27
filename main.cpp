#include <iostream> // For input/output
#include <chrono> // For timing

// Include necessary headers for the network and layers
#include "layers/layer.h"
#include "layers/convolutional_layers/convolutional_layer.h"
#include "layers/flatten_layers/flatten_layer.h"
#include "layers/fully_connected_layers/fully_connected_layer.h"
#include "layers/activation_functions/relu/relu.h"
#include "layers/activation_functions/softmax/softmax.h"
#include "tensor/tensor_view.h"

#include "network/dataset.h"
#include "network/network.h"
#include "tensor/tensor_view.h"

#include "loss_functions/cross_entropy_loss.h"

int main()
{
  // Start timing the execution
  auto start_time = std::chrono::high_resolution_clock::now();  

  double LEARNING_RATE{0.1}; // Learning rate for SGD
  int NUM_EPOCHS{10}; // Number of epochs for training
  int BATCH_SIZE{8}; // Define batch size for minibatch SGD

  // Create network with dimensions for MNIST data (1x28x28 images)
  Network net = NetworkBuilder()
                    .addConvLayer({1, 28, 28}, 4, 3) // Input: 1 channel, 28x28 pixels
                    .addReluLayer() // Activation function after convolution
                    .addFlattenLayer({4, 26, 26}) // After 3x3 convolution on 28x28, output is 26x26
                    .addFullyConnectedLayer(26 * 26 * 4, 128) // Flattened size to 128 hidden units
                    .addReluLayer() // Activation function after fully connected layer
                    .addFullyConnectedLayer(128, 10) // 10 output classes (digits 0-9)
                    .addSoftmaxLayer({1, 10}) // Softmax activation for output layer
                    .build(); // Build the network

  
  // Load dataset
  int numImages{200}; // Number of images to load into the dataset
  int numLabels{10}; // Number of labels (10 classes for MNIST for each digit)
  Dataset dataset; 
  dataset.loadFromCSVFiles({"data/images.csv"}, numImages, numLabels, 28);

  // Initialize batch size for minibatch SGD
  dataset.setBatchSize(BATCH_SIZE);

  // Print network and dataset summaries
  std::cout << std::endl;
  std::cout << net.summary() << std::endl;  
  std::cout << dataset.summary() << std::endl;
  std::cout << "Training with batch size: " << BATCH_SIZE << std::endl;
  std::cout << "Number of batches: " << dataset.getNumBatches() << std::endl;

  // Set loss function
  net.set_criterion(new CrossEntropyLoss());
  net.enable_parallelization(true); // Enable parallelization for the network


  double epoch_loss{0.0}; // Variable to store loss for each epoch
  for (int epoch{0}; epoch < NUM_EPOCHS; epoch++)
  {
    
    // std::cout << "Epoch " << epoch + 1 << "/" << NUM_EPOCHS << std::endl;
    
    dataset.shuffle(); // Shuffle dataset at the beginning of each epoch
    

    epoch_loss = 0.0; // Reset epoch loss for each epoch

    // Loop through each batch in the dataset
    for (int batch_idx{0}; batch_idx < dataset.getNumBatches(); batch_idx++)
    {
      Tensor batch_inputs = dataset.getBatchInputs(batch_idx); 
      Tensor batch_labels = dataset.getBatchLabels(batch_idx); 

      Tensor outputs = net.forward(batch_inputs); // Forward pass through the network
      double loss = net.calculate_loss(outputs, batch_labels); // Calculate loss
      net.backward(); // Backward pass through the network                              

      epoch_loss += loss; // Accumulate loss for the epoch
      // std::cout << "Batch " << batch_idx << " processed." << std::endl;
      // std::cout << "Loss: " << loss << std::endl;

      net.update_weights(LEARNING_RATE); // Update weights using SGD
    }

    epoch_loss /= dataset.getNumBatches(); // Average loss over all batches
    std::cout << "Epoch " << epoch + 1 << " loss: " << epoch_loss << std::endl; 
  }

  // End timing and calculate duration
  auto end_time = std::chrono::high_resolution_clock::now();

  // Calculate the duration in different units
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

  // Print timing information
  std::cout << "Total execution time: " << duration_sec.count() << " seconds";
  std::cout << " (" << duration_ms.count() << " milliseconds)" << std::endl;

  return 0;
}