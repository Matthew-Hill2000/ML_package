#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "network.h"
#include "../layers/layer.h"
#include "../layers/convolutional_layers/convolutional_layer.h"
#include "../layers/flatten_layers/flatten_layer.h"
#include "../layers/fully_connected_layers/fully_connected_layer.h"
#include "../layers/activation_functions/relu/relu.h"
#include "../layers/activation_functions/softmax/softmax.h"
#include "../tensor/tensor_view.h"
#include "../loss_functions/cross_entropy_loss.h"

// Test function for a detailed walkthrough of a neural network forward and backward pass
void test_network_detailed_pass()
{
    std::cout << "\n===== DETAILED NEURAL NETWORK COMPUTATION TRACE =====\n\n";
    
    // Create a smaller network for easier tracing
    std::cout << "CREATING SIMPLIFIED NETWORK FOR DETAILED ANALYSIS\n";
    std::cout << "------------------------------------------------\n";
    std::cout << "Network architecture:\n";
    std::cout << "1. Convolutional Layer: 2 filters, 3x3 kernel\n";
    std::cout << "2. ReLU Activation\n";
    std::cout << "3. Flatten Layer\n";
    std::cout << "4. Fully Connected Layer: 8 neurons\n";
    std::cout << "5. ReLU Activation\n";
    std::cout << "6. Fully Connected Layer: 3 output classes\n";
    std::cout << "7. Softmax Activation\n\n";

    // Define small input dimensions for easier inspection
    std::vector<int> input_shape = {1, 6, 6}; // 1 channel, 6x6 input image
    
    // Create simplified network for detailed inspection
    NetworkBuilder builder;
    Network network = builder
                        .addConvLayer(input_shape, 2, 3)          // 2 filters, 3x3 kernel
                        .addReluLayer()                           // ReLU activation
                        .addFlattenLayer({2, 4, 4})               // Flatten (output after conv)
                        .addFullyConnectedLayer(2 * 4 * 4, 8)     // FC layer with 8 neurons
                        .addReluLayer()                           // ReLU activation
                        .addFullyConnectedLayer(8, 3)             // Output layer with 3 classes
                        .addSoftmaxLayer({3})                     // Softmax activation
                        .build();

    // Set loss function
    CrossEntropyLoss* loss_function = new CrossEntropyLoss();
    network.set_criterion(loss_function);

    // Print network summary
    std::cout << network.summary() << "\n\n";

    // Create a simple input tensor
    std::cout << "STEP 1: INPUT TENSOR PREPARATION\n";
    std::cout << "-------------------------------\n";
    
    Tensor input({1, 1, 6, 6}); // batch_size=1, channels=1, height=6, width=6
    
    // Fill with structured pattern for easier tracking
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            // Simple diagonal gradient pattern
            input[{0, 0, i, j}] = (i + j) / 10.0;
        }
    }
    
    std::cout << "Input image (6x6):\n";
    input.print();
    std::cout << "\n";

    // ==================== FORWARD PASS ====================
    std::cout << "STEP 2: DETAILED FORWARD PASS\n";
    std::cout << "---------------------------\n";
    
    // Get layers for detailed inspection
    auto layers = network.get_layers();
    
    // Initialize intermediate output
    Tensor layer_output = input;
    
    // Layer 1: Convolutional Layer
    std::cout << "LAYER 1: CONVOLUTIONAL LAYER (2 filters, 3x3 kernel)\n";
    auto conv_layer = dynamic_cast<ConvolutionalLayer*>(layers[0]);
    
    // Print initial filter weights
    std::cout << "Filter weights (before training):\n";
    conv_layer->print_kernels();
    
    // Print initial bias values
    std::cout << "Bias values (before training):\n";
    conv_layer->print_biases();
    
    // Forward pass through conv layer
    layer_output = conv_layer->forward(layer_output);
    
    std::cout << "Convolution Calculation Example:\n";
    std::cout << "For filter 0, position (0,0):\n";
    std::cout << "  = bias + sum(filter * input_patch):\n";
    std::cout << "  = Detailed calculation left as an exercise to demonstrate tensor operations\n\n";
    
    std::cout << "Output feature maps (2 filters, 4x4 each):\n";
    layer_output.print();
    
    // Layer 2: ReLU Activation
    std::cout << "LAYER 2: RELU ACTIVATION\n";
    auto relu1 = dynamic_cast<ReLU*>(layers[1]);
    
    // Forward pass through ReLU
    Tensor relu1_input = layer_output; // Save for backprop explanation
    layer_output = relu1->forward(layer_output);
    
    std::cout << "ReLU Operation: max(0, x) for each element\n";
    std::cout << "Output after ReLU (values < 0 set to 0):\n";
    layer_output.print();
    
    // Layer 3: Flatten Layer
    std::cout << "LAYER 3: FLATTEN LAYER\n";
    auto flatten = dynamic_cast<FlattenLayer*>(layers[2]);
    
    // Forward pass through flatten
    Tensor flatten_input = layer_output; // Save for backprop explanation
    layer_output = flatten->forward(layer_output);
    
    std::cout << "Flatten operation: Reshapes 3D tensor to 1D vector (2x4x4 → 32)\n";
    std::cout << "Flattened output (32 elements):\n";
    layer_output.print();
    
    // Layer 4: Fully Connected Layer
    std::cout << "LAYER 4: FULLY CONNECTED LAYER (32→8)\n";
    auto fc1 = dynamic_cast<FullyConnectedLayer*>(layers[3]);
    
    // Print FC1 weights
    std::cout << "FC1 weights and biases (before training):\n";
    fc1->print_weights();
    fc1->print_biases();
    
    // Forward pass through FC1
    Tensor fc1_input = layer_output; // Save for backprop explanation
    layer_output = fc1->forward(layer_output);
    
    std::cout << "FC1 Calculation: output = weights * input + bias\n";
    std::cout << "Output after FC1 (8 neurons):\n";
    layer_output.print();
    
    // Layer 5: ReLU Activation
    std::cout << "LAYER 5: RELU ACTIVATION\n";
    auto relu2 = dynamic_cast<ReLU*>(layers[4]);
    
    // Forward pass through ReLU
    Tensor relu2_input = layer_output; // Save for backprop explanation
    layer_output = relu2->forward(layer_output);
    
    std::cout << "ReLU Operation: max(0, x) for each element\n";
    std::cout << "Output after ReLU (values < 0 set to 0):\n";
    layer_output.print();
    
    // Layer 6: Fully Connected Layer
    std::cout << "LAYER 6: FULLY CONNECTED LAYER (8→3)\n";
    auto fc2 = dynamic_cast<FullyConnectedLayer*>(layers[5]);
    
    // Print FC2 weights
    std::cout << "FC2 weights and biases (before training):\n";
    fc2->print_weights();
    fc2->print_biases();
    
    // Forward pass through FC2
    Tensor fc2_input = layer_output; // Save for backprop explanation
    layer_output = fc2->forward(layer_output);
    
    std::cout << "FC2 Calculation: output = weights * input + bias\n";
    std::cout << "Output after FC2 (3 class logits):\n";
    layer_output.print();
    
    // Layer 7: Softmax Activation
    std::cout << "LAYER 7: SOFTMAX ACTIVATION\n";
    auto softmax = dynamic_cast<SoftmaxLayer*>(layers[6]);
    
    // Forward pass through softmax
    Tensor softmax_input = layer_output; // Save for backprop explanation
    layer_output = softmax->forward(layer_output);
    
    std::cout << "Softmax Calculation: exp(x_i) / sum(exp(x_j))\n";
    std::cout << "Final output (class probabilities):\n";
    layer_output.print();
    
    // Create target data for loss calculation
    std::cout << "STEP 3: LOSS CALCULATION\n";
    std::cout << "---------------------\n";
    
    Tensor target({1, 3}); // batch_size=1, num_classes=3
    target = 0.0;
    target[{0, 1}] = 1.0; // Set class 1 as the target
    
    std::cout << "Target one-hot vector:\n";
    target.print();
    
    // Calculate loss
    double loss = loss_function->forward(layer_output, target);
    std::cout << "Cross-Entropy Loss: " << loss << "\n\n";
    std::cout << "Loss Calculation: -sum(target * log(prediction))\n";
    std::cout << "Expected class is 1, probability assigned: " << layer_output[{0, 1}] << "\n";
    std::cout << "Loss = -log(" << layer_output[{0, 1}] << ") = " << -std::log(layer_output[{0, 1}]) << "\n\n";
    
    // ==================== BACKWARD PASS ====================
    std::cout << "STEP 4: DETAILED BACKWARD PASS\n";
    std::cout << "----------------------------\n";
    
    // Compute output gradients
    std::cout << "Computing initial gradients from loss function...\n";
    Tensor output_gradients = loss_function->backward(layer_output, target);
    
    std::cout << "Initial gradients (dL/dOutput):\n";
    output_gradients.print();
    
    // Layer 7 Backward: Softmax
    std::cout << "LAYER 7 BACKWARD: SOFTMAX\n";
    Tensor softmax_gradients = softmax->backward(output_gradients);
    
    std::cout << "Softmax gradient calculation already combined with Cross-Entropy.\n";
    std::cout << "Gradient after softmax (dL/dFC2_output):\n";
    softmax_gradients.print();
    
    // Layer 6 Backward: FC2
    std::cout << "LAYER 6 BACKWARD: FC2\n";
    Tensor fc2_gradients = fc2->backward(softmax_gradients);
    
    std::cout << "FC2 weight gradients (average partial derivatives):\n";
    fc2->print_weight_gradient();
    
    std::cout << "FC2 bias gradients (average partial derivatives):\n";
    fc2->print_bias_gradient();
    
    std::cout << "Gradient explanation: Each weight gradient is input[i] * output_grad[j]\n";
    std::cout << "Example: For weight[0,0], gradient = input[0] * output_grad[0]\n\n";
    
    std::cout << "Gradients after FC2 (dL/dReLU2_output):\n";
    fc2_gradients.print();
    
    // Layer 5 Backward: ReLU2
    std::cout << "LAYER 5 BACKWARD: RELU2\n";
    Tensor relu2_gradients = relu2->backward(fc2_gradients);
    
    std::cout << "ReLU gradient: 1 if input > 0, else 0\n";
    std::cout << "Gradients after ReLU2 (dL/dFC1_output):\n";
    relu2_gradients.print();
    
    // Layer 4 Backward: FC1
    std::cout << "LAYER 4 BACKWARD: FC1\n";
    Tensor fc1_gradients = fc1->backward(relu2_gradients);
    
    std::cout << "FC1 weight gradients (average partial derivatives):\n";
    fc1->print_weight_gradient();
    
    std::cout << "FC1 bias gradients (average partial derivatives):\n";
    fc1->print_bias_gradient();
    
    std::cout << "Gradients after FC1 (dL/dFlatten_output):\n";
    fc1_gradients.print();
    
    // Layer 3 Backward: Flatten
    std::cout << "LAYER 3 BACKWARD: FLATTEN\n";
    Tensor flatten_gradients = flatten->backward(fc1_gradients);
    
    std::cout << "Flatten backward: Reshape 1D gradient back to original 3D shape\n";
    std::cout << "Gradients after Flatten reshape (dL/dReLU1_output):\n";
    flatten_gradients.print();
    
    // Layer 2 Backward: ReLU1
    std::cout << "LAYER 2 BACKWARD: RELU1\n";
    Tensor relu1_gradients = relu1->backward(flatten_gradients);
    
    std::cout << "ReLU gradient: 1 if input > 0, else 0\n";
    std::cout << "Gradients after ReLU1 (dL/dConv_output):\n";
    relu1_gradients.print();
    
    // Layer 1 Backward: Convolution
    std::cout << "LAYER 1 BACKWARD: CONVOLUTION\n";
    Tensor conv_gradients = conv_layer->backward(relu1_gradients);
    
    std::cout << "Convolution kernel gradients:\n";
    conv_layer->print_weight_gadient();
    
    std::cout << "Convolution bias gradients:\n";
    conv_layer->print_biases();
    
    std::cout << "Gradients after Convolution (dL/dInput):\n";
    conv_gradients.print();
    
    // ==================== WEIGHT UPDATE ====================
    std::cout << "STEP 5: WEIGHT UPDATE DEMONSTRATION\n";
    std::cout << "--------------------------------\n";
    
    double learning_rate = 0.01;
    std::cout << "Learning rate: " << learning_rate << "\n\n";
    
    // Update FC2 weights (demonstration)
    std::cout << "FC2 weights before update (sample):\n";
    std::cout << "Weight[0,0]: " << fc2->get_weights()[{0, 0}] << "\n";
    
    // Calculate expected new weight
    double grad_w00 = softmax_gradients[{0, 0}] * fc2_input[{0, 0}];
    double new_w00 = fc2->get_weights()[{0, 0}] - learning_rate * grad_w00;
    
    std::cout << "Gradient for weight[0,0]: " << grad_w00 << "\n";
    std::cout << "Expected new weight[0,0]: " << fc2->get_weights()[{0, 0}] << " - " 
              << learning_rate << " * " << grad_w00 << " = " << new_w00 << "\n\n";
    
    // Update Conv weights (demonstration)
    std::cout << "Conv filter 0 before update (sample):\n";
    std::cout << "Filter[0,0,0,0]: " << conv_layer->get_kernels()[{0, 0, 0, 0}] << "\n";
    
    // Update all weights
    std::cout << "Updating all weights across the network...\n";
    network.update_weights(learning_rate);
    
    // Show updated weights
    std::cout << "FC2 weights after update (sample):\n";
    std::cout << "Weight[0,0]: " << fc2->get_weights()[{0, 0}] << "\n";
    
    std::cout << "Conv filter 0 after update (sample):\n";
    std::cout << "Filter[0,0,0,0]: " << conv_layer->get_kernels()[{0, 0, 0, 0}] << "\n\n";
    
    std::cout << "STEP 6: VALIDATE UPDATE EFFECT WITH SECOND FORWARD PASS\n";
    std::cout << "---------------------------------------------------\n";
    
    // Perform another forward pass
    Tensor new_output = network.forward(input);
    double new_loss = loss_function->forward(new_output, target);
    
    std::cout << "New network output:\n";
    new_output.print();
    
    std::cout << "Original loss: " << loss << "\n";
    std::cout << "New loss after one update: " << new_loss << "\n";
    
    // Calculate and show improvement
    double loss_change = loss - new_loss;
    std::cout << "Loss change: " << loss_change;
    if (loss_change > 0) {
        std::cout << " (improved)\n";
    } else {
        std::cout << " (not improved - may need hyperparameter adjustment)\n";
    }
    
    // Clean up
    delete loss_function;
}

// Main function to run the test
int main()
{
    test_network_detailed_pass();
    return 0;
}