#include <iostream>
#include <iomanip>
#include "../../tensor/tensor_view.h"
#include "fully_connected_layer.h"

// Test function for FullyConnectedLayer
void test_fully_connected_layer()
{
    std::cout << "\n===== EDUCATIONAL GUIDE: FULLY CONNECTED LAYER =====\n";
    std::cout << "PART 1: UNDERSTANDING FULLY CONNECTED LAYERS\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Fully connected (or dense) layers are the building blocks of traditional\n";
    std::cout << "neural networks, where every input is connected to every output neuron.\n\n";

    std::cout << "Key concepts of fully connected layers:\n";
    std::cout << "- Linear Transformation: y = Wx + b (weights, inputs, biases)\n";
    std::cout << "- Learnable Parameters: Both weights and biases are learned during training\n";
    std::cout << "- High Connectivity: Each output depends on every input element\n";
    std::cout << "- Parameter Count: input_size × output_size weights + output_size biases\n\n";

    // Create input tensor with batch size = 2, features = 4
    int input_size = 4;
    int output_size = 3;
    Tensor input({2, input_size}); // batch=2, features=4

    std::cout << "Step 1: Create Input Data\n";
    std::cout << "We'll use a simple 2D input tensor with 2 samples, each with 4 features.\n";
    std::cout << "Shape: [batch_size=2, features=" << input_size << "]\n\n";

    // Fill with sequential values
    for (int b = 0; b < 2; b++)
    {
        for (int i = 0; i < input_size; i++)
        {
            input[{b, i}] = b * 10 + i + 1; // batch 0: 1,2,3,4; batch 1: 11,12,13,14
        }
    }

    std::cout << "Input tensor (batch=2, features=4):\n";
    input.print();

    std::cout << "\nSample 1: [1, 2, 3, 4] - 4 feature values\n";
    std::cout << "Sample 2: [11, 12, 13, 14] - 4 feature values\n\n";

    // Create fully connected layer
    std::cout << "Step 2: Create Fully Connected Layer\n";
    std::cout << "A fully connected layer maps " << input_size << " input features to " << output_size << " output features.\n";
    std::cout << "Total parameters: " << input_size << "×" << output_size << " weights + " << output_size
              << " biases = " << (input_size * output_size + output_size) << "\n\n";

    FullyConnectedLayer fc_layer(input_size, output_size);

    // Set custom weights for better demonstration and verification
    std::cout << "Step 3: Set Custom Weights and Biases for Educational Purposes\n";
    std::cout << "Instead of random initialization, we'll use easily recognizable patterns:\n";
    std::cout << "- Weights: Simple incremental values\n";
    std::cout << "- Biases: All set to 1.0\n\n";

    // Create and set custom weights
    Tensor custom_weights({input_size, output_size});
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            double weight_value = 0.1 * (i + j + 1);  // Simple pattern: 0.1, 0.2, 0.3, etc.
            custom_weights[{i, j}] = weight_value;
            fc_layer.set_weight(i, j, weight_value);
        }
    }
    
    // Set all biases to 1.0 for simplicity
    for (int j = 0; j < output_size; j++) {
        fc_layer.set_bias(j, 1.0);
    }

    std::cout << "Custom weights (rows=input_features, columns=output_features):\n";
    fc_layer.print_weights();

    std::cout << "Custom biases (one per output neuron):\n";
    fc_layer.print_biases();
    std::cout << "\n";

    std::cout << "Step 4: Understand the Forward Pass (Linear Transformation)\n";
    std::cout << "For each output neuron j:\n";
    std::cout << "output[j] = bias[j] + sum(weights[i,j] * input[i]) for all inputs i\n\n";

    std::cout << "For example, for the first output neuron and first sample:\n";
    std::cout << "output[0,0] = bias[0] + weights[0,0]*input[0,0] + weights[1,0]*input[0,1] + ...\n\n";

    // Forward pass
    std::cout << "Step 5: Perform Forward Pass\n";
    Tensor output = fc_layer.forward(input);

    // Output should be [batch_size=2, output_size=3]
    std::cout << "Forward pass output (batch=2, outputs=3):\n";
    output.print();

    // Manual calculation for educational purposes
    std::cout << "\nLet's verify a sample calculation manually:\n";
    
    // Calculate output[0,0] manually - first batch, first output neuron
    Tensor weights = fc_layer.get_weights();
    Tensor biases = fc_layer.get_biases();
    
    double manual_output_00 = biases[{0, 0}];
    std::cout << "output[0,0] = bias[0] (" << biases[{0, 0}] << ")";

    for (int i = 0; i < input_size; i++)
    {
        manual_output_00 += weights[{i, 0}] * input[{0, i}];
        std::cout << " + weights[" << i << ",0] (" << weights[{i, 0}]
                  << ") * input[0," << i << "] (" << input[{0, i}] << ")";
    }

    std::cout << "\nExpected result: " << manual_output_00;
    std::cout << "\nActual output[0,0]: " << output[{0, 0}] << "\n";
    
    // Calculate output[0,1] manually - first batch, second output neuron
    double manual_output_01 = biases[{0, 1}];
    std::cout << "\noutput[0,1] = bias[1] (" << biases[{0, 1}] << ")";

    for (int i = 0; i < input_size; i++)
    {
        manual_output_01 += weights[{i, 1}] * input[{0, i}];
        std::cout << " + weights[" << i << ",1] (" << weights[{i, 1}]
                  << ") * input[0," << i << "] (" << input[{0, i}] << ")";
    }

    std::cout << "\nExpected result: " << manual_output_01;
    std::cout << "\nActual output[0,1]: " << output[{0, 1}] << "\n\n";

    std::cout << "PART 2: BACKPROPAGATION IN FULLY CONNECTED LAYERS\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "During backpropagation, we calculate:\n";
    std::cout << "1. Gradients for inputs (to pass to previous layers)\n";
    std::cout << "2. Gradients for weights and biases (to update parameters)\n\n";

    // Create output gradients for backward pass
    std::cout << "Step 6: Create Gradients from Next Layer\n";
    std::cout << "In a real network, these gradients represent how the loss changes\n";
    std::cout << "with respect to each output. For our test, we'll use simple values.\n\n";

    Tensor output_gradients({2, output_size}); // batch=2, outputs=3

    // Fill with gradient values
    for (int b = 0; b < 2; b++)
    {
        for (int i = 0; i < output_size; i++)
        {
            output_gradients[{b, i}] = 0.1 * (b * output_size + i + 1);
        }
    }

    std::cout << "Output gradients (from next layer):\n";
    output_gradients.print();

    std::cout << "\nStep 7: Understand the Backward Pass\n";
    std::cout << "For backpropagation, we need to calculate:\n\n";

    std::cout << "1. Input gradients (for previous layer):\n";
    std::cout << "   grad_input[i] = sum(grad_output[j] * weights[i,j]) for all outputs j\n\n";

    std::cout << "2. Weight gradients (for parameter updates):\n";
    std::cout << "   grad_weight[i,j] = sum(grad_output[b,j] * input[b,i]) for all batches b\n\n";

    std::cout << "3. Bias gradients (for parameter updates):\n";
    std::cout << "   grad_bias[j] = sum(grad_output[b,j]) for all batches b\n\n";

    // Backward pass
    std::cout << "Step 8: Perform Backward Pass\n";
    Tensor input_gradients = fc_layer.backward(output_gradients);

    std::cout << "Input gradients (shape matches input):\n";
    input_gradients.print();

    // Weight and bias gradients
    std::cout << "\nWeight gradients:\n";
    fc_layer.print_weight_gradient();
    
    std::cout << "\nBias gradients:\n";
    fc_layer.print_bias_gradient();

    // Manual verification for input gradients
    std::cout << "\nStep 9: Manual Verification of Gradients\n";
    std::cout << "Let's verify input, weight, and bias gradients manually:\n\n";
    
    // Verify input gradient [0,0]
    double manual_input_grad_00 = 0.0;
    std::cout << "Input gradient[0,0] calculation:\n";
    std::cout << "input_grad[0,0] = ";
    bool first = true;

    for (int j = 0; j < output_size; j++)
    {
        manual_input_grad_00 += output_gradients[{0, j}] * weights[{0, j}];

        if (!first)
            std::cout << " + ";
        std::cout << "output_grad[0," << j << "] (" << output_gradients[{0, j}]
                  << ") * weights[0," << j << "] (" << weights[{0, j}] << ")";
        first = false;
    }

    std::cout << "\nExpected input_grad[0,0]: " << manual_input_grad_00;
    std::cout << "\nActual input_grad[0,0]: " << input_gradients[{0, 0}] << "\n";
    
    // Verify weight gradient [0,0]
    double manual_weight_grad_00 = 0.0;
    std::cout << "\nWeight gradient[0,0] calculation:\n";
    std::cout << "weight_grad[0,0] = ";
    first = true;

    for (int b = 0; b < 2; b++)
    {
        manual_weight_grad_00 += output_gradients[{b, 0}] * input[{b, 0}];

        if (!first)
            std::cout << " + ";
        std::cout << "output_grad[" << b << ",0] (" << output_gradients[{b, 0}]
                  << ") * input[" << b << ",0] (" << input[{b, 0}] << ")";
        first = false;
    }

    std::cout << "\nExpected weight_grad[0,0]: " << manual_weight_grad_00;
    std::cout << "\nActual weight_grad[0,0]: Verify in above weight gradients output\n";
    
    // Verify bias gradient [0]
    double manual_bias_grad_0 = 0.0;
    std::cout << "\nBias gradient[0] calculation:\n";
    std::cout << "bias_grad[0] = ";
    first = true;

    for (int b = 0; b < 2; b++)
    {
        manual_bias_grad_0 += output_gradients[{b, 0}];

        if (!first)
            std::cout << " + ";
        std::cout << "output_grad[" << b << ",0] (" << output_gradients[{b, 0}] << ")";
        first = false;
    }

    std::cout << "\nExpected bias_grad[0]: " << manual_bias_grad_0;
    std::cout << "\nActual bias_grad[0]: Verify in above bias gradients output\n\n";

    // Update weights
    double learning_rate = 0.01;
    std::cout << "Step 10: Update Weights and Biases (Gradient Descent)\n";
    std::cout << "Using learning rate: " << learning_rate << "\n";
    std::cout << "For each parameter: param = param - learning_rate * gradient\n\n";

    // Store original weights for comparison
    std::cout << "Original weights before update:\n";
    fc_layer.print_weights();
    
    // Store original bias for comparison
    std::cout << "Original biases before update:\n";
    fc_layer.print_biases();
    
    // Perform weight update
    fc_layer.update_weights(learning_rate);

    std::cout << "\nUpdated weights after gradient descent step:\n";
    fc_layer.print_weights();

    std::cout << "\nUpdated biases after gradient descent step:\n";
    fc_layer.print_biases();

    // Verify weight update calculation for one weight
    std::cout << "\nStep 11: Precise Weight Update Verification\n";
    double old_weight_00 = weights[{0, 0}];
    double expected_new_weight_00 = old_weight_00 - learning_rate * manual_weight_grad_00;
    
    std::cout << "For weight[0,0] (initially " << old_weight_00 << "):\n";
    std::cout << "  Gradient: " << manual_weight_grad_00 << "\n";
    std::cout << "  Update: " << old_weight_00 << " - " << learning_rate << " * " << manual_weight_grad_00 << "\n";
    std::cout << "  Expected new weight: " << expected_new_weight_00 << "\n";
    std::cout << "  Verify against the updated weight[0,0] above\n\n";

    // Forward pass again to see the effect of the updates
    std::cout << "Step 12: See the Effect of Weight Updates\n";
    Tensor new_output = fc_layer.forward(input);
    std::cout << "Output after weight update:\n";
    new_output.print();
    
    // Compare specific outputs to see the effect of weight updates
    std::cout << "\nComparing specific outputs before and after weight updates:\n";
    std::cout << "Position (0,0):\n";
    std::cout << "  Before: " << output[{0, 0}] << "\n";
    std::cout << "  After: " << new_output[{0, 0}] << "\n";
    std::cout << "  Difference: " << new_output[{0, 0}] - output[{0, 0}] << "\n\n";
    
    std::cout << "Notice how the values have changed due to the updated weights and biases.\n";
    std::cout << "In real network training, this process would repeat for many iterations.\n\n";

    std::cout << "PART 3: ROLE OF FULLY CONNECTED LAYERS IN NEURAL NETWORKS\n";
    std::cout << "----------------------------------------------------\n";
    std::cout << "Fully connected layers are especially important for:\n";
    std::cout << "1. Feature Integration: Combining features learned by earlier layers\n";
    std::cout << "2. Classification: Final layer often outputs class probabilities\n";
    std::cout << "3. Regression: Mapping learned features to continuous outputs\n";
    std::cout << "4. Representation Learning: Transforming data into useful embeddings\n\n";

    std::cout << "Compared to convolutional layers:\n";
    std::cout << "- More parameters (less parameter sharing)\n";
    std::cout << "- No spatial structure preservation\n";
    std::cout << "- Able to model global relationships across all inputs\n";
    std::cout << "- Often used in the later stages of a neural network\n\n";

    std::cout << "SUMMARY: THE FULLY CONNECTED LAYER\n";
    std::cout << "-------------------------------\n";
    std::cout << "The fully connected layer performs a linear transformation that:\n";
    std::cout << "- Maps inputs to outputs via weights and biases\n";
    std::cout << "- Learns complex relationships between all input features\n";
    std::cout << "- Updates parameters via gradient descent during training\n";
    std::cout << "- Forms the backbone of traditional neural networks\n";
    std::cout << "- Often works with activation functions to model non-linear relationships\n";
}

// Main function to run the test
int main()
{
    test_fully_connected_layer();
    return 0;
}