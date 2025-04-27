#include <iostream>
#include <iomanip>
#include "../../tensor/tensor_view.h"
#include "flatten_layer.h"

// Test function for FlattenLayer
void test_flatten_layer()
{
    std::cout << "\n===== EDUCATIONAL GUIDE: FLATTEN LAYER =====\n";
    std::cout << "PART 1: UNDERSTANDING THE FLATTEN OPERATION\n";
    std::cout << "----------------------------------------\n";
    std::cout << "The flatten layer is a simple but critical component that connects\n";
    std::cout << "convolutional layers to fully connected (dense) layers in a neural network.\n\n";

    std::cout << "Key concepts of flatten layers:\n";
    std::cout << "- Transformation: Converts multi-dimensional data to a single vector\n";
    std::cout << "- Preserves Information: No learnable parameters, just reorganizes data\n";
    std::cout << "- Bridge Component: Connects spatial (CNN) and non-spatial (FC) layers\n";
    std::cout << "- Batch Preservation: Each sample in a batch is flattened independently\n\n";

    // Create a small 3D input with batch size = 2
    std::vector<int> input_shape = {1, 2, 3}; // channels=1, height=2, width=3
    Tensor input({2, 1, 2, 3});               // batch=2, channels=1, height=2, width=3

    std::cout << "Step 1: Create Input Data\n";
    std::cout << "We'll create a small 3D input tensor with sequential values for clarity.\n";
    std::cout << "Shape: [batch_size=2, channels=1, height=2, width=3]\n\n";

    // Fill with sequential values for clear verification
    int value = 1;
    for (int b = 0; b < 2; b++)
    {
        for (int c = 0; c < 1; c++)
        {
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    input[{b, c, i, j}] = value++;
                }
            }
        }
    }

    std::cout << "Input tensor:\n";
    input.print();

    std::cout << "\nIn this 3D input, we have:\n";
    std::cout << "- 2 samples in the batch\n";
    std::cout << "- Each sample has 1 channel, 2 rows, and 3 columns\n";
    std::cout << "- Values 1-6 for first sample, 7-12 for second sample\n\n";

    // Create flatten layer
    std::cout << "Step 2: Create Flatten Layer\n";
    std::cout << "A flatten layer needs to know the input shape (excluding batch dimension).\n";
    std::cout << "Input shape: [" << input_shape[0] << ", " << input_shape[1] << ", " << input_shape[2] << "]\n";
    std::cout << "Expected output shape: [batch_size, " << input_shape[0] * input_shape[1] * input_shape[2] << "]\n\n";

    FlattenLayer flatten_layer(input_shape);

    std::cout << "Step 3: Understand the Flattening Operation\n";
    std::cout << "Flattening works by:\n";
    std::cout << "1. Preserving the batch dimension\n";
    std::cout << "2. Concatenating all other dimensions into a single vector\n\n";

    std::cout << "For our example, the mapping works like this (for first sample):\n";
    std::cout << "input[0,0,0,0] = " << input[{0, 0, 0, 0}] << " → output[0,0]\n";
    std::cout << "input[0,0,0,1] = " << input[{0, 0, 0, 1}] << " → output[0,1]\n";
    std::cout << "input[0,0,0,2] = " << input[{0, 0, 0, 2}] << " → output[0,2]\n";
    std::cout << "input[0,0,1,0] = " << input[{0, 0, 1, 0}] << " → output[0,3]\n";
    std::cout << "input[0,0,1,1] = " << input[{0, 0, 1, 1}] << " → output[0,4]\n";
    std::cout << "input[0,0,1,2] = " << input[{0, 0, 1, 2}] << " → output[0,5]\n\n";

    std::cout << "Mathematical formula for index mapping:\n";
    std::cout << "output[b,i] = input[b, i/(W*H), (i%(W*H))/W, i%W]\n";
    std::cout << "Where b is batch index, W is width, H is height\n\n";

    // Forward pass
    std::cout << "Step 4: Perform Forward Pass (Flattening)\n";
    Tensor output = flatten_layer.forward(input);

    // Output should be [batch_size=2, flattened_size=6]
    std::cout << "Forward pass output (batch=2, flattened_size=6):\n";
    output.print();

    std::cout << "\nVerification: The output contains the same values as the input\n";
    std::cout << "but arranged in a different shape. Notice how the first row contains\n";
    std::cout << "values 1-6 and the second row contains values 7-12.\n\n";

    std::cout << "PART 2: BACKPROPAGATION THROUGH FLATTEN LAYER\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "During backpropagation, we need to:\n";
    std::cout << "1. Take gradients from the next layer (in flattened form)\n";
    std::cout << "2. Reshape them back to the original input dimensions\n";
    std::cout << "3. Pass them to the previous layer\n\n";

    // Create output gradients for backward pass
    std::cout << "Step 5: Create Gradients from Next Layer\n";
    std::cout << "In a real network, these would come from the next layer (typically fully connected).\n";
    std::cout << "For our test, we'll use sequential values scaled by 0.1.\n\n";

    Tensor output_gradients({2, 6}); // batch=2, flattened_size=6

    // Fill with sequential values
    for (int b = 0; b < 2; b++)
    {
        for (int i = 0; i < 6; i++)
        {
            output_gradients[{b, i}] = 0.1 * (b * 6 + i + 1);
        }
    }

    std::cout << "Output gradients (from next layer):\n";
    output_gradients.print();

    std::cout << "\nStep 6: Understand Backward Pass\n";
    std::cout << "The backward pass is essentially the inverse of the forward pass.\n";
    std::cout << "For our example, the mapping works like this (for first sample):\n";
    std::cout << "output_grad[0,0] = " << output_gradients[{0, 0}] << " → input_grad[0,0,0,0]\n";
    std::cout << "output_grad[0,1] = " << output_gradients[{0, 1}] << " → input_grad[0,0,0,1]\n";
    std::cout << "output_grad[0,2] = " << output_gradients[{0, 2}] << " → input_grad[0,0,0,2]\n";
    std::cout << "output_grad[0,3] = " << output_gradients[{0, 3}] << " → input_grad[0,0,1,0]\n";
    std::cout << "output_grad[0,4] = " << output_gradients[{0, 4}] << " → input_grad[0,0,1,1]\n";
    std::cout << "output_grad[0,5] = " << output_gradients[{0, 5}] << " → input_grad[0,0,1,2]\n\n";

    // Backward pass
    std::cout << "Step 7: Perform Backward Pass (Unflattening)\n";
    Tensor input_gradients = flatten_layer.backward(output_gradients);

    std::cout << "Input gradients (reshaped to original input dimensions):\n";
    input_gradients.print();

    std::cout << "\nVerification: The input gradients match the shape of the original input,\n";
    std::cout << "with values arranged according to the reverse mapping.\n\n";

    std::cout << "PART 3: PRACTICAL APPLICATIONS\n";
    std::cout << "----------------------------\n";
    std::cout << "1. CNN to FC Transition: The flatten layer allows connecting\n";
    std::cout << "   convolutional feature maps to fully connected layers\n";
    std::cout << "2. Preserving Batch Processing: Each sample remains independent\n";
    std::cout << "3. No Information Loss: Unlike pooling, flattening preserves all data\n";
    std::cout << "4. No Learnable Parameters: Flattening is a fixed transformation\n\n";

    std::cout << "SUMMARY: THE FLATTEN LAYER\n";
    std::cout << "------------------------\n";
    std::cout << "The flatten layer is a simple reshaping operation that:\n";
    std::cout << "- Transforms multi-dimensional features into a 1D vector\n";
    std::cout << "- Has no learnable parameters (zero training cost)\n";
    std::cout << "- Preserves all information from the input\n";
    std::cout << "- Acts as a critical bridge between different layer types\n";
}

// Main function to run the test
int main()
{
    test_flatten_layer();
    return 0;
}