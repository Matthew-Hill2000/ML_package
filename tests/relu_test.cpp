#include <iostream>
#include <iomanip>
#include "../../../tensor/tensor_view.h"
#include "relu.h"

// Test function for ReLU activation
void test_relu_layer()
{
    std::cout << "\n===== EDUCATIONAL GUIDE: ReLU ACTIVATION =====\n";
    std::cout << "PART 1: UNDERSTANDING THE ReLU FUNCTION\n";
    std::cout << "-------------------------------------\n";
    std::cout << "ReLU (Rectified Linear Unit) is one of the most popular activation\n";
    std::cout << "functions in deep learning, known for its simplicity and effectiveness.\n\n";

    std::cout << "Key concepts of ReLU:\n";
    std::cout << "- Mathematical Definition: f(x) = max(0, x)\n";
    std::cout << "- Simple Behavior: Outputs x if x > 0, and 0 if x <= 0\n";
    std::cout << "- Non-linearity: Introduces non-linear behavior essential for neural networks\n";
    std::cout << "- Sparsity: Naturally creates sparse representations (many 0 outputs)\n";
    std::cout << "- Computational Efficiency: Very fast to compute compared to sigmoid/tanh\n\n";

    // Create input tensor with both positive and negative values
    std::cout << "Step 1: Create Input Data with Mixed Values\n";
    std::cout << "We'll create a tensor with both positive and negative values\n";
    std::cout << "to clearly demonstrate ReLU's effect. Shape: [batch=2, channels=1, height=3, width=2]\n\n";

    Tensor input({2, 1, 3, 2}); // batch=2, channels=1, height=3, width=2

    // Set specific values for clear ReLU effect
    // First batch
    input[{0, 0, 0, 0}] = -2.0;
    input[{0, 0, 0, 1}] = 3.0;
    input[{0, 0, 1, 0}] = 0.0;
    input[{0, 0, 1, 1}] = -1.5;
    input[{0, 0, 2, 0}] = 4.0;
    input[{0, 0, 2, 1}] = -0.5;

    // Second batch
    input[{1, 0, 0, 0}] = 1.0;
    input[{1, 0, 0, 1}] = -3.0;
    input[{1, 0, 1, 0}] = -2.5;
    input[{1, 0, 1, 1}] = 0.0;
    input[{1, 0, 2, 0}] = -1.0;
    input[{1, 0, 2, 1}] = 2.0;

    std::cout << "Input tensor with mixed positive/negative values:\n";
    input.print();

    std::cout << "\nNotice the range of values:\n";
    std::cout << "- Positive values (1.0, 2.0, 3.0, 4.0)\n";
    std::cout << "- Negative values (-0.5, -1.0, -1.5, -2.0, -2.5, -3.0)\n";
    std::cout << "- Zero values (0.0)\n\n";

    // Create ReLU layer
    std::cout << "Step 2: Create ReLU Layer\n";
    std::cout << "The ReLU layer applies the function f(x) = max(0, x) element-wise to its input.\n";
    std::cout << "ReLU has no parameters to learn, unlike fully connected or convolutional layers.\n\n";

    ReLU relu_layer;

    // Forward pass
    std::cout << "Step 3: Perform Forward Pass (Apply ReLU)\n";
    std::cout << "Each value in the input will be transformed according to the ReLU function.\n";
    std::cout << "ReLU(x) = x if x > 0, and 0 if x <= 0\n\n";

    Tensor output = relu_layer.forward(input);

    std::cout << "Forward pass output (all negative values become 0):\n";
    output.print();

    // Manual verification of some values
    std::cout << "\nLet's verify some key transformations:\n";
    std::cout << "Input[0,0,0,0] = " << input[{0, 0, 0, 0}] << " (negative) -> Output[0,0,0,0] = " << output[{0, 0, 0, 0}] << " (set to 0)\n";
    std::cout << "Input[0,0,0,1] = " << input[{0, 0, 0, 1}] << " (positive) -> Output[0,0,0,1] = " << output[{0, 0, 0, 1}] << " (unchanged)\n";
    std::cout << "Input[0,0,1,0] = " << input[{0, 0, 1, 0}] << " (zero)     -> Output[0,0,1,0] = " << output[{0, 0, 1, 0}] << " (unchanged)\n";

    std::cout << "\nObservations:\n";
    std::cout << "- All negative values were 'rectified' to 0\n";
    std::cout << "- All positive values passed through unchanged\n";
    std::cout << "- Zero values remained unchanged (0 -> 0)\n\n";

    std::cout << "PART 2: BACKPROPAGATION THROUGH ReLU\n";
    std::cout << "---------------------------------\n";
    std::cout << "During backpropagation, ReLU has a very simple gradient behavior:\n";
    std::cout << "- If the input was positive: Gradient passes through unchanged\n";
    std::cout << "- If the input was negative or zero: Gradient is blocked (set to 0)\n\n";

    std::cout << "Mathematically, the derivative of ReLU is:\n";
    std::cout << "f'(x) = 1 if x > 0\n";
    std::cout << "f'(x) = 0 if x <= 0\n\n";

    // Create output gradients for backward pass
    std::cout << "Step 4: Create Gradients for Backward Pass\n";
    std::cout << "In a real network, these gradients would come from the next layer.\n";
    std::cout << "For our test, we'll use sequential values scaled by 0.1.\n\n";

    Tensor output_gradients({2, 1, 3, 2}); // same shape as output

    // Fill with gradient values
    int value = 1;
    for (int b = 0; b < 2; b++)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                output_gradients[{b, 0, i, j}] = 0.1 * value++;
            }
        }
    }

    std::cout << "Output gradients (coming from the next layer):\n";
    output_gradients.print();

    std::cout << "\nStep 5: Understand Gradient Flow Through ReLU\n";
    std::cout << "When backpropagating through ReLU, gradients are multiplied by the derivative:\n";
    std::cout << "input_gradient = output_gradient * f'(input)\n\n";

    std::cout << "This means:\n";
    std::cout << "- For inputs > 0: input_gradient = output_gradient * 1 = output_gradient\n";
    std::cout << "- For inputs <= 0: input_gradient = output_gradient * 0 = 0\n\n";

    // Backward pass
    std::cout << "Step 6: Perform Backward Pass\n";
    std::cout << "The ReLU will let gradients flow only where the input was positive.\n";
    Tensor input_gradients = relu_layer.backward(output_gradients);

    std::cout << "Input gradients after backward pass:\n";
    input_gradients.print();

    // Verify some gradients manually
    std::cout << "\nLet's verify key gradient flows:\n";
    std::cout << "Position [0,0,0,0]: Input was " << input[{0, 0, 0, 0}] << " (negative)\n";
    std::cout << "  Output gradient: " << output_gradients[{0, 0, 0, 0}] << "\n";
    std::cout << "  Input gradient: " << input_gradients[{0, 0, 0, 0}] << " (blocked because input was negative)\n\n";

    std::cout << "Position [0,0,0,1]: Input was " << input[{0, 0, 0, 1}] << " (positive)\n";
    std::cout << "  Output gradient: " << output_gradients[{0, 0, 0, 1}] << "\n";
    std::cout << "  Input gradient: " << input_gradients[{0, 0, 0, 1}] << " (passes through because input was positive)\n\n";

    std::cout << "PART 3: THE DYING ReLU PROBLEM\n";
    std::cout << "----------------------------\n";
    std::cout << "While ReLU is powerful, it has a known issue called the 'dying ReLU problem':\n";
    std::cout << "- If a neuron's input is consistently negative, its gradient will always be zero\n";
    std::cout << "- This means the neuron can get 'stuck' and stop learning entirely\n";
    std::cout << "- Solutions include careful weight initialization, learning rate tuning,\n";
    std::cout << "  or using variations like Leaky ReLU that allow small gradients for negative inputs\n\n";

    std::cout << "PART 4: ADVANTAGES AND USE CASES\n";
    std::cout << "------------------------------\n";
    std::cout << "ReLU has become the default activation function for many neural networks because:\n\n";

    std::cout << "1. Computational Efficiency:\n";
    std::cout << "   - Simple max(0,x) operation is extremely fast\n";
    std::cout << "   - No expensive exponential calculations like in sigmoid/tanh\n\n";

    std::cout << "2. Mitigates Vanishing Gradient Problem:\n";
    std::cout << "   - Gradient is exactly 1 for all positive inputs (doesn't saturate)\n";
    std::cout << "   - Helps deep networks train more effectively\n\n";

    std::cout << "3. Sparsity:\n";
    std::cout << "   - Natural tendency to create sparse activations (many zeros)\n";
    std::cout << "   - Can be beneficial for representation learning\n\n";

    std::cout << "4. Linear Behavior for Positive Inputs:\n";
    std::cout << "   - Makes optimization easier in many cases\n";
    std::cout << "   - Simplifies convergence compared to sigmoid/tanh\n\n";

    std::cout << "SUMMARY: THE ReLU ACTIVATION\n";
    std::cout << "-------------------------\n";
    std::cout << "ReLU is a simple but powerful activation function that:\n";
    std::cout << "- Applies max(0,x) element-wise to the input\n";
    std::cout << "- Passes gradients only where input was positive\n";
    std::cout << "- Has no parameters to learn\n";
    std::cout << "- Introduces crucial non-linearity to neural networks\n";
    std::cout << "- Has become the default choice for many deep learning architectures\n";
}

// Main function to run the test
int main()
{
    test_relu_layer();
    return 0;
}