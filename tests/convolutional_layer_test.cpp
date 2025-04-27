#include <iostream>
#include <iomanip>
#include "../../tensor/tensor_view.h"
#include "convolutional_layer.h"

// Test function for ConvolutionalLayer
void test_conv_layer()
{
    std::cout << "\n===== EDUCATIONAL GUIDE: CONVOLUTIONAL LAYERS =====\n";
    std::cout << "PART 1: UNDERSTANDING CONVOLUTIONS\n";
    std::cout << "--------------------------------\n";
    std::cout << "Convolutional layers are fundamental to computer vision tasks.\n";
    std::cout << "They work by sliding filters (kernels) across input data to detect\n";
    std::cout << "spatial patterns like edges, textures, and shapes.\n\n";

    std::cout << "Key concepts of convolutional layers:\n";
    std::cout << "- Kernels: Small matrices of learnable weights\n";
    std::cout << "- Feature maps: Output produced by applying a kernel to the input\n";
    std::cout << "- Convolution operation: Element-wise multiplication and summation\n";
    std::cout << "- Parameter sharing: Same weights applied across the entire input\n";
    std::cout << "- Spatial locality: Focusing on small regions at a time\n\n";

    // Create a small input tensor with batch size = 2, channels = 1, height = 4, width = 4
    std::vector<int> input_shape = {1, 4, 4}; // channels, height, width
    Tensor input({2, 1, 4, 4});               // batch, channels, height, width

    std::cout << "Step 1: Create Input Data\n";
    std::cout << "We'll use a 4x4 input with a recognizable pattern.\n";
    std::cout << "Shape: [batch_size=2, channels=1, height=4, width=4]\n\n";

    // Initialize with distinguishable pattern
    for (int b = 0; b < 2; b++)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                // Create a recognizable pattern: b * 1000 + i * 10 + j
                input[{b, 0, i, j}] = b * 1000 + i * 10 + j;
            }
        }
    }

    std::cout << "Input tensor values (pattern: batch*1000 + row*10 + col):\n";
    input.print();
    std::cout << "Each value follows the pattern: batch*1000 + row*10 + column\n";
    std::cout << "This makes it easy to track which input affects which output.\n\n";

    // Create convolutional layer with 3 filters of size 2x2
    int num_filters = 3;
    int kernel_size = 2;

    std::cout << "Step 2: Create Convolutional Layer\n";
    std::cout << "We'll create a layer with " << num_filters << " filters, each of size "
              << kernel_size << "x" << kernel_size << ".\n";
    std::cout << "Input shape: [batch_size, " << input_shape[0] << ", " << input_shape[1] << ", " << input_shape[2] << "]\n";
    std::cout << "Expected output shape: [batch_size, " << num_filters << ", "
              << input_shape[1] - kernel_size + 1 << ", " << input_shape[2] - kernel_size + 1 << "]\n\n";

    ConvolutionalLayer conv_layer(input_shape, num_filters, kernel_size);

    std::cout << "Step 3: Define Custom Kernels for Educational Purposes\n";
    std::cout << "We'll create 3 filters with special patterns to demonstrate different effects:\n\n";

    // Create custom kernels with easy-to-verify values
    // Filter 1: Identity kernel (detects the original pattern)
    Tensor kernel1({2, 2});
    kernel1[{0, 0}] = 1.0;
    kernel1[{0, 1}] = 0.0;
    kernel1[{1, 0}] = 0.0;
    kernel1[{1, 1}] = 0.0;

    std::cout << "Filter 1: Identity/Top-Left Detector\n";
    std::cout << "| 1.0  0.0 |\n";
    std::cout << "| 0.0  0.0 |\n";
    std::cout << "Purpose: Extracts just the top-left value of each window\n\n";

    // Filter 2: Horizontal edge detector
    Tensor kernel2({2, 2});
    kernel2[{0, 0}] = 1.0;
    kernel2[{0, 1}] = 1.0;
    kernel2[{1, 0}] = -1.0;
    kernel2[{1, 1}] = -1.0;

    std::cout << "Filter 2: Horizontal Edge Detector\n";
    std::cout << "| 1.0  1.0 |\n";
    std::cout << "|-1.0 -1.0 |\n";
    std::cout << "Purpose: Detects horizontal edges by comparing top row to bottom row\n\n";

    // Filter 3: Vertical edge detector
    Tensor kernel3({2, 2});
    kernel3[{0, 0}] = 1.0;
    kernel3[{0, 1}] = -1.0;
    kernel3[{1, 0}] = 1.0;
    kernel3[{1, 1}] = -1.0;

    std::cout << "Filter 3: Vertical Edge Detector\n";
    std::cout << "| 1.0 -1.0 |\n";
    std::cout << "| 1.0 -1.0 |\n";
    std::cout << "Purpose: Detects vertical edges by comparing left column to right column\n\n";

    // Set the custom kernels for each filter
    conv_layer.set_kernel(0, 0, kernel1);
    conv_layer.set_kernel(1, 0, kernel2);
    conv_layer.set_kernel(2, 0, kernel3);

    // Set all biases to zero for easier verification
    Tensor zero_biases({3, 3, 3}); // num_filters, output_height, output_width
    zero_biases = 0.0;
    
    std::cout << "Step 4: Understand the Convolution Operation\n";
    std::cout << "The convolution slides each filter across the input, performing:\n";
    std::cout << "1. Element-wise multiplication between the filter and input window\n";
    std::cout << "2. Summation of all multiplied values\n";
    std::cout << "3. Addition of a bias term (set to zero in this example)\n\n";

    std::cout << "Let's manually calculate one convolution window as an example:\n";
    std::cout << "For batch=0, first window with Filter 1 (Identity):\n";
    std::cout << "Input window:\n";
    std::cout << "| " << input[{0, 0, 0, 0}] << " " << input[{0, 0, 0, 1}] << " |\n";
    std::cout << "| " << input[{0, 0, 1, 0}] << " " << input[{0, 0, 1, 1}] << " |\n\n";

    std::cout << "Filter 1:\n";
    std::cout << "| 1.0  0.0 |\n";
    std::cout << "| 0.0  0.0 |\n\n";

    std::cout << "Calculation:\n";
    std::cout << "(" << input[{0, 0, 0, 0}] << " * 1.0) + (" << input[{0, 0, 0, 1}] << " * 0.0) + ";
    std::cout << "(" << input[{0, 0, 1, 0}] << " * 0.0) + (" << input[{0, 0, 1, 1}] << " * 0.0) = " << input[{0, 0, 0, 0}] << "\n\n";

    // Forward pass
    std::cout << "Step 5: Perform Forward Pass (Convolution)\n";
    Tensor output = conv_layer.forward(input);

    // Output dimensions should be [batch_size=2, filters=3, height=3, width=3]
    std::cout << "Forward pass output [batch=2, filters=3, height=3, width=3]:\n";
    output.print();

    // Verify expected outputs for some key positions
    std::cout << "\nStep 6: Understanding the Output Pattern\n";
    std::cout << "Let's verify some key outputs to understand how each filter works:\n\n";

    // For filter 1 (identity), output should match the top-left value of each convolution window
    std::cout << "Filter 1 (Identity/Top-Left Detector):\n";
    std::cout << "Output[0,0,0,0] = " << output[{0, 0, 0, 0}]
              << " (Expected: " << input[{0, 0, 0, 0}] << ")\n";
    std::cout << "This matches exactly the top-left value of the input window.\n\n";

    // For filter 2 (horizontal edge), verify one position
    double expected_h_edge = (input[{0, 0, 0, 0}] + input[{0, 0, 0, 1}]) - (input[{0, 0, 1, 0}] + input[{0, 0, 1, 1}]);
    std::cout << "Filter 2 (Horizontal Edge Detector):\n";
    std::cout << "Output[0,1,0,0] = " << output[{0, 1, 0, 0}] << "\n";
    std::cout << "Calculation: (top row - bottom row)\n";
    std::cout << "(" << input[{0, 0, 0, 0}] << " + " << input[{0, 0, 0, 1}] << ") - ("
              << input[{0, 0, 1, 0}] << " + " << input[{0, 0, 1, 1}] << ") = " << expected_h_edge << "\n";
    std::cout << "A large absolute value indicates a strong horizontal edge.\n\n";

    // For filter 3 (vertical edge), verify one position
    double expected_v_edge = (input[{0, 0, 0, 0}] + input[{0, 0, 1, 0}]) - (input[{0, 0, 0, 1}] + input[{0, 0, 1, 1}]);
    std::cout << "Filter 3 (Vertical Edge Detector):\n";
    std::cout << "Output[0,2,0,0] = " << output[{0, 2, 0, 0}] << "\n";
    std::cout << "Calculation: (left column - right column)\n";
    std::cout << "(" << input[{0, 0, 0, 0}] << " + " << input[{0, 0, 1, 0}] << ") - ("
              << input[{0, 0, 0, 1}] << " + " << input[{0, 0, 1, 1}] << ") = " << expected_v_edge << "\n";
    std::cout << "A large absolute value indicates a strong vertical edge.\n\n";

    std::cout << "PART 2: BACKWARD PASS & GRADIENT COMPUTATION\n";
    std::cout << "--------------------------------------------\n";
    std::cout << "The backward pass has two main components:\n";
    std::cout << "1. Computing gradients for the inputs (for backpropagation)\n";
    std::cout << "2. Computing gradients for the weights (for parameter updates)\n\n";

    // Create output gradients for backward pass
    std::cout << "Step 7: Create Output Gradients\n";
    std::cout << "In a real network, these would come from the next layer.\n";
    std::cout << "For our test, we'll use a simple pattern: batch + filter + 1.0\n\n";

    Tensor output_gradients({2, 3, 3, 3}); // batch=2, filters=3, height=3, width=3

    // Fill with consistent gradient values for easy verification
    for (int b = 0; b < 2; b++)
    {
        for (int f = 0; f < 3; f++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    // Simple pattern: b + f + 1.0 (constant per filter and batch)
                    output_gradients[{b, f, i, j}] = b + f + 1.0;
                }
            }
        }
    }

    std::cout << "Output gradients (following pattern: batch + filter + 1.0):\n";
    output_gradients.print();

    std::cout << "\nStep 8: Understanding Input Gradient Calculation (Backpropagation)\n";
    std::cout << "The mathematical operation for calculating input gradients can be viewed as a transposed convolution,\n";
    std::cout << "where the kernels are flipped and the input-output channel relationship is reversed.\n";
    std::cout << "\nFor each element in the input tensor, its gradient is the sum of all output gradient values\n";
    std::cout << "it influenced during the forward pass, weighted by the corresponding filter values.\n\n";
    
    std::cout << "Mathematically, for a single element in the input tensor:\n";
    std::cout << "∂Loss/∂Input[b, c_in, i, j] = ∑(c_out, k, l) [∂Loss/∂Output[b, c_out, i-k, j-l] * Kernel[c_out, c_in, k, l]]\n";
    std::cout << "Where the summation runs over all output channels (c_out) and all valid kernel positions (k, l).\n\n";
    
    std::cout << "Let's break down what this means with our example:\n";
    std::cout << "1. Each input value affects multiple output values (depending on the kernel size)\n";
    std::cout << "2. Each of these influences is weighted by the corresponding kernel value\n";
    std::cout << "3. The gradient for an input value is the sum of all these weighted influences\n\n";

    // Backward pass
    Tensor input_gradients = conv_layer.backward(output_gradients);

    std::cout << "Input gradients (shape matches input):\n";
    input_gradients.print();

    // Verify expected gradient values for specific positions
    std::cout << "\nStep 9: Detailed Input Gradient Verification\n";
    std::cout << "Let's verify several positions to understand how gradients combine:\n\n";

    // Position (0,0,0,0) - detailed calculation
    std::cout << "Position (0,0,0,0) Gradient Calculation:\n";
    std::cout << "This input value appears in the following output calculations:\n";
    std::cout << "- Output[0,0,0,0] with Filter 1 weight 1.0 (top-left)\n";
    std::cout << "- Output[0,1,0,0] with Filter 2 weight 1.0 (top-left)\n";
    std::cout << "- Output[0,2,0,0] with Filter 3 weight 1.0 (top-left)\n\n";
    
    std::cout << "Computing the gradient contributions:\n";
    std::cout << "From Filter 1: " << output_gradients[{0, 0, 0, 0}] << " * " << kernel1[{0, 0}] << " = "
              << output_gradients[{0, 0, 0, 0}] * kernel1[{0, 0}] << "\n";
    std::cout << "From Filter 2: " << output_gradients[{0, 1, 0, 0}] << " * " << kernel2[{0, 0}] << " = "
              << output_gradients[{0, 1, 0, 0}] * kernel2[{0, 0}] << "\n";
    std::cout << "From Filter 3: " << output_gradients[{0, 2, 0, 0}] << " * " << kernel3[{0, 0}] << " = "
              << output_gradients[{0, 2, 0, 0}] * kernel3[{0, 0}] << "\n";

    double expected_gradient_00 =
        output_gradients[{0, 0, 0, 0}] * kernel1[{0, 0}] +
        output_gradients[{0, 1, 0, 0}] * kernel2[{0, 0}] +
        output_gradients[{0, 2, 0, 0}] * kernel3[{0, 0}];

    std::cout << "Total expected gradient: " << expected_gradient_00 << "\n";
    std::cout << "Actual input gradient at [0,0,0,0]: " << input_gradients[{0, 0, 0, 0}] << "\n";
    std::cout << "Verification: " << (std::abs(expected_gradient_00 - input_gradients[{0, 0, 0, 0}]) < 1e-6 ? "PASS" : "FAIL") << "\n\n";

    // Position (0,0,1,1) - More complex example with multiple window contributions
    std::cout << "Position (0,0,1,1) Gradient Calculation:\n";
    std::cout << "This position appears in multiple output calculations as different positions in the filter window:\n";
    std::cout << "- As bottom-right in the window for Output[0,x,0,0]\n";
    std::cout << "- As bottom-left in the window for Output[0,x,0,1]\n";
    std::cout << "- As top-right in the window for Output[0,x,1,0]\n";
    std::cout << "- As top-left in the window for Output[0,x,1,1]\n";
    std::cout << "Where x represents each filter index (0, 1, 2)\n\n";
    
    // Check one contribution in detail (when it's at bottom-right of window)
    std::cout << "Let's verify just one position where input[0,0,1,1] is the bottom-right of the window:\n";
    std::cout << "For Output[0,0,0,0] with Filter 1 weight at [1,1]: " 
              << output_gradients[{0, 0, 0, 0}] << " * " << kernel1[{1, 1}] << " = "
              << output_gradients[{0, 0, 0, 0}] * kernel1[{1, 1}] << "\n";
    std::cout << "For Output[0,1,0,0] with Filter 2 weight at [1,1]: " 
              << output_gradients[{0, 1, 0, 0}] << " * " << kernel2[{1, 1}] << " = "
              << output_gradients[{0, 1, 0, 0}] * kernel2[{1, 1}] << "\n";
    std::cout << "For Output[0,2,0,0] with Filter 3 weight at [1,1]: " 
              << output_gradients[{0, 2, 0, 0}] << " * " << kernel3[{1, 1}] << " = "
              << output_gradients[{0, 2, 0, 0}] * kernel3[{1, 1}] << "\n\n";
    
    // Explain the full calculation
    std::cout << "The full gradient calculation would include all instances where this input value appears\n";
    std::cout << "in the sliding window across all filters, which is complex to show manually.\n";
    std::cout << "Actual input gradient at [0,0,1,1]: " << input_gradients[{0, 0, 1, 1}] << "\n\n";

    // Explain the transposed convolution analogy
    std::cout << "Important Insight: Transposed Convolution\n";
    std::cout << "Computing input gradients is mathematically equivalent to a transposed convolution operation.\n";
    std::cout << "The process involves:\n";
    std::cout << "1. Flipping the kernel in both spatial dimensions (180-degree rotation)\n";
    std::cout << "2. Swapping input and output channels (transposing the channel dimensions)\n";
    std::cout << "3. Performing convolution with the output gradients\n\n";
    
    std::cout << "In our implementation, this is handled by the fully_convolve() method, which performs\n";
    std::cout << "a full convolution with proper handling of edge cases where the kernel overlaps the\n";
    std::cout << "boundary of the input tensor.\n\n";

    std::cout << "Step 10: Computing Weight Gradients\n";
    std::cout << "For each kernel weight, we compute how changes affect the loss:\n";
    std::cout << "- Each weight is applied at multiple positions as the filter slides\n";
    std::cout << "- The gradient sums the product of each input value and output gradient\n";
    std::cout << "- This reveals which weights most influence the network's output\n\n";

    std::cout << "Description of kernel gradient computation:\n";
    std::cout << "The gradient for each kernel weight is calculated by:\n";
    std::cout << "1. Identifying all the positions in the output where the kernel was applied.\n";
    std::cout << "2. For each position, multiplying the corresponding output gradient value\n";
    std::cout << "   by the input value that contributed to that position.\n";
    std::cout << "3. Summing up all these contributions across all batches, output positions,\n";
    std::cout << "   and channels to get the total gradient for the kernel weight.\n\n";

    std::cout << "Mathematically:\n";
    std::cout << "∂Loss/∂Kernel[c_out, c_in, k, l] = ∑(b, i, j) [∂Loss/∂Output[b, c_out, i, j] * Input[b, c_in, i+k, j+l]]\n";
    std::cout << "Where the summation runs over all batches (b) and all output positions (i, j).\n\n";

    std::cout << "Computed kernel gradients:\n";
    conv_layer.print_weight_gadient();

    // Verify weight gradients for specific filters and positions
    std::cout << "\nManual Verification of Kernel Gradients\n";
    std::cout << "----------------------------------------\n";
    
    // Verify filter 1, position (0,0)
    std::cout << "Kernel 1, position (0,0) gradient calculation:\n";
    double expected_k1_00 = 0.0;
    
    // For batch 0
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double contrib = output_gradients[{0, 0, i, j}] * input[{0, 0, i+0, j+0}];
            expected_k1_00 += contrib;
            std::cout << "  Batch 0, Position (" << i << "," << j << "): " 
                      << output_gradients[{0, 0, i, j}] << " * " << input[{0, 0, i+0, j+0}] 
                      << " = " << contrib << "\n";
        }
    }
    
    // For batch 1
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double contrib = output_gradients[{1, 0, i, j}] * input[{1, 0, i+0, j+0}];
            expected_k1_00 += contrib;
            std::cout << "  Batch 1, Position (" << i << "," << j << "): " 
                      << output_gradients[{1, 0, i, j}] << " * " << input[{1, 0, i+0, j+0}] 
                      << " = " << contrib << "\n";
        }
    }
    
    std::cout << "Total expected gradient for kernel 1, position (0,0): " << expected_k1_00 << "\n\n";
    
    // Verify filter 2, position (0,1)
    std::cout << "Kernel 2, position (0,1) gradient calculation:\n";
    double expected_k2_01 = 0.0;
    
    // For batch 0
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double contrib = output_gradients[{0, 1, i, j}] * input[{0, 0, i+0, j+1}];
            expected_k2_01 += contrib;
            std::cout << "  Batch 0, Position (" << i << "," << j << "): " 
                      << output_gradients[{0, 1, i, j}] << " * " << input[{0, 0, i+0, j+1}] 
                      << " = " << contrib << "\n";
        }
    }
    
    // For batch 1 (showing just a summary)
    double batch1_sum = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            batch1_sum += output_gradients[{1, 1, i, j}] * input[{1, 0, i+0, j+1}];
        }
    }
    expected_k2_01 += batch1_sum;
    std::cout << "  Batch 1 (all positions): Sum = " << batch1_sum << "\n"; 
    
    std::cout << "Total expected gradient for kernel 2, position (0,1): " << expected_k2_01 << "\n\n";
    
    std::cout << "Step 11: Computing Bias Gradients\n";
    std::cout << "The bias gradients are simpler than weight gradients:\n";
    std::cout << "- Each bias is added once to each output position\n";
    std::cout << "- The gradient is simply the sum of output gradients\n";
    std::cout << "- Biases allow the network to shift the activation function\n\n";
    
    std::cout << "Mathematically:\n";
    std::cout << "∂Loss/∂Bias[c_out, i, j] = ∑(b) [∂Loss/∂Output[b, c_out, i, j]]\n";
    std::cout << "Where the summation runs over all batches (b).\n\n";
    
    std::cout << "Computed bias gradients:\n";
    conv_layer.print_bias_gradient();
    
    // Verify bias gradients for one filter
    std::cout << "\nManual Verification of Bias Gradients\n";
    std::cout << "-------------------------------------\n";
    
    // Verify filter 1, position (0,0)
    std::cout << "Bias for filter 1, position (0,0) gradient calculation:\n";
    double expected_b1_00 = output_gradients[{0, 0, 0, 0}] + output_gradients[{1, 0, 0, 0}];
    std::cout << "  Batch 0: " << output_gradients[{0, 0, 0, 0}] << "\n";
    std::cout << "  Batch 1: " << output_gradients[{1, 0, 0, 0}] << "\n";
    std::cout << "  Total: " << expected_b1_00 << "\n\n";

    // Update weights
    double learning_rate = 0.01;
    std::cout << "Step 12: Updating Weights and Biases (Gradient Descent)\n";
    std::cout << "Using learning rate: " << learning_rate << "\n";
    std::cout << "For each parameter: param = param - learning_rate * gradient\n\n";

    // Store original weights for comparison
    std::cout << "Original kernels before update:\n";
    conv_layer.print_kernels();
    
    // Perform weight update
    conv_layer.update_weights(learning_rate);

    std::cout << "\nUpdated kernels after gradient update:\n";
    conv_layer.print_kernels();

    std::cout << "\nUpdated biases after gradient update:\n";
    conv_layer.print_biases();

    // Verify kernel updates with precise calculations
    std::cout << "\nStep 13: Precise Weight Update Verification\n";
    std::cout << "Let's verify that the weight updates precisely follow gradient descent:\n\n";
    
    // Calculate expected new weight for filter 1, position (0,0)
    std::cout << "For kernel 1, position (0,0) (initially 1.0):\n";
    double old_weight_k1_00 = 1.0; // From our initial setup
    double expected_new_weight_k1_00 = old_weight_k1_00 - learning_rate * expected_k1_00;
    std::cout << "  Original weight: " << old_weight_k1_00 << "\n";
    std::cout << "  Gradient: " << expected_k1_00 << "\n";
    std::cout << "  Update: " << old_weight_k1_00 << " - " << learning_rate << " * " << expected_k1_00 << "\n";
    std::cout << "  Expected new weight: " << expected_new_weight_k1_00 << "\n\n";

    // Forward pass again to see the effect of the updates
    Tensor new_output = conv_layer.forward(input);
    std::cout << "Step 14: See the Effect of Weight Updates\n";
    std::cout << "Output after weight update:\n";
    new_output.print();
    
    // Compare specific outputs to see the effect of weight updates
    std::cout << "\nComparing specific outputs before and after weight updates:\n";
    std::cout << "Position (0,0,0,0):\n";
    std::cout << "  Before: " << output[{0, 0, 0, 0}] << "\n";
    std::cout << "  After: " << new_output[{0, 0, 0, 0}] << "\n";
    std::cout << "  Difference: " << new_output[{0, 0, 0, 0}] - output[{0, 0, 0, 0}] << "\n\n";
    
    std::cout << "Notice how the values have changed due to the updated weights.\n";
    std::cout << "In real network training, this process would repeat for many iterations.\n\n";

    std::cout << "SUMMARY: KEY CONCEPTS OF CONVOLUTIONAL BACKPROPAGATION\n";
    std::cout << "------------------------------------------------\n";
    std::cout << "1. Input Gradient Calculation:\n";
    std::cout << "   - Mathematically equivalent to a transposed convolution operation\n";
    std::cout << "   - Each input's gradient is the sum of all output gradients it influenced\n";
    std::cout << "   - Formally: ∂Loss/∂Input[b,c,i,j] = ∑ [∂Loss/∂Output[b,c_out,i-k,j-l] * Kernel[c_out,c,k,l]]\n\n";
    
    std::cout << "2. Kernel Gradient Calculation:\n";
    std::cout << "   - Each kernel weight's gradient is the sum of products of corresponding\n";
    std::cout << "     output gradients and input activations\n";
    std::cout << "   - Formally: ∂Loss/∂Kernel[c_out,c_in,k,l] = ∑ [∂Loss/∂Output[b,c_out,i,j] * Input[b,c_in,i+k,j+l]]\n\n";
    
    std::cout << "3. Bias Gradient Calculation:\n";
    std::cout << "   - Simply the sum of output gradients for that filter\n";
    std::cout << "   - Formally: ∂Loss/∂Bias[c_out,i,j] = ∑ [∂Loss/∂Output[b,c_out,i,j]]\n\n";
    
    std::cout << "4. Parameter Updates:\n";
    std::cout << "   - Use gradient descent: param = param - learning_rate * gradient\n";
    std::cout << "   - Updates affect forward pass outputs in the next iteration\n\n";
    
    std::cout << "5. Chain Rule in Action:\n";
    std::cout << "   - The output gradients (∂Loss/∂Output) come from the next layer\n";
    std::cout << "   - The input gradients (∂Loss/∂Input) are passed to the previous layer\n";
    std::cout << "   - This chaining of gradients implements the backpropagation algorithm\n\n";
    
    std::cout << "These mathematical operations form the foundation of learning in\n";
    std::cout << "convolutional neural networks, allowing them to automatically adjust\n";
    std::cout << "their parameters to minimize error on training data.\n";
}

// Main function to run the test
int main()
{
    test_conv_layer();
    return 0;
}