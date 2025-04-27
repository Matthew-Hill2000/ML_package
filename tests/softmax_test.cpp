#include <iostream>
#include <iomanip>
#include <cmath>
#include "../../../tensor/tensor_view.h"
#include "softmax.h"
#include "../../../loss_functions/cross_entropy_loss.h"

// Test function for SoftmaxLayer with Cross Entropy Loss
void test_softmax_layer()
{
    std::cout << "\n===== EDUCATIONAL GUIDE: SOFTMAX AND CROSS ENTROPY LOSS =====\n";

    std::cout << "PART 1: UNDERSTANDING SOFTMAX\n";
    std::cout << "--------------------------\n";
    std::cout << "Softmax converts raw model outputs (logits) into a probability distribution.\n";
    std::cout << "Mathematical formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j\n\n";
    std::cout << "Properties of softmax:\n";
    std::cout << "- Outputs are between 0 and 1\n";
    std::cout << "- Outputs sum to 1 (representing a valid probability distribution)\n";
    std::cout << "- Preserves the rank order of inputs (higher inputs lead to higher probabilities)\n";
    std::cout << "- Amplifies differences between inputs (due to the exponential function)\n\n";

    // Create input tensor with batch size = 2, features = 4
    std::vector<int> input_shape = {4}; // 4 features
    Tensor input({2, 4});               // batch=2, features=4

    // Set values with clear differences for softmax visualization
    input[{0, 0}] = 1.0;  // Example 1, Class 1
    input[{0, 1}] = 2.0;  // Example 1, Class 2
    input[{0, 2}] = 3.0;  // Example 1, Class 3
    input[{0, 3}] = 4.0;  // Example 1, Class 4
    input[{1, 0}] = 0.5;  // Example 2, Class 1
    input[{1, 1}] = 0.5;  // Example 2, Class 2
    input[{1, 2}] = 10.0; // Example 2, Class 3 (much larger - should dominate)
    input[{1, 3}] = 0.5;  // Example 2, Class 4

    std::cout << "Step 1: Raw model outputs (logits) for 2 examples, 4 classes each:\n";
    input.print();

    std::cout << "\nExample 1 demonstrates a gradual increase across classes.\n";
    std::cout << "Example 2 has one much larger value (10.0) to demonstrate softmax's behavior with outliers.\n\n";

    // Create softmax layer
    SoftmaxLayer softmax_layer(input_shape);

    // Forward pass through softmax
    Tensor softmax_output = softmax_layer.forward(input);

    std::cout << "Step 2: Applying Softmax Transformation\n";
    std::cout << "Let's manually calculate softmax for Example 1:\n";

    // Manual calculation for first example
    double exp_sum = std::exp(1.0) + std::exp(2.0) + std::exp(3.0) + std::exp(4.0);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  exp(1.0) = " << std::exp(1.0) << "\n";
    std::cout << "  exp(2.0) = " << std::exp(2.0) << "\n";
    std::cout << "  exp(3.0) = " << std::exp(3.0) << "\n";
    std::cout << "  exp(4.0) = " << std::exp(4.0) << "\n";
    std::cout << "  Sum of exponentials = " << exp_sum << "\n\n";

    std::cout << "  softmax(1.0) = exp(1.0) / sum = " << std::exp(1.0) / exp_sum << "\n";
    std::cout << "  softmax(2.0) = exp(2.0) / sum = " << std::exp(2.0) / exp_sum << "\n";
    std::cout << "  softmax(3.0) = exp(3.0) / sum = " << std::exp(3.0) / exp_sum << "\n";
    std::cout << "  softmax(4.0) = exp(4.0) / sum = " << std::exp(4.0) / exp_sum << "\n\n";

    std::cout << "Softmax output from our implementation:\n";
    softmax_output.print();

    // Verify outputs sum to 1 for each batch
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < 4; i++)
    {
        sum1 += softmax_output[{0, i}];
        sum2 += softmax_output[{1, i}];
    }
    std::cout << "Verification: Probabilities sum to 1\n";
    std::cout << "  Sum of Example 1 probabilities: " << sum1 << "\n";
    std::cout << "  Sum of Example 2 probabilities: " << sum2 << "\n\n";

    std::cout << "Numerical Stability Note: Softmax implementations often subtract the max value\n";
    std::cout << "before exponentiating to prevent overflow: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))\n";
    std::cout << "This doesn't change the mathematical result but prevents numerical issues.\n\n";

    std::cout << "Observations:\n";
    std::cout << "- Example 1: Gradual increase in probabilities matching gradual increase in logits\n";
    std::cout << "- Example 2: Class 3 dominates with probability near 1.0 due to much larger input\n";
    std::cout << "  This demonstrates how softmax amplifies differences.\n\n";

    std::cout << "PART 2: CROSS ENTROPY LOSS\n";
    std::cout << "------------------------\n";
    std::cout << "Cross entropy measures the difference between two probability distributions:\n";
    std::cout << "- The predicted distribution from our model (softmax outputs)\n";
    std::cout << "- The target distribution (usually one-hot encoded ground truth)\n\n";

    std::cout << "Mathematical formula: CE(y, p) = -sum(y_i * log(p_i)) for all i\n";
    std::cout << "Where y is the target and p is the prediction.\n\n";

    std::cout << "For classification with one-hot encoded targets:\n";
    std::cout << "CE simplifies to: -log(p_c) where c is the correct class.\n\n";

    // Create target labels (one-hot encoded)
    Tensor target({2, 4});
    target = 0.0;

    // First batch: the correct class is 2 (index starts at 0)
    target[{0, 2}] = 1.0;

    // Second batch: the correct class is 2 (index starts at 0)
    target[{1, 2}] = 1.0;

    std::cout << "Step 3: Creating one-hot encoded target labels\n";
    std::cout << "One-hot encoding: Only the correct class has value 1.0, others are 0.0\n";
    target.print();
    std::cout << "Here, class 2 (third position, index 2) is the correct class for both examples.\n\n";

    // Create cross entropy loss
    CrossEntropyLoss cross_entropy;

    // Calculate loss
    double loss = cross_entropy.forward(softmax_output, target);

    std::cout << "Step 4: Computing Cross Entropy Loss\n";

    // Manual calculation for educational purposes
    double manual_loss1 = -std::log(softmax_output[{0, 2}]);
    double manual_loss2 = -std::log(softmax_output[{1, 2}]);
    double avg_manual_loss = (manual_loss1 + manual_loss2) / 2.0;

    std::cout << "Manual calculation for Example 1:\n";
    std::cout << "  Probability of correct class (2): " << softmax_output[{0, 2}] << "\n";
    std::cout << "  -log(" << softmax_output[{0, 2}] << ") = " << manual_loss1 << "\n\n";

    std::cout << "Manual calculation for Example 2:\n";
    std::cout << "  Probability of correct class (2): " << softmax_output[{1, 2}] << "\n";
    std::cout << "  -log(" << softmax_output[{1, 2}] << ") = " << manual_loss2 << "\n\n";

    std::cout << "Average loss: (" << manual_loss1 << " + " << manual_loss2 << ") / 2 = " << avg_manual_loss << "\n";
    std::cout << "Cross entropy loss from our implementation: " << loss << "\n\n";

    std::cout << "Interpretation of Loss Values:\n";
    std::cout << "- Perfect prediction (prob = 1.0) gives loss = 0\n";
    std::cout << "- As prediction approaches 0, loss approaches infinity\n";
    std::cout << "- Lower loss = better predictions\n";
    std::cout << "- Example 2 has lower loss because our model is more confident about the correct class\n\n";

    std::cout << "PART 3: GRADIENTS FOR BACKPROPAGATION\n";
    std::cout << "-----------------------------------\n";
    std::cout << "When softmax is combined with cross-entropy loss, the gradient calculation simplifies to:\n";
    std::cout << "dLoss/dlogit_i = softmax_output_i - target_i\n\n";

    // Calculate cross entropy gradients
    Tensor loss_gradients = cross_entropy.backward(softmax_output, target);

    std::cout << "Step 5: Computing Gradients\n";
    std::cout << "The gradient with respect to softmax inputs is simply (prediction - target)/batch_size:\n";
    loss_gradients.print();

    // Verify the gradients for some positions
    std::cout << "\nManual verification for Example 1, Class 2 (correct class):\n";
    std::cout << "  Prediction (softmax output): " << softmax_output[{0, 2}] << "\n";
    std::cout << "  Target: " << target[{0, 2}] << "\n";
    std::cout << "  Gradient = " << softmax_output[{0, 2}] << " - " << target[{0, 2}]
              << "/2 = " << (softmax_output[{0, 2}] - target[{0, 2}]) / 2 << "\n";
    std::cout << "  Gradient from implementation: " << loss_gradients[{0, 2}] << "\n\n";

    std::cout << "Manual verification for Example 1, Class 3 (incorrect class):\n";
    std::cout << "  Prediction (softmax output): " << softmax_output[{0, 3}] << "\n";
    std::cout << "  Target: " << target[{0, 3}] << "\n";
    std::cout << "  Gradient = " << softmax_output[{0, 3}] << " - " << target[{0, 3}]
              << " =/2 " << (softmax_output[{0, 3}] - target[{0, 3}]) / 2 << "\n";
    std::cout << "  Gradient from implementation: " << loss_gradients[{0, 3}] << "\n\n";

    std::cout << "Key Properties of These Gradients:\n";
    std::cout << "- Negative gradients for correct class (prediction < 1) push logits up\n";
    std::cout << "- Positive gradients for incorrect classes push logits down\n";
    std::cout << "- Larger errors produce larger gradients, accelerating learning\n";
    std::cout << "- The sum of all gradients for one example is zero\n\n";

    // Backpropagate through softmax
    Tensor input_gradients = softmax_layer.backward(loss_gradients);

    std::cout << "Step 6: Backpropagating Through Softmax\n";
    std::cout << "These gradients will be passed to earlier layers in the network:\n";
    input_gradients.print();

    // Test with different targets
    std::cout << "\nPART 4: IMPACT OF DIFFERENT TARGETS\n";
    std::cout << "--------------------------------\n";
    std::cout << "Let's change our targets to see how the loss and gradients change:\n";

    // Change targets to make it more challenging
    target = 0.0;
    // First batch: the correct class is now 0
    target[{0, 0}] = 1.0;
    // Second batch: the correct class is now 3
    target[{1, 3}] = 1.0;

    std::cout << "New target labels (different correct classes):\n";
    target.print();
    std::cout << "Example 1: Class 0 is now correct (previously class 2)\n";
    std::cout << "Example 2: Class 3 is now correct (previously class 2)\n\n";

    // Calculate new loss
    double new_loss = cross_entropy.forward(softmax_output, target);
    std::cout << "New cross entropy loss: " << new_loss << "\n";
    std::cout << "Previous loss: " << loss << "\n";
    std::cout << "The loss increased because our model assigned lower probabilities to these classes.\n\n";

    // Calculate new gradients
    Tensor new_loss_gradients = cross_entropy.backward(softmax_output, target);

    std::cout << "New gradients with changed targets:\n";
    new_loss_gradients.print();

    std::cout << "\nObservation: The gradients for example 2, class 3 are smaller because\n";
    std::cout << "the model already assigned a small probability to that class, so the error is larger.\n\n";

    std::cout << "SUMMARY: WHY SOFTMAX + CROSS ENTROPY WORKS WELL\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "1. Softmax ensures our outputs form a valid probability distribution\n";
    std::cout << "2. Cross entropy effectively measures the difference between predicted and true distributions\n";
    std::cout << "3. The combined gradient has a simple form: (prediction - target)\n";
    std::cout << "4. This pairing creates larger gradients for larger errors, speeding up learning\n";
    std::cout << "5. The cross entropy loss approaches 0 as predictions approach perfect confidence\n\n";

    std::cout << "In neural network training, we would use these gradients to update weights:\n";
    std::cout << "weights = weights - learning_rate * gradients\n";
    std::cout << "This would gradually shift the network's predictions toward the correct classes.\n";
}

// Main function to run the test
int main()
{
    test_softmax_layer();
    return 0;
}