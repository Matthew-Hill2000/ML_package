import torch
import torch.nn as nn
import numpy as np

# Configure printing options
torch.set_printoptions(precision=6, sci_mode=False)
np.set_printoptions(precision=6, suppress=True)

print("\n===== PYTORCH VERIFICATION: FULLY CONNECTED LAYER =====\n")

# Step 1: Create Input Data (matching the C++ implementation)
input_size = 4
output_size = 3
batch_size = 2

# First create a numpy array with our desired values
input_values = np.zeros((batch_size, input_size))
for b in range(batch_size):
    for i in range(input_size):
        input_values[b, i] = b * 10 + i + 1

# Convert to tensor with gradients
input_tensor = torch.tensor(input_values, dtype=torch.float32, requires_grad=True)

print("Input tensor (batch=2, features=4):")
print(input_tensor)

# Step 2: Create Fully Connected Layer
class CustomLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomLinear, self).__init__()
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.zeros(input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        
    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

fc_layer = CustomLinear(input_size, output_size)

# Step 3: Set Custom Weights and Biases (matching the C++ implementation)
# Create weight and bias tensors with desired values
weight_values = torch.zeros((input_size, output_size))
for i in range(input_size):
    for j in range(output_size):
        weight_values[i, j] = 0.1 * (i + j + 1)

bias_values = torch.ones(output_size)  # All biases set to 1.0

# Create parameters with these values
with torch.no_grad():
    fc_layer.weight.copy_(weight_values)
    fc_layer.bias.copy_(bias_values)

print("\nCustom weights:")
print(fc_layer.weight.detach())

print("\nCustom biases:")
print(fc_layer.bias.detach())

# Step 4: Perform Forward Pass
output = fc_layer(input_tensor)

print("\nForward pass output:")
print(output)

# Step 5: Create Output Gradients (from next layer)
output_gradients = torch.zeros_like(output)
for b in range(batch_size):
    for i in range(output_size):
        output_gradients[b, i] = 0.1 * (b * output_size + i + 1)

print("\nOutput gradients:")
print(output_gradients)

# Step 6: Perform Backward Pass
output.backward(output_gradients)

print("\nInput gradients:")
print(input_tensor.grad)

print("\nWeight gradients:")
print(fc_layer.weight.grad)

print("\nBias gradients:")
print(fc_layer.bias.grad)

# Step 7: Update Weights and Biases
learning_rate = 0.01

# Store original weights and biases for comparison
original_weights = fc_layer.weight.clone().detach()
original_biases = fc_layer.bias.clone().detach()

print("\nOriginal weights before update:")
print(original_weights)

print("\nOriginal biases before update:")
print(original_biases)

# Update weights
with torch.no_grad():
    fc_layer.weight -= learning_rate * fc_layer.weight.grad
    fc_layer.bias -= learning_rate * fc_layer.bias.grad

print("\nUpdated weights after gradient descent step:")
print(fc_layer.weight)

print("\nUpdated biases after gradient descent step:")
print(fc_layer.bias)

# Forward pass with updated weights
new_output = fc_layer(input_tensor)
print("\nOutput after weight update:")
print(new_output)

print("\nOutput change after weight update:")
print(new_output - output.detach())