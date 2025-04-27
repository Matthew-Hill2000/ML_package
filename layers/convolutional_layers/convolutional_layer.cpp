#include "convolutional_layer.h"
#include <cmath>
#include <omp.h>

// Constructor - initializes the layer with specified dimensions and Xavier weight initialization
ConvolutionalLayer::ConvolutionalLayer(const std::vector<int> &input_shape, int output_channels, int kernel_size)
    : num_filters{output_channels}, input_shape{input_shape}, kernel_size{kernel_size}
{
  // Calculate output shape based on input dimensions and kernel size
  this->output_shape = {num_filters,
      input_shape[1] - kernel_size + 1, // output height
      input_shape[2] - kernel_size + 1  // output width
      };

  // Initialize output tensor
  this->output = Tensor(output_shape);

  // Initialize kernels with Xavier/Glorot initialization for better convergence
  this->kernels = Tensor({num_filters, input_shape[0], kernel_size, kernel_size});
  double xavier_limit =
      sqrt(6.0 / (input_shape[0] * kernel_size * kernel_size +
                  output_channels * kernel_size * kernel_size));

  // Initialize kernel weights with random values within Xavier bounds
  #pragma omp parallel for collapse(4) if (enable_parallelization)
  for (int f = 0; f < num_filters; f++)
  {
    for (int c = 0; c < input_shape[0]; c++)
    {
      for (int i = 0; i < kernel_size; i++)
      {
        for (int j = 0; j < kernel_size; j++)
        {
          double rand_val =
              ((double)rand() / RAND_MAX) * 2 * xavier_limit - xavier_limit;
          kernels[{f, c, i, j}] = rand_val;
        }
      }
    }
  }

  // Initialize biases to zero
  this->biases = Tensor(output_shape);
  biases = 0.0;

  // Initialize gradient tensors with appropriate dimensions
  this->kernel_gradients =
      Tensor({num_filters, input_shape[0], kernel_size, kernel_size});
  this->bias_gradients = Tensor(output_shape);
  this->input_gradients = Tensor(input_shape);
}

// Forward pass - applies filters to input and adds biases
Tensor ConvolutionalLayer::forward(const Tensor &input)
{
  // Get batch size from first dimension of input
  int batch_size = input.get_dimensions()[0];

  // Store input for backward pass
  this->input = input;

  // Create output tensor with shape [batch_size, num_filters, output_height, output_width]
  std::vector<int> output_shape = {
      batch_size,
      num_filters,
      input_shape[1] - kernel_size + 1, // output height
      input_shape[2] - kernel_size + 1  // output width
  };

  // Reset output tensor to ensure clean state
  this->output = Tensor(output_shape);

  // Process each example in the batch
  #pragma omp parallel for collapse(2) if (enable_parallelization)
  for (int b = 0; b < batch_size; b++)
  {
    // Get the input for this batch item
    Tensor batch_input = input[b];

    // For each filter
    for (int f = 0; f < num_filters; f++)
    {
      // For each input channel
      for (int c = 0; c < input_shape[0]; c++)
      {
        // Get the current input channel and kernel
        Tensor input_channel = batch_input[c];
        Tensor kernel = kernels[f][c];

        // Perform cross correlation (convolution without flipping the kernel)
        Tensor conv_result = input_channel.cross_correlate(kernel, enable_parallelization);

        // Add to existing values (for channel-wise sum) and add biases
        #pragma omp parallel for collapse(2) if (enable_parallelization)
        for (int i = 0; i < output_shape[1]; i++)
        {
          for (int j = 0; j < output_shape[2]; j++)
          {
            this->output[{b, f, i, j}] += conv_result[{i, j}] + biases[{f, i, j}];
          }
        }
      }

    }
  }
  return output;
}

// Backward pass - computes gradients for kernels, biases, and inputs
Tensor ConvolutionalLayer::backward(Tensor &output_gradients)
{
  // Get batch size from output gradients
  int batch_size = output_gradients.get_dimensions()[0];

  // Store output gradients for the backward pass
  this->output_gradients = output_gradients;

  // Reset gradient tensors to ensure clean state
  this->kernel_gradients = Tensor({num_filters, input_shape[0], kernel_size, kernel_size});
  this->bias_gradients = Tensor(output_shape);

  // Create input gradients tensor with batch dimension
  this->input_gradients = Tensor(this->input.get_dimensions());

  // Process each example in the batch
  #pragma omp parallel for collapse(2) if (enable_parallelization)
  for (int b = 0; b < batch_size; b++)
  {
    // Get the output gradients for this batch item
    Tensor batch_output_gradients = output_gradients[b];
    // Get the input for this batch item
    Tensor batch_input = this->input[b];

    Tensor channel_gradient({kernel_size, kernel_size});
    Tensor input_grad_contribution;

    // For each filterTensor
    for (int f = 0; f < num_filters; f++)
    {
      // For each input channel
      for (int c = 0; c < input_shape[0]; c++)
      {
        // Compute kernel gradients using cross correlation between input and output gradients
        channel_gradient = batch_input[c].cross_correlate(batch_output_gradients[f], enable_parallelization);

        // Accumulate gradients to kernel_gradients tensor
        #pragma omp parallel for collapse(2) if (enable_parallelization)
        for (int i = 0; i < kernel_size; i++)
        {
          for (int j = 0; j < kernel_size; j++)
          {
            kernel_gradients[{f, c, i, j}] += channel_gradient[{i, j}];
          }
        }

        // Compute input gradients using full convolution with output gradients and kernels
        input_grad_contribution = batch_output_gradients[f].fully_convolve(kernels[f][c], enable_parallelization);

        // Add to input gradients for this batch item
        #pragma omp parallel for collapse(2) if (enable_parallelization)
        for (int i = 0; i < input_shape[1]; i++)
        {
          for (int j = 0; j < input_shape[2]; j++)
          {
            if (i < input_grad_contribution.get_dimensions()[0] &&
                j < input_grad_contribution.get_dimensions()[1])
            {
              this->input_gradients[{b, c, i, j}] += input_grad_contribution[{i, j}];
            }
          }
        }
      }

      // Accumulate bias gradients (same as output gradients) for this batch item
      #pragma omp parallel for collapse(2) if (enable_parallelization)
      for (int i = 0; i < output_shape[1]; i++)
      {
        for (int j = 0; j < output_shape[2]; j++)
        {
          bias_gradients[{f, i, j}] += batch_output_gradients[{f, i, j}];
        }
      }
    }
  }

  return input_gradients;
}

// Updates kernels and biases using calculated gradients and learning rate
void ConvolutionalLayer::update_weights(double learning_rate)
{
  // Update kernels using gradients
  #pragma omp parallel for collapse(4) if (enable_parallelization)
  for (int f = 0; f < num_filters; f++)
  {
    for (int c = 0; c < input_shape[0]; c++)
    {
      for (int i = 0; i < kernel_size; i++)
      {
        for (int j = 0; j < kernel_size; j++)
        {
          kernels[{f, c, i, j}] -= learning_rate * kernel_gradients[{f, c, i, j}];
        }
      }
    }
  }

  // Update biases using gradients
  #pragma omp parallel for collapse(3) if (enable_parallelization)
  for (int f = 0; f < num_filters; f++)
  {
    for (int i = 0; i < output_shape[1]; i++)
    {
      for (int j = 0; j < output_shape[2]; j++)
      {
        biases[{f, i, j}] -= learning_rate * bias_gradients[{f, i, j}];
      }
    }
  }
}

// Resets all gradients to zero for next iteration
void ConvolutionalLayer::reset_gradients()
{
  // Reset all gradient tensors to zeros
  this->kernel_gradients = Tensor({num_filters, input_shape[0], kernel_size, kernel_size});
  this->bias_gradients = Tensor(output_shape);

  // Only initialize input_gradients if we have valid input shape
  if (!input_shape.empty())
  {
    input_gradients = Tensor(input_shape);
  }
}