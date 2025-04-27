#include "convolutional_layer.h"

// Prints kernel values to console for debugging
void ConvolutionalLayer::print_kernels()
{
  std::cout << "Kernels:" << std::endl;
  kernels.print();
}

// Prints bias values to console for debugging
void ConvolutionalLayer::print_biases()
{
  std::cout << "Biases:" << std::endl;
  biases.print();
}

// Prints kernel gradient values to console for debugging
void ConvolutionalLayer::print_weight_gadient()
{
  std::cout << "Kernel Gradients:" << std::endl;
  kernel_gradients.print();
}

// Prints bias gradient values to console for debugging
void ConvolutionalLayer::print_bias_gradient()
{
  std::cout << "Bias Gradients:" << std::endl;
  bias_gradients.print();
}

// Sets a single kernel for a specific filter and channel
void ConvolutionalLayer::set_kernel(int filter_idx, int channel_idx, const Tensor &new_kernel)
{
  // Check if the indices are valid
  if (filter_idx < 0 || filter_idx >= num_filters)
  {
    throw std::out_of_range("Filter index out of range");
  }

  if (channel_idx < 0 || channel_idx >= input_shape[0])
  {
    throw std::out_of_range("Channel index out of range");
  }

  // Check if the new kernel has the correct dimensions
  auto kernel_dims = new_kernel.get_dimensions();
  if (kernel_dims.size() != 2 || kernel_dims[0] != kernel_size || kernel_dims[1] != kernel_size)
  {
    throw std::invalid_argument("Kernel dimensions must be [kernel_size, kernel_size]");
  }

  // Copy values from the new kernel to the specified filter and channel
  for (int i = 0; i < kernel_size; i++)
  {
    for (int j = 0; j < kernel_size; j++)
    {
      kernels[{filter_idx, channel_idx, i, j}] = new_kernel[{i, j}];
    }
  }
}

// Replaces all kernels with provided tensor
void ConvolutionalLayer::set_all_kernels(const Tensor &new_kernels)
{
  // Check if the new kernels tensor has the correct dimensions
  auto kernel_dims = new_kernels.get_dimensions();

  if (kernel_dims.size() != 4 ||
      kernel_dims[0] != num_filters ||
      kernel_dims[1] != input_shape[0] ||
      kernel_dims[2] != kernel_size ||
      kernel_dims[3] != kernel_size)
  {
    throw std::invalid_argument(
        "Kernels dimensions must be [num_filters, input_channels, kernel_size, kernel_size]");
  }

  // Replace the entire kernels tensor
  this->kernels = new_kernels.deep_copy();
}

// Returns a copy of all kernels
Tensor ConvolutionalLayer::get_kernels() const
{
  return kernels.deep_copy(); // Return a deep copy of the kernels
}

// Sets the enable_parallelization flag
void ConvolutionalLayer::set_enable_parallelization(bool enable_parallelization)
{
  this->enable_parallelization = enable_parallelization;
}