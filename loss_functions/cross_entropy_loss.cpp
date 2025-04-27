#include "cross_entropy_loss.h"
#include <omp.h>

// Computes the forward pass of the cross-entropy loss
// Returns the average loss across the batch
double CrossEntropyLoss::forward(const Tensor &output, const Tensor &target)
{
    // Get batch size from first dimension
    std::vector<int> output_dims = output.get_dimensions();
    int batch_size = output_dims[0];

    double total_loss = 0.0;

    // Process each sample in the batch
    #pragma omp parallel for if(enable_parallelization) 
    for (int b = 0; b < batch_size; b++)
    {
        Tensor sample_output = output[b];
        Tensor sample_target = target[b];
        double sample_loss = 0.0;

        // Calculate cross-entropy loss for this sample with numerical stability
        for (int i = 0; i < sample_output.get_n_values(); i++)
        {
            // Add small epsilon to prevent log(0)
            double epsilon = 1e-10;
            double safe_output = std::max(sample_output.get_value_direct(i), epsilon);
            sample_loss += sample_target.get_value_direct(i) * std::log(safe_output);
        }

        // Accumulate negative loss for this sample
        total_loss -= sample_loss;
    }

    // Return average loss across the batch
    return total_loss / batch_size;
}

// Computes the gradients for backpropagation
// Returns a tensor of same shape as output with loss gradients
Tensor CrossEntropyLoss::backward(const Tensor &output, const Tensor &target)
{
    // Get batch size and dimensions
    std::vector<int> dimensions = output.get_dimensions();
    int batch_size = dimensions[0];

    // Create gradient tensor with same dimensions as output (including batch)
    Tensor output_gradients(dimensions);

    // Process each sample in the batch
    #pragma omp parallel for if(enable_parallelization)
    for (int b = 0; b < batch_size; b++)
    {
        Tensor sample_output = output[b];
        Tensor sample_target = target[b];

        // For softmax followed by cross-entropy, the gradient is (output - target)
        for (int i = 0; i < sample_output.get_n_values(); i++)
        {
            double gradient = (sample_output.get_value_direct(i) - sample_target.get_value_direct(i)) / batch_size;

            // Scale gradient by 1/batch_size since we're returning average loss
            output_gradients[b].set_value_direct(i, gradient);
        }
    }

    return output_gradients;
}