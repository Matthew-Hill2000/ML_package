#include "../../tensor/tensor_view.h"
#include "convolutional_layer.h"
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <iomanip>
#include <chrono>

int main() {
    std::cout << "Testing ConvolutionalLayer Forward and Backward Pass\n";
    std::cout << "==================================================\n";

    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());  // Mersenne Twister random generator
    std::uniform_real_distribution<double> dist(-0.5, 0.5);  // Uniform distribution between -0.5 and 0.5

    // Create a batch of 8 inputs with 3 channels and 28x28 dimensions (like MNIST with color)
    int batch_size = 8;
    int input_channels = 3;
    int input_height = 28;
    int input_width = 28;
    Tensor input({batch_size, input_channels, input_height, input_width});
    
    // Fill with random values
    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < input_channels; channel++) {
            for (int h = 0; h < input_height; h++) {
                for (int w = 0; w < input_width; w++) {
                    double value = dist(gen);
                    input[{batch, channel, h, w}] = value;
                }
            }
        }
    }
    
    // Create a convolutional layer with 3 input channels, 16 output channels, and 5x5 kernel
    int output_channels = 16;
    int kernel_size = 5;
    std::vector<int> input_shape = {input_channels, input_height, input_width};

    ConvolutionalLayer conv_layer(input_shape, output_channels, kernel_size);
    conv_layer.set_enable_parallelization(true);  // Enable parallelization for performance

    // Calculate output dimensions
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    
    // Perform forward pass
    Tensor output = conv_layer.forward(input);
    
    // Create random gradient for backward pass
    Tensor output_gradient({batch_size, output_channels, output_height, output_width});
    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < output_channels; channel++) {
            for (int h = 0; h < output_height; h++) {
                for (int w = 0; w < output_width; w++) {
                    double grad_value = dist(gen);
                    output_gradient[{batch, channel, h, w}] = grad_value;
                }
            }
        }
    }
    
    // Warm-up passes (not timed) to eliminate initialization effects
    Tensor input_gradient = conv_layer.backward(output_gradient);
    
    // Time the forward pass
    std::cout << "\nTimings for forward pass:\n";
    
    // Number of iterations for timing
    const int num_iterations = 100;  // Reduced from 1000 because conv ops are more expensive
    
    auto forward_start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        output = conv_layer.forward(input);
    }
    
    auto forward_end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate timing statistics for forward pass
    std::chrono::duration<double, std::milli> forward_elapsed = forward_end_time - forward_start_time;
    double forward_total_ms = forward_elapsed.count();
    double forward_avg_ms = forward_total_ms / (num_iterations*batch_size);
    
    std::cout << "Total time for " << num_iterations << " iterations: " 
              << std::fixed << std::setprecision(2) << forward_total_ms << " ms\n";
    std::cout << "Average time per forward pass: " 
              << std::fixed << std::setprecision(4) << forward_avg_ms << " ms\n";
    std::cout << "Throughput: " 
              << std::fixed << std::setprecision(2) << (1000.0 / forward_avg_ms) << " forward passes per second\n";
    
    // Time the backward pass
    std::cout << "\nTimings for backward pass:\n";
    
    auto backward_start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        input_gradient = conv_layer.backward(output_gradient);
    }
    
    auto backward_end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate timing statistics for backward pass
    std::chrono::duration<double, std::milli> backward_elapsed = backward_end_time - backward_start_time;
    double backward_total_ms = backward_elapsed.count();
    double backward_avg_ms = backward_total_ms / (num_iterations*batch_size);
    
    std::cout << "Total time for " << num_iterations << " iterations: " 
              << std::fixed << std::setprecision(2) << backward_total_ms << " ms\n";
    std::cout << "Average time per backward pass: " 
              << std::fixed << std::setprecision(4) << backward_avg_ms << " ms\n";
    std::cout << "Throughput: " 
              << std::fixed << std::setprecision(2) << (1000.0 / backward_avg_ms) << " backward passes per second\n";
    
    // Output basic information about layer and gradients
    std::cout << "\nLayer Configuration:\n";
    std::cout << "Input shape: " << batch_size << "x" << input_channels << "x" 
              << input_height << "x" << input_width << "\n";
    std::cout << "Kernel shape: " << output_channels << "x" << input_channels << "x" 
              << kernel_size << "x" << kernel_size << "\n";
    std::cout << "Output shape: " << batch_size << "x" << output_channels << "x" 
              << output_height << "x" << output_width << "\n";
    
    // Check dimensions of input gradient
    std::vector<int> input_grad_dims = input_gradient.get_dimensions();
    std::cout << "\nInput gradient tensor dimensions: ";
    for (size_t i = 0; i < input_grad_dims.size(); i++) {
        std::cout << input_grad_dims[i];
        if (i < input_grad_dims.size() - 1) {
            std::cout << " x ";
        }
    }
    std::cout << "\n";
    
    return 0;
}