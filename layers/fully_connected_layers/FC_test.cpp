#include "../../tensor/tensor_view.h"
#include "fully_connected_layer.h"
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <iomanip>
#include <chrono>

int main() {
    std::cout << "Testing FullyConnectedLayer Forward and Backward Pass\n";
    std::cout << "==================================================\n";

    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());  // Mersenne Twister random generator
    std::uniform_real_distribution<double> dist(-0.5, 0.5);  // Uniform distribution between -0.5 and 0.5

    // Create a batch of 8 inputs with 64 features each
    int batch_size = 8;
    Tensor input({batch_size, 64});
    
    // Fill with random values
    for (int batch = 0; batch < 8; batch++) {
        for (int feature = 0; feature < 64; feature++) {
            double value = dist(gen);
            input[{batch, feature}] = value;
        }
    }
    
    // Create a fully connected layer with 64 inputs and 32 outputs
    int input_size = 64;
    int output_size = 32;

    FullyConnectedLayer fc_layer(input_size, output_size);
    fc_layer.set_enable_parallelization(false);  // Enable parallelization for the layer
    
    // Generate random weights
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            double weight_value = dist(gen);
            fc_layer.set_weight(i, j, weight_value);
        }
    }
    
    // Generate random biases
    for (int j = 0; j < output_size; j++) {
        double bias_value = dist(gen);
        fc_layer.set_bias(j, bias_value);
    }
    
    // Perform forward pass
    Tensor output = fc_layer.forward(input);
    
    // Create random gradient for backward pass
    Tensor output_gradient({batch_size, output_size});
    for (int batch = 0; batch < batch_size; batch++) {
        for (int feature = 0; feature < output_size; feature++) {
            double grad_value = dist(gen);
            output_gradient[{batch, feature}] = grad_value;
        }
    }
    
    // Warm-up passes (not timed) to eliminate initialization effects
    Tensor input_gradient = fc_layer.backward(output_gradient);
    
    // Time the forward pass
    std::cout << "\nTimings for forward pass:\n";
    
    // Number of iterations for timing
    const int num_iterations = 1000;
    
    auto forward_start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        output = fc_layer.forward(input);
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
        input_gradient = fc_layer.backward(output_gradient);
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
    
    // Output basic information about the gradients
    std::cout << "\nInput gradient tensor dimensions: " << input_gradient.get_dimensions()[0] << " x " 
              << input_gradient.get_dimensions()[1] << "\n";
    
    return 0;
}