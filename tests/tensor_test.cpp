#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cassert>
#include <random>
#include "tensor_view.h"

// Test function for Tensor class
void test_tensor()
{
    std::cout << "\n===== EDUCATIONAL GUIDE: TENSOR OPERATIONS =====\n";
    std::cout << "PART 1: UNDERSTANDING TENSORS\n";
    std::cout << "--------------------------\n";
    std::cout << "Tensors are multi-dimensional arrays that form the foundation of\n";
    std::cout << "deep learning, representing inputs, weights, gradients, and activations.\n\n";

    std::cout << "Key concepts of tensors:\n";
    std::cout << "- Dimensionality: Number of axes (rank) in the tensor\n";
    std::cout << "- Shape: Size along each dimension (e.g., [batch, channels, height, width])\n";
    std::cout << "- Storage: How values are laid out in memory\n";
    std::cout << "- Views: Different ways to interpret the same underlying data\n";
    std::cout << "- Operations: Mathematical functions that transform tensors\n\n";

    // Step 1: Basic Tensor Creation
    std::cout << "Step 1: Creating Tensors of Different Dimensions\n";
    std::cout << "Tensors can have different ranks (number of dimensions):\n";
    std::cout << "- Scalars: 0D tensors (single values)\n";
    std::cout << "- Vectors: 1D tensors (arrays)\n";
    std::cout << "- Matrices: 2D tensors (tables)\n";
    std::cout << "- Cubes: 3D tensors (volumes)\n";
    std::cout << "- Higher dimensions: Used for batch processing or complex data\n\n";

    // Create tensors of different dimensions
    Tensor scalar({1});            // 0D tensor (scalar)
    Tensor vector({5});            // 1D tensor (vector)
    Tensor matrix({3, 4});         // 2D tensor (matrix)
    Tensor cube({2, 3, 4});        // 3D tensor (cube)
    Tensor tensor4d({2, 3, 4, 5}); // 4D tensor

    // Scalar example
    scalar.set_value({0}, 42.0);
    std::cout << "Scalar tensor (value = 42.0):\n";
    scalar.print();
    std::cout << "Rank: " << scalar.get_rank() << "\n";
    std::cout << "Number of elements: " << scalar.get_n_values() << "\n\n";

    // Vector example
    std::cout << "Vector tensor (1D array with 5 elements):\n";
    for (int i = 0; i < 5; ++i)
    {
        vector.set_value({i}, i * 2.0);
    }
    vector.print();
    std::cout << "Rank: " << vector.get_rank() << "\n";
    std::cout << "Shape: [" << vector.get_dimensions()[0] << "]\n";
    std::cout << "Number of elements: " << vector.get_n_values() << "\n\n";

    // Matrix example
    std::cout << "Matrix tensor (2D array with 3 rows, 4 columns):\n";
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            matrix.set_value({i, j}, i * 10 + j);
        }
    }
    matrix.print();
    std::cout << "Rank: " << matrix.get_rank() << "\n";
    std::cout << "Shape: [" << matrix.get_dimensions()[0] << ", "
              << matrix.get_dimensions()[1] << "]\n";
    std::cout << "Number of elements: " << matrix.get_n_values() << "\n";
    std::cout << "Strides: [" << matrix.get_strides()[0] << ", "
              << matrix.get_strides()[1] << "]\n\n";

    // Step 2: Tensor Storage and Memory Layout
    std::cout << "Step 2: Understanding Tensor Storage and Memory Layout\n";
    std::cout << "Tensors store data in a contiguous block of memory, but can be viewed\n";
    std::cout << "in different ways based on dimensions and strides:\n";
    std::cout << "- Dimensions: The shape of the tensor\n";
    std::cout << "- Strides: How many steps to take in memory for each dimension\n";
    std::cout << "- Contiguous: Whether the data is stored without gaps\n";
    std::cout << "- Offset: Starting position in the storage\n\n";

    std::cout << "For the 3x4 matrix above:\n";
    std::cout << "- Total storage size: " << matrix.get_n_values() << " elements\n";
    std::cout << "- Memory layout: Row-major (rows are stored contiguously)\n";
    std::cout << "- To access element [1,2], we compute:\n";
    std::cout << "  index = 1*" << matrix.get_strides()[0] << " + 2*" << matrix.get_strides()[1] << " = "
              << (1 * matrix.get_strides()[0] + 2 * matrix.get_strides()[1]) << "\n\n";

    // Step 3: Tensor Views and Slicing
    std::cout << "Step 3: Tensor Views and Slicing\n";
    std::cout << "Tensor views allow accessing parts of a tensor without copying data.\n";
    std::cout << "This is crucial for efficiency in deep learning.\n\n";

    std::cout << "Creating a 3D tensor (2x3x4) with sequential values:\n";
    int counter = 0;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                cube.set_value({i, j, k}, counter++);
            }
        }
    }
    cube.print();

    std::cout << "\nCreating a view of the first slice (index 0):\n";
    Tensor slice = cube[0]; // Get the first slice
    slice.print();
    std::cout << "Slice shape: [" << slice.get_dimensions()[0] << ", "
              << slice.get_dimensions()[1] << "]\n";
    std::cout << "Slice shares the same underlying storage as the original tensor.\n\n";

    std::cout << "Modifying an element in the slice (setting [1,2] to 99):\n";
    slice.set_value({1, 2}, 99);
    slice.print();
    std::cout << "The modification is reflected in the original tensor:\n";
    cube.print();
    std::cout << "This demonstrates that slice is a view, not a copy.\n\n";

    // Step 4: Contiguity and its Importance
    std::cout << "Step 4: Tensor Contiguity and Performance\n";
    std::cout << "Tensor contiguity affects performance dramatically.\n";
    std::cout << "- Contiguous tensors: Elements are adjacent in memory\n";
    std::cout << "- Non-contiguous tensors: Elements may have gaps\n\n";

    std::cout << "Creating a transposed view of our matrix:\n";
    Tensor transposed = matrix.transpose();
    transposed.print();
    std::cout << "Original matrix is contiguous: " << (matrix.is_contiguous() ? "Yes" : "No") << "\n";
    std::cout << "Transposed matrix is contiguous: " << (transposed.is_contiguous() ? "Yes" : "No") << "\n\n";

    std::cout << "Creating a contiguous copy of the transposed view:\n";
    Tensor contiguous_transposed = transposed.make_contiguous();
    contiguous_transposed.print();
    std::cout << "Contiguous copy is contiguous: " << (contiguous_transposed.is_contiguous() ? "Yes" : "No") << "\n";
    std::cout << "Note: In deep learning frameworks, operations often require contiguous tensors for maximum performance.\n\n";

    // PART 2: TENSOR OPERATIONS
    std::cout << "PART 2: TENSOR OPERATIONS\n";
    std::cout << "--------------------\n";
    std::cout << "Tensors support various mathematical operations, which are the foundation\n";
    std::cout << "of neural network computations.\n\n";

    // Step 5: Element-wise Operations
    std::cout << "Step 5: Element-wise Operations\n";
    std::cout << "Element-wise operations apply the same function to each element:\n\n";

    // Create two 2x3 matrices for demonstrating operations
    Tensor a({2, 3});
    Tensor b({2, 3});

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            a.set_value({i, j}, i + j + 1);         // Values 1-4
            b.set_value({i, j}, (i + 1) * (j + 1)); // Values 1-6
        }
    }

    std::cout << "Matrix A:\n";
    a.print();
    std::cout << "Matrix B:\n";
    b.print();

    // Addition
    std::cout << "\nElement-wise Addition (A + B):\n";
    Tensor c = a + b;
    c.print();
    std::cout << "Each element is the sum of corresponding elements in A and B.\n\n";

    // Multiplication
    std::cout << "Element-wise Multiplication (A * B):\n";
    Tensor d = a * b;
    d.print();
    std::cout << "Each element is the product of corresponding elements in A and B.\n";
    std::cout << "This is also known as the Hadamard product.\n\n";

    // Scalar operations
    std::cout << "Scalar Operations:\n";
    std::cout << "A + 2.0:\n";
    Tensor a_plus_2 = a + 2.0;
    a_plus_2.print();

    std::cout << "A * 3.0:\n";
    Tensor a_times_3 = a * 3.0;
    a_times_3.print();

    std::cout << "A / 2.0:\n";
    Tensor a_div_2 = a / 2.0;
    a_div_2.print();
    std::cout << "Scalar operations are applied to every element.\n\n";

    // Step 6: Matrix Multiplication
    std::cout << "Step 6: Matrix Multiplication\n";
    std::cout << "Matrix multiplication is a fundamental operation in neural networks,\n";
    std::cout << "used in fully connected layers, attention mechanisms, and more.\n\n";

    Tensor p({2, 3}); // 2x3 matrix
    Tensor q({3, 2}); // 3x2 matrix

    // Fill with recognizable values
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            p.set_value({i, j}, i + j + 1);
        }
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            q.set_value({i, j}, i * 2 + j + 1);
        }
    }

    std::cout << "Matrix P (2x3):\n";
    p.print();
    std::cout << "Matrix Q (3x2):\n";
    q.print();

    std::cout << "\nMatrix Multiplication (P·Q):\n";
    Tensor pq = p.matrix_multiplication(q);
    pq.print();

    std::cout << "Mathematical calculation for P·Q[0,0]:\n";
    std::cout << "P[0,0] * Q[0,0] + P[0,1] * Q[1,0] + P[0,2] * Q[2,0]\n";
    std::cout << "= " << p.get_value({0, 0}) << " * " << q.get_value({0, 0})
              << " + " << p.get_value({0, 1}) << " * " << q.get_value({1, 0})
              << " + " << p.get_value({0, 2}) << " * " << q.get_value({2, 0}) << "\n";
    double expected = p.get_value({0, 0}) * q.get_value({0, 0}) +
                      p.get_value({0, 1}) * q.get_value({1, 0}) +
                      p.get_value({0, 2}) * q.get_value({2, 0});
    std::cout << "= " << expected << "\n";
    std::cout << "Actual value in P·Q[0,0]: " << pq.get_value({0, 0}) << "\n\n";

    // Step 7: Advanced Operations (Convolution)
    std::cout << "Step 7: Convolution Operations\n";
    std::cout << "Convolution is the core operation in Convolutional Neural Networks (CNNs),\n";
    std::cout << "used for feature extraction in images and other spatial data.\n\n";

    // Create a simple input and kernel
    Tensor input({5, 5});  // 5x5 input
    Tensor kernel({3, 3}); // 3x3 kernel

    // Fill input with a recognizable pattern (X shape)
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            input.set_value({i, j}, (i == j || i == 4 - j) ? 1.0 : 0.0);
        }
    }

    // Create a basic edge detection kernel
    kernel.set_value({0, 0}, -1.0);
    kernel.set_value({0, 1}, -1.0);
    kernel.set_value({0, 2}, -1.0);
    kernel.set_value({1, 0}, -1.0);
    kernel.set_value({1, 1}, 8.0);
    kernel.set_value({1, 2}, -1.0);
    kernel.set_value({2, 0}, -1.0);
    kernel.set_value({2, 1}, -1.0);
    kernel.set_value({2, 2}, -1.0);

    std::cout << "Input (5x5 with X pattern):\n";
    input.print();
    std::cout << "Kernel (3x3 edge detection):\n";
    kernel.print();

    std::cout << "\nConvolution Result:\n";
    Tensor conv_result = input.convolve(kernel);
    conv_result.print();

    std::cout << "The convolution detected the edges of the X pattern.\n";
    std::cout << "Note: In CNNs, we typically use many different kernels to detect various features.\n\n";

    // PART 3: MEMORY MANAGEMENT
    std::cout << "PART 3: TENSOR MEMORY MANAGEMENT\n";
    std::cout << "----------------------------\n";
    std::cout << "Understanding how tensors share and copy memory is crucial for\n";
    std::cout << "efficient and correct neural network implementation.\n\n";

    // Step 8: Shallow vs Deep Copies
    std::cout << "Step 8: Shallow vs Deep Copies\n";
    std::cout << "Tensors can share memory (views) or have independent copies:\n\n";

    // Create a base tensor
    Tensor base_tensor({2, 3});
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            base_tensor.set_value({i, j}, i * 10 + j);
        }
    }

    std::cout << "Base tensor:\n";
    base_tensor.print();

    // Create a shallow copy (view)
    std::cout << "\nCreating a view (shallow copy):\n";
    Tensor view_tensor(base_tensor);
    view_tensor.print();

    // Create a deep copy
    std::cout << "\nCreating a deep copy:\n";
    Tensor copy_tensor = base_tensor.deep_copy();
    copy_tensor.print();

    // Modify the base tensor
    std::cout << "\nModifying base_tensor[0,0] to 99:\n";
    base_tensor.set_value({0, 0}, 99);

    // Show results
    std::cout << "Base tensor after modification:\n";
    base_tensor.print();
    std::cout << "View tensor (shallow copy):\n";
    view_tensor.print();
    std::cout << "Deep copy tensor:\n";
    copy_tensor.print();

    std::cout << "\nObservations:\n";
    std::cout << "- The view tensor changed when the base tensor was modified (shares memory)\n";
    std::cout << "- The deep copy remained unchanged (independent memory)\n\n";

    // Step 9: Memory Efficiency with Views
    std::cout << "Step 9: Memory Efficiency with Views\n";
    std::cout << "Tensor views are critical for performance in deep learning:\n";
    std::cout << "- Avoid unnecessary copies of large data\n";
    std::cout << "- Enable efficient slicing and reshaping\n";
    std::cout << "- Allow multiple 'perspectives' on the same data\n\n";

    // Create a larger tensor to demonstrate efficiency
    Tensor large_tensor({100, 100});
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 100; ++j)
        {
            large_tensor.set_value({i, j}, i * 0.01 + j * 0.0001);
        }
    }

    std::cout << "Created a 100x100 tensor with 10,000 elements\n";
    std::cout << "Creating a view of the first 5x5 corner:\n";

    // Extract a view of the first 5x5 block (without copying data)
    Tensor corner_view({5, 5});
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            corner_view.set_value({i, j}, large_tensor.get_value({i, j}));
        }
    }

    corner_view.print();
    std::cout << "This view shares memory with the original tensor's corner.\n";
    std::cout << "In deep learning, this pattern is used extensively, for example in:\n";
    std::cout << "- Batch processing (views for each sample)\n";
    std::cout << "- Convolutional operations (sliding windows)\n";
    std::cout << "- Attention mechanisms (key, query, value projections)\n\n";

    // Summary
    std::cout << "SUMMARY: TENSOR FUNDAMENTALS\n";
    std::cout << "-------------------------\n";
    std::cout << "Tensors are the building blocks of neural networks:\n";
    std::cout << "1. Multi-dimensional arrays with flexible views and operations\n";
    std::cout << "2. Efficient memory management through shared storage\n";
    std::cout << "3. Support for element-wise operations and matrix multiplication\n";
    std::cout << "4. Advanced operations like convolution for spatial data\n";
    std::cout << "5. The foundation for representing and transforming data in deep learning\n\n";

    std::cout << "Every computation in neural networks - from forward passes to\n";
    std::cout << "backpropagation and weight updates - is expressed as tensor operations.\n";
    std::cout << "Understanding tensors is crucial for designing and optimizing\n";
    std::cout << "neural networks for various applications.\n";
}

// Main function to run the test
int main()
{
    test_tensor();
    return 0;
}