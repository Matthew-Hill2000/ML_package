#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>

// The Tensor class implements a multi-dimensional tensor data structure.
// The implementation uses a flat 1D vector to store values, with an indexing
// system based on multiplicative factors to access elements efficiently.
//
// Storage Implementation:
// - Values are stored in a 1D std::vector<double>
// - Access to multi-dimensional indices is achieved through multiplicative
// factors
// - For a tensor with dimensions {2, 2, 3}, the factors are {6, 3, 1}
//   This means:
//   - First dimension spans 6 values (2 * 3)
//   - Second dimension spans 3 values
//   - Third dimension spans 1 value
//
// Example:
// For a tensor with dimensions {2, 2, 3}:
// - To access element [1,1,2]
// - Index = 1*6 + 1*3 + 2*1 = 11 in the flat array

class Tensor {
  private:
    std::vector<int> dimensions; // Stores the size of each dimension
    std::vector<double> values;  // Flat array storing all tensor values
    std::vector<int> factors;    // Multiplicative factors for index calculation
    int n_values;                // Total number of values in the tensor
    int rank;                    // Number of dimensions in the tensor

  public:
    // Default constructor
    // Creates a 1D tensor of size 1 initialized to 0
    // Resulting tensor: [0]
    Tensor();

    // Takes a vector specifying size of each dimension
    // Creates a tensor with the specified dimensions, initialized to 0
    // Example: Tensor({2,2,2}) creates [[[0,0],[0,0]],[[0,0],[0,0]]]
    Tensor(std::vector<int> dims);

    // Creates a tensor with all attributes directly specified
    // dims: Vector of dimension sizes
    // values: Vector of tensor values in row-major order
    // factors: Vector of multiplicative factors for indexing
    // n_values: Total number of values
    // rank: Number of dimensions
    Tensor(std::vector<int> dims, std::vector<double> values,
           std::vector<int> factors, int n_values, int rank);

    // Retrieves value at specified indices
    // indices: Vector of indices, one for each dimension
    // Returns the value at the specified location
    // Example: For a 2x3x3 tensor, get_value({1,2,1}) returns value at
    // [1][2][1]
    double get_value(const std::vector<int> &indices) const;

    // Returns vector containing size of each dimension
    std::vector<int> get_dimensions() const;

    // Sets value at specified indices
    // indices: Vector of indices, one for each dimension
    // value: New value to set
    // Example: For a 2x3x3 tensor, set_value({1,2,1}, 5.0) sets [1][2][1]
    // to 5.0
    void set_value(const std::vector<int> &indices, double value);

    // Performs matrix multiplication with another tensor
    // rhs: Right-hand side tensor for multiplication
    // Returns result tensor
    // Will throw an error if dimensions are incompatible
    // Requires: this->dimensions[1] == rhs.dimensions[0]
    // Result dimensions: {this->dimensions[0], rhs.dimensions[1]}
    Tensor matrix_multiplication(const Tensor &rhs) const;

    // Prints tensor structure with nested brackets and indentation
    // index: Starting index in flat array
    // dim: Current dimension being processed
    // indent: Current indentation level
    void print(int index = 0, int dim = 0, int indent = 0) const;

    // Performs 2D convolution with given kernel
    // kernel: Convolution kernel tensor
    // Returns result of convolution
    // Will throw an error if dimensions are incompatible
    // Output dimensions: {N-K+1} for each dimension, where N is input size and
    // K is kernel size
    Tensor convolve(const Tensor &kernel);

    // Performs 2D cross-correlation with given kernel
    // kernel: Cross-correlation kernel tensor
    // Returns result of cross-correlation
    // Will throw an error if dimensions are incompatible
    // Output dimensions: {N-K+1} for each dimension, where N is input size and
    // K is kernel size
    Tensor cross_correlate(const Tensor &kernel);

    // Performs elementwise multiplication with another tensor
    // rhs: Right-hand side tensor
    // Returns result tensor with same dimensions
    // Will throw an error if dimensions don't match
    Tensor elementwise_prod(const Tensor &rhs);

    // Single index operator - returns subtensor
    // index: Index into first dimension
    // Returns tensor with rank-1 dimensions
    // Will throw an error if index is invalid
    Tensor operator[](int index);
    const Tensor operator[](int index) const;

    // Multi-index operator - returns value reference
    // indices: Vector of indices for all dimensions
    // Returns reference to value at specified location
    // Will throw an error if any index is invalid
    double &operator[](const std::vector<int> &indices);
    const double &operator[](const std::vector<int> &indices) const;

    // Assignment operator - copies all data from other tensor
    // other: Source tensor to copy from
    // Returns reference to this tensor
    Tensor &operator=(const Tensor &other);

    // Scalar assignment operator - sets all elements to given value
    // val: Value to set all elements to
    // Returns reference to modified value
    double &operator=(double val);
};

#endif
