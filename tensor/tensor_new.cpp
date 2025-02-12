#include "tensor_new.h"
#include <iostream>
#include <vector>

// Default constructor
// Creates a 1D tensor of size 1 initialized to 0
Tensor::Tensor() {
  dimensions = {1};
  factors = {1};
  values = {0};
  n_values = 1;
  rank = 1;
}

// Constructor taking dimensions vector
// Creates tensor with specified dimensions, calculating appropriate factors
// for indexing and initializing all values to 0
Tensor::Tensor(std::vector<int> dims) {
  // Set dimensions to be equal to the input dims : {2,2,3}
  dimensions = dims;
  // Set rank to represent the number of dimensions : 3
  rank = dims.size();
  // Set the size of factors to be equal to the number of dimensions: {x, x, x}
  factors.resize(rank);

  // Calculate multiplicative factors for indexing
  // For single dimension tensors, factor is 1
  // For multi-dimensional tensors, factors are calculated from right to left
  // Example: for dims {2,2,3}, factors will be {6,3,1}
  if (rank == 1) {
    factors[0] = 1;
    n_values = dimensions[0];
  } else {
    factors[rank - 1] = 1; // Rightmost factor is always 1
    factors[rank - 2] =
        dimensions[rank - 1]; // Second from right is size of last dimension
    n_values = dimensions[rank - 1] * dimensions[rank - 2];

    // Calculate remaining factors right to left
    // Each factor is the product of the next factor and its dimension
    for (int i{rank - 3}; i >= 0; i--) {
      factors[i] = factors[i + 1] * dimensions[i + 1];
      n_values *= dimensions[i];
    }
  }
  values.resize(n_values, 0); // Initialize all values to 0
}

// Constructor that takes all tensor attributes directly
// Used primarily for creating tensor slices and internal operations
Tensor::Tensor(std::vector<int> dims, std::vector<double> vals,
               std::vector<int> facs, int n_vals, int rnk) {
  dimensions = dims;
  values = vals;
  factors = facs;
  n_values = n_vals;
  rank = rnk;
}

// Sets the value at a specific location in the tensor
// Converts n-dimensional indices to flat array index using factors
// Example: for indices {1,2,1} and factors {6,3,1}, index = 1*6 + 2*3 + 1*1 =
// 13
void Tensor::set_value(const std::vector<int> &indices, double value) {
  int index = 0;
  for (int i = 0; i < rank; i++) {
    index += factors[i] * indices[i];
  }
  values[index] = value;
}

// Gets the value at a specific location in the tensor
// Uses same indexing scheme as set_value
double Tensor::get_value(const std::vector<int> &indices) const {
  int index = 0;
  for (int i = 0; i < rank; i++) {
    index += factors[i] * indices[i];
  }
  return values[index];
}

// Returns the dimensions of the tensor
std::vector<int> Tensor::get_dimensions() const { return dimensions; }

// Performs matrix multiplication with another tensor
// Requires both tensors to be 2D and have compatible dimensions
// Returns a new tensor with the result
Tensor Tensor::matrix_multiplication(const Tensor &rhs) const {
  // Check if dimensions are compatible for matrix multiplication
  if (dimensions[1] != rhs.dimensions[0]) {
    throw std::runtime_error("Invalid dimensions for matrix multiplication");
  }

  // Create result tensor with appropriate dimensions
  std::vector<int> result_dims = {dimensions[0], rhs.dimensions[1]};
  Tensor result(result_dims);

  // Perform matrix multiplication
  for (int i = 0; i < dimensions[0]; i++) {
    for (int j = 0; j < rhs.dimensions[1]; j++) {
      double sum = 0;
      for (int k = 0; k < dimensions[1]; k++) {
        sum += (*this)[{i, k}] * rhs[{k, j}];
      }
      result[{i, j}] = sum;
    }
  }
  return result;
}

// Prints the tensor with appropriate formatting and indentation
// Recursively handles multiple dimensions
void Tensor::print(int index, int dim, int indent) const {
  std::string indentation(indent, ' ');
  std::cout << indentation << "[";

  // For the innermost dimension, print values directly
  if (dim == rank - 1) {
    for (int i = 0; i < dimensions[dim]; ++i) {
      std::cout << values[index + i];
      if (i < dimensions[dim] - 1) {
        std::cout << ", ";
      }
    }
  } else {
    // For outer dimensions, recursively print nested structures
    std::cout << "\n";
    for (int i = 0; i < dimensions[dim]; ++i) {
      print(index + i * factors[dim], dim + 1, indent + 1);
      if (i < dimensions[dim] - 1) {
        std::cout << ",\n";
      }
    }
    std::cout << "\n" << indentation;
  }

  std::cout << "]";
  if (dim == 0) {
    std::cout << "\n";
  }
}

// Returns a subtensor by indexing into the first dimension
// Creates a new tensor with rank-1 dimensions
Tensor Tensor::operator[](int index) {
  // Check for valid index
  if (index < 0 || index >= dimensions[0]) {
    throw std::out_of_range("Index out of bounds");
  }

  if (rank == 0) {
    throw std::out_of_range("Cant index scalar");
  }

  // Prepare attributes for the subtensor
  std::vector<int> new_factors;
  std::vector<int> new_dimensions;
  std::vector<double> new_values;

  // Handle special case for 1D tensors
  if (factors.size() == 1) {
    new_factors = factors;
    new_dimensions = {1};
    new_values = std::vector<double>{values[index]};
  } else {
    // Create subtensor attributes by removing first dimension
    new_factors = std::vector<int>(factors.begin() + 1, factors.end());
    new_dimensions = std::vector<int>(dimensions.begin() + 1, dimensions.end());
    new_values = std::vector<double>(values.begin() + index * factors[0],
                                     values.begin() + (index + 1) * factors[0]);
  }
  int new_n_values{n_values / dimensions[0]};
  int new_rank{rank - 1};

  return Tensor(new_dimensions, new_values, new_factors, new_n_values,
                new_rank);
}

// Const version of subtensor indexing operator
const Tensor Tensor::operator[](int index) const {
  // Implementation same as non-const version
  if (index < 0 || index >= dimensions[0]) {
    throw std::out_of_range("Index out of bounds");
  }

  if (rank == 0) {
    throw std::out_of_range("Cant index scalar");
  }

  std::vector<int> new_factors;
  std::vector<int> new_dimensions;
  std::vector<double> new_values;

  if (factors.size() == 1) {
    new_factors = factors;
    new_dimensions = {1};
    new_values = std::vector<double>{values[index]};
  } else {
    new_factors = std::vector<int>(factors.begin() + 1, factors.end());
    new_dimensions = std::vector<int>(dimensions.begin() + 1, dimensions.end());
    new_values = std::vector<double>(values.begin() + index * factors[0],
                                     values.begin() + (index + 1) * factors[0]);
  }
  int new_n_values{n_values / dimensions[0]};
  int new_rank{rank - 1};

  return Tensor(new_dimensions, new_values, new_factors, new_n_values,
                new_rank);
}

// Multi-dimensional indexing operator
// Returns reference to value at specified indices
double &Tensor::operator[](const std::vector<int> &indices) {
  // Verify indices match tensor rank
  if (indices.size() != rank) {
    throw std::out_of_range("Index dimensions don't match tensor rank");
  }

  // Check each index is within bounds
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] < 0 || indices[i] >= dimensions[i]) {
      throw std::out_of_range("Index out of bounds for dimension " +
                              std::to_string(i));
    }
  }

  // Calculate flat array index using factors
  int index = 0;
  for (int i = 0; i < rank; i++) {
    index += factors[i] * indices[i];
  }
  return values[index];
}

// Const version of multi-dimensional indexing operator
const double &Tensor::operator[](const std::vector<int> &indices) const {
  // Implementation same as non-const version
  if (indices.size() != rank) {
    throw std::out_of_range("Index dimensions don't match tensor rank");
  }

  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] < 0 || indices[i] >= dimensions[i]) {
      throw std::out_of_range("Index out of bounds for dimension " +
                              std::to_string(i));
    }
  }

  int index = 0;
  for (int i = 0; i < rank; i++) {
    index += factors[i] * indices[i];
  }
  return values[index];
}

// Assignment operator for copying another tensor
Tensor &Tensor::operator=(const Tensor &other) {
  if (this != &other) { // Prevent self-assignment
    dimensions = other.dimensions;
    values = other.values;
    factors = other.factors;
    n_values = other.n_values;
    rank = other.rank;
  }
  return *this;
}

// Assignment operator for setting all values to a scalar
double &Tensor::operator=(double val) {
  static double result = val;
  for (int i = 0; i < n_values; i++) {
    values[i] = val;
  }
  return result;
}

// Performs 2D convolution with specified kernel
// Slides kernel over tensor and computes sum of element-wise products
Tensor Tensor::cross_correlate(const Tensor &kernel) {
  if (rank != kernel.rank) {
    throw std::runtime_error("Tensor and kernel must have same rank");
  }

  // Calculate dimensions of output tensor
  std::vector<int> output_dims;
  for (int i = 0; i < rank; i++) {
    output_dims.push_back(dimensions[i] - kernel.dimensions[i] + 1);
    if (output_dims[i] <= 0) {
      throw std::runtime_error("Kernel too large for dimension " +
                               std::to_string(i));
    }
  }

  Tensor result(output_dims);

  // Perform 2D cross-correlation
  if (rank == 2) {
    for (int i = 0; i < output_dims[0]; i++) {
      for (int j = 0; j < output_dims[1]; j++) {
        double sum = 0.0;
        // Compute sum of element-wise products
        for (int ki = 0; ki < kernel.dimensions[0]; ki++) {
          for (int kj = 0; kj < kernel.dimensions[1]; kj++) {
            sum += (*this)[{i + ki, j + kj}] * kernel[{ki, kj}];
          }
        }
        result[{i, j}] = sum;
      }
    }
  }

  return result;
}

// Performs 2D convolution with specified kernel
// Similar to cross_correlation but with kernel flipped in both dimensions
Tensor Tensor::convolve(const Tensor &kernel) {
  if (rank != kernel.rank) {
    throw std::runtime_error("Tensor and kernel must have same rank");
  }

  std::vector<int> output_dims;
  for (int i = 0; i < rank; i++) {
    output_dims.push_back(dimensions[i] - kernel.dimensions[i] + 1);
    if (output_dims[i] <= 0) {
      throw std::runtime_error("Kernel too large for dimension " +
                               std::to_string(i));
    }
  }

  Tensor result(output_dims);

  // Perform 2D convolution with flipped kernel
  if (rank == 2) {
    for (int i = 0; i < output_dims[0]; i++) {
      for (int j = 0; j < output_dims[1]; j++) {
        double sum = 0.0;
        for (int ki = 0; ki < kernel.dimensions[0]; ki++) {
          for (int kj = 0; kj < kernel.dimensions[1]; kj++) {
            sum += (*this)[{i + ki, j + kj}] *
                   kernel[{kernel.dimensions[0] - 1 - ki,
                           kernel.dimensions[1] - 1 - kj}];
          }
        }
        result[{i, j}] = sum;
      }
    }
  }

  return result;
}

// Performs element-wise multiplication with another tensor
// Requires both tensors to have identical dimensions
Tensor Tensor::elementwise_prod(const Tensor &rhs) {
  if (dimensions != rhs.dimensions) {
    throw std::runtime_error(
        "Tensors must have same dimensions for elementwise multiplication");
  }

  Tensor result(dimensions);
  // Multiply corresponding elements
  for (int i = 0; i < n_values; i++) {
    result.values[i] = values[i] * rhs.values[i];
  }
  return result;
}
