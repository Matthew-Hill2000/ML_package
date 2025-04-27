#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <memory>

// Forward declaration for TensorStorage
class TensorStorage;

// The TensorView class represents a view into tensor data.
// It can either own its data or reference data owned by another tensor.
// This approach avoids unnecessary copying when accessing subtensors or
// performing operations that don't need to modify the underlying data.
class TensorView
{
private:
    // Shared pointer to the underlying storage
    std::shared_ptr<TensorStorage> storage;

    // Tensor parameters
    std::vector<int> dimensions; // Size of each dimension
    std::vector<int> strides;    // Stride for each dimension
    int offset;                  // Offset into the shared storage
    int n_values;                // Total number of values in the view
    int rank;                    // Number of dimensions

    // Calculate linear index from multidimensional indices
    int calculate_index(const std::vector<int> &indices) const;

public:
    // Default constructor
    TensorView();
    // Creates a tensor with the specified shape filled with zeros.
    TensorView(const std::vector<int>& dims);
    // Copy constructor. Creates a view that shares storage with the source tensor.
    TensorView(const TensorView &other);

    // Constructor for creating a view into an existing tensor storage.
    TensorView(std::shared_ptr<TensorStorage> storage,
               std::vector<int> dims,
               std::vector<int> strides,
               int offset,
               int n_vals,
               int rnk);

    // Assigns another tensor's data to this tensor.
    TensorView &operator=(const TensorView &other);
    // Sets all elements in the tensor to a single value.
    TensorView& operator=(double val);

    // Set value using multidimensional indices
    void set_value(const std::vector<int> &indices, double value);  
    // Get value using multidimensional indices
    double get_value(const std::vector<int> &indices) const;
    // Set value direct
    void set_value_direct(int index, double value);
    // Get value direct
    double get_value_direct(int index) const;

    // Create a subtensor view (single index)
    TensorView operator[](int index);
    // Create a subtensor view (single index, const version)
    const TensorView operator[](int index) const;
    // Access using multidimensional indices
    double &operator[](const std::vector<int> &indices);
    // Access using multidimensional indices (const version)
    const double &operator[](const std::vector<int> &indices) const;

    // Mathematical operations
    // Matrix multiplication
    TensorView matrix_multiplication(const TensorView &rhs, bool enable_parallelization= false) const;
    // Element-wise product
    TensorView elementwise_prod(const TensorView &rhs) const;
    // Transpose
    TensorView transpose() const;
    // Cross correlation
    TensorView cross_correlate(const TensorView &kernel, bool enable_parallelization = false) const;
    // Full convolution
    TensorView fully_convolve(const TensorView &kernel, bool enable_parallelization= false) const;
    // Convolution
    TensorView convolve(const TensorView &kernel, bool enable_parallelization = false) const;

    // Arithmetic operators
    // Element-wise addition
    TensorView operator+(const TensorView &other) const;
    // Element-wise subtraction
    TensorView operator-(const TensorView &other) const;
    // Element-wise multiplication
    TensorView operator*(const TensorView &other) const;
    // Element-wise division
    TensorView operator/(const TensorView &other) const;
    // Element-wise addition assignment
    TensorView &operator+=(const TensorView &other);
    // Element-wise subtraction assignment
    TensorView &operator-=(const TensorView &other);
    // Element-wise multiplication assignment
    TensorView &operator*=(const TensorView &other);
    // Element-wise division assignment
    TensorView &operator/=(const TensorView &other);

    // Scalar operations
    // Scalar addition
    TensorView operator+(double scalar) const;
    // Scalar subtraction
    TensorView operator-(double scalar) const;
    // Scalar multiplication
    TensorView operator*(double scalar) const;
    // Scalar division
    TensorView operator/(double scalar) const;
    // Scalar addition assignment
    TensorView &operator+=(double scalar);
    // Scalar subtraction assignment
    TensorView &operator-=(double scalar);
    // Scalar multiplication assignment
    TensorView &operator*=(double scalar);
    // Scalar division assignment
    TensorView &operator/=(double scalar);
    // Equality operator
    bool operator==(const TensorView &other) const;
    // Inequality operator
    bool operator!=(const TensorView &other) const;
    
    // Create a deep copy of this view (when modification is necessary)
    TensorView deep_copy() const;

    // Print tensor values
    void print(int index = 0, int dim = 0, int indent = 0) const;

    // Get values
    std::vector<double> get_values() const;
    // Get strides
    std::vector<int> get_strides() const;
    // Get offset
    int get_offset() const;
    // Get number of values
    int get_n_values() const;
    // Get rank
    int get_rank() const;
    // Get dimensions
    std::vector<int> get_dimensions() const;
    // Check if this is a contiguous tensor
    bool is_contiguous() const;
    // Make a contiguous copy if needed
    TensorView make_contiguous() const;
};

// Class to manage the underlying storage for TensorView
// Provides a memory-efficient storage mechanism that can be shared among multiple views
class TensorStorage
{
private:
    std::vector<double> data;    // The actual storage for tensor values

public:
    // Constructor with size and optional initialization value
    TensorStorage(size_t size, double init_value = 0.0);

    // Constructor that takes ownership of existing data
    TensorStorage(std::vector<double> &&values);

    // Copy constructor is deleted to prevent copying
    TensorStorage(const TensorStorage &other) = delete; 

    // Access with bounds checking
    double &at(int index);
    // Const access with bounds checking
    const double &at(int index) const;

    // Get the number of elements in storage
    size_t size() const;

    // Direct access to the underlying data vector
    std::vector<double> &get_data();
    // Const direct access to the underlying data vector
    const std::vector<double> &get_data() const;
};

// define Tensor as an alias for TensorView
typedef TensorView Tensor;

#endif