#include "tensor_view.h"


// Returns a view accros one of the first dimensions.
TensorView TensorView::operator[](int index)
{
    // Check if the index is within bounds for the first dimension
    if (index < 0 || index >= dimensions[0])
    {
        throw std::out_of_range("Index out of bounds for first dimension");
    }
    if (rank == 1)
    {
        // For 1D tensor, return a scalar view
        return TensorView(storage, {1}, {1}, offset + index * strides[0], 1, 1);
    }
    else
    {
        // For higher rank tensors, create a view with one fewer dimension
        std::vector<int> new_dims(dimensions.begin() + 1, dimensions.end());
        std::vector<int> new_strides(strides.begin() + 1, strides.end());
        int new_offset(offset + index * strides[0]);
        int new_n_values{n_values / dimensions[0]};
        int new_rank{rank - 1};

        return TensorView(storage, new_dims, new_strides, new_offset, new_n_values, new_rank);
    }
}

// Create a subtensor view (single index, const version)
const TensorView TensorView::operator[](int index) const
{
    // Check if the index is within bounds for the first dimension
    if (index < 0 || index >= dimensions[0])
    {
        throw std::out_of_range("Index out of bounds for first dimension");
    }

    if (rank == 1)
    {
        // For 1D tensor, return a scalar view
        return TensorView(storage, {1}, {1}, offset + index * strides[0], 1, 1);
    }
    else
    {
        // For higher rank tensors, create a view with one fewer dimension
        std::vector<int> new_dims(dimensions.begin() + 1, dimensions.end());
        std::vector<int> new_strides(strides.begin() + 1, strides.end());
        int new_offset{offset + index * strides[0]};
        int new_n_values{n_values / dimensions[0]};
        int new_rank{rank - 1};

        return TensorView(storage, new_dims, new_strides, new_offset, new_n_values, new_rank);
    }
}

// Direct access to a tensor element using vector of indices.
double &TensorView::operator[](const std::vector<int> &indices)
{
    int index = calculate_index(indices);
    return storage->at(index);
}

// Direct access to a tensor element using vector of indices. (const version)
const double &TensorView::operator[](const std::vector<int> &indices) const
{
    int index = calculate_index(indices);
    return storage->at(index);
}

// Calculate linear index from multidimensional indices
int TensorView::calculate_index(const std::vector<int> &indices) const
{
    // Check if the number of indices matches the tensor rank
    if (static_cast<int>(indices.size()) != rank)
    {
        throw std::invalid_argument("Number of indices doesn't match tensor rank");
    }

    // Check bounds
    for (int i{0}; i < static_cast<int>(indices.size()); i++)
    {
        if (indices[i] < 0 || indices[i] >= dimensions[i])
        {
            throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
        }
    }

    // Calculate linear index using strides
    int index{offset};
    for (int i{0}; i < rank; i++)
    {
        index += indices[i] * strides[i];
    }

    return index;
}

// Sets the value at the specified position in the tensor.
void TensorView::set_value(const std::vector<int> &indices, double value)
{
    int index = calculate_index(indices);
    storage->at(index) = value;
}

// Retrieves the value at the specified position in the tensor.
double TensorView::get_value(const std::vector<int> &indices) const
{
    int index = calculate_index(indices);
    return storage->at(index);
}

// Sets a value using a direct index within this view.
// For contiguous tensors, maps directly to storage with offset.
// For non-contiguous tensors, converts linear index to multidimensional
// indices and then to the correct storage location.
void TensorView::set_value_direct(int linear_index, double value) {
    // Check if the linear index is within bounds
    if (linear_index < 0 || linear_index >= n_values)
    {
        throw std::out_of_range("Linear index out of bounds");
    }
        storage->at(offset + linear_index) = value;
}

// Retrieves a value using a direct index within this view.
// For contiguous tensors, maps directly to storage with offset.
// For non-contiguous tensors, converts linear index to multidimensional
// indices and then to the correct storage location.
double TensorView::get_value_direct(int linear_index) const
{
    // Check if the linear index is within bounds
    if (linear_index < 0 || linear_index >= n_values)
    {
        throw std::out_of_range("Linear index out of bounds");
    }
        return storage->at(offset + linear_index);
}

// Get all values as a flat vector (creates a copy)
std::vector<double> TensorView::get_values() const
{
    std::vector<double> result(n_values);

    for (int i{0}; i < n_values; i++)
    {
        result[i] = get_value_direct(i);
    }

    return result;
}

// Get strides
std::vector<int> TensorView::get_strides() const
{
    return strides;
}

// Get offset
int TensorView::get_offset() const
{
    return offset;
}

// Get number of values
int TensorView::get_n_values() const
{
    return n_values;
}

// Get rank
int TensorView::get_rank() const
{
    return rank;
}

// Get dimensions
std::vector<int> TensorView::get_dimensions() const
{
    return dimensions;
}