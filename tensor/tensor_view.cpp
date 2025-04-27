#include "tensor_view.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <string>

// Creates a scalar tensor with value 0.
TensorView::TensorView()
    : dimensions{1}, strides{1}, offset(0), n_values(1), rank(1)
{
    // Create new storage for this tensor
    storage = std::make_shared<TensorStorage>(1, 0.0);
}

// Creates a tensor with the specified shape filled with zeros.
TensorView::TensorView(const std::vector<int>& dims)
    : dimensions(dims), strides(), offset(0), n_values(1), rank(dims.size())
{
    // allocate storage for strides
    strides.resize(rank);

    if (rank == 0)
    {
        n_values = 0; // No dimensions, no values
    }
    else if (rank == 1)
    {
        n_values = dimensions[0]; // 1D tensor, just the size of the first dimension
        strides[0] = 1; // Stride for the only dimension is 1
    }
    else
    {
        // Calculate strides for each dimension
        strides[rank - 1] = 1; // Stride for the last dimension is 1
        n_values = dimensions[rank - 1]; // Start with the size of the last dimension

        for (int i(rank - 2); i >= 0; i--)
        {
            strides[i] = strides[i + 1] * dimensions[i + 1]; // Stride is the product of the sizes of all dimensions to the right
            n_values *= dimensions[i]; // Update total number of values
        }
    }

    // Create new storage for this tensor
    storage = std::make_shared<TensorStorage>(n_values, 0.0);
}

// Creates a view that shares storage with the source tensor.
// This performs a shallow copy - no data is duplicated. (technically a mixed copy)
TensorView::TensorView(const TensorView &other)
    : storage(other.storage), dimensions(other.dimensions),
      strides(other.strides), offset(other.offset),
      n_values(other.n_values), rank(other.rank)
{
}

// Constructor for creating a view
// Creates a tensor view into existing storage with custom dimensions, strides, and offset.
TensorView::TensorView(std::shared_ptr<TensorStorage> storage,
                       std::vector<int> dims,
                       std::vector<int> strides,
                       int offset,
                       int n_vals,
                       int rnk)
    : storage(storage), dimensions(dims), strides(strides),
      offset(offset), n_values(n_vals), rank(rnk)
{
}

// Returns a new tensor with identical values but independent storage.
TensorView TensorView::deep_copy() const
{
    // Create a new tensor with the same dimensions
    TensorView result(dimensions);

    // Copy all values with direct linear indexing
    for (int i = 0; i < n_values; i++)
    {
        result.set_value_direct(i, get_value_direct(i));
    }

    return result;
}

// Print the tensor with proper formatting
void TensorView::print(int index, int dim, int indent) const
{
    std::string indentation(indent, ' ');
    std::cout << indentation << "[";

    if (dim == rank - 1)
    {
        // Print the innermost dimension
        for (int i{0}; i < dimensions[dim]; i++)
        {
            std::vector<int> indices(rank);
            for (int j{0}; j < dim; j++)
            {
                indices[j] = (index / strides[j]) % dimensions[j];
            }
            indices[dim] = i;

            std::cout << get_value(indices);
            if (i < dimensions[dim] - 1)
            {
                std::cout << ", ";
            }
        }
    }
    else
    {
        // Recursively print nested dimensions
        std::cout << std::endl;
        for (int i{0}; i < dimensions[dim]; i++)
        {
            int next_index{index + i * strides[dim]};
            print(next_index, dim + 1, indent + 2);
            if (i < dimensions[dim] - 1)
            {
                std::cout << "," << std::endl;
            }
        }
        std::cout << std::endl
                  << indentation;
    }

    std::cout << "]";
    if (dim == 0)
    {
        std::cout << std::endl;
    }
}

