#include "tensor_view.h"
#include <omp.h>

// Performs matrix multiplication for 2D tensors.
// Returns a new tensor containing the matrix multiplication result.
TensorView TensorView::matrix_multiplication(const TensorView &rhs, bool enable_parallelization) const
{
    // Check dimensions are compatible
    if (rank != 2 || rhs.rank != 2 || dimensions[1] != rhs.dimensions[0])
    {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }

    // Create result tensor
    std::vector<int> result_dims = {dimensions[0], rhs.dimensions[1]};
    TensorView result(result_dims);

    // Perform matrix multiplication
    #pragma omp parallel for collapse(2) if(enable_parallelization)
    for (int i = 0; i < dimensions[0]; i++)
    {
        for (int j = 0; j < rhs.dimensions[1]; j++)
        {
            double sum{0.0};
            for (int k = 0; k < dimensions[1]; k++)
            {
                sum += get_value({i, k}) * rhs.get_value({k, j});
            }
            result.set_value({i, j}, sum);
        }
    }

    return result;
}

// Performs element-wise multiplication between tensors.
// Each element in the result is the product of corresponding elements.
// Returns a new tensor containing the element-wise product.
TensorView TensorView::elementwise_prod(const TensorView &rhs) const
{
    // Ensure dimensions match
    if (dimensions != rhs.dimensions)
    {
        throw std::invalid_argument("Dimensions must match for elementwise multiplication");
    }

    // Create result tensor
    TensorView result(dimensions);

    // Multiply elements
    for (int i{0}; i < n_values; i++)
    {
        result.set_value_direct(i, get_value_direct(i) * rhs.get_value_direct(i));
    }

    return result;
}

// Creates a view with transposed dimensions and strides.
// reverses the order of dimensions.
// Returns a view into the same storage with modified strides.
TensorView TensorView::transpose() const
{
    if (rank <= 1)
    {
        // Scalar or 1D tensor remains unchanged
        return *this;
    }

    //  reverse all dimensions and strides
    std::vector<int> transposed_dims(dimensions.rbegin(), dimensions.rend());
    std::vector<int> transposed_strides(strides.rbegin(), strides.rend());

    return TensorView(storage, transposed_dims, transposed_strides, offset, n_values, rank);
}

// performs 2D cross-correlation with the kernel tensor
// Returns a new tensor containing the cross-correlation result
TensorView TensorView::cross_correlate(const TensorView &kernel, bool enable_parallelization) const
{
    // Ensure dimensions match
    if (rank != kernel.rank)
    {
        throw std::runtime_error("Tensor and kernel must have same rank");
    }

    if (rank != 2)
    {
        throw std::runtime_error("Cross-correlation is only implemented for 2D tensors");
    }

    // Calculate output dimensions
    std::vector<int> output_dims;
    for (int i{0}; i < rank; i++)
    {
        output_dims.push_back(dimensions[i] - kernel.dimensions[i] + 1);
        if (output_dims[i] <= 0)
        {
            throw std::runtime_error("Kernel too large for dimension " + std::to_string(i));
        }
    }

    TensorView result(output_dims);

    #pragma omp parallel for collapse(2) if(enable_parallelization)
    for (int i = 0; i < output_dims[0]; i++)
    {
        for (int j = 0; j < output_dims[1]; j++)
        {
            double sum{0.0};
            for (int ki = 0; ki < kernel.dimensions[0]; ki++)
            {
                for (int kj = 0; kj < kernel.dimensions[1]; kj++)
                {
                    sum += get_value({i + ki, j + kj}) * kernel.get_value({ki, kj});
                }
            }
            result.set_value({i, j}, sum);
        }
    }

    return result;
}

// Performs convolution with all possible overlaps between input and kernel.
// Output dimensions are: input_dims + kernel_dims - 1
// Returns a new tensor containing the full convolution result.
TensorView TensorView::fully_convolve(const TensorView &kernel, bool enable_parallelization) const
{
    if (rank != kernel.rank)
    {
        throw std::runtime_error("Tensor and kernel must have same rank");
    }

    if (rank != 2)
    {
        throw std::runtime_error("Full convolution is only implemented for 2D tensors");
    }

    // Calculate output dimensions
    std::vector<int> output_dims;
    for (int i{0}; i < rank; i++)
    {
        output_dims.push_back(dimensions[i] + kernel.dimensions[i] - 1);
    }

    TensorView result(output_dims);

    // Perform full convolution for 2D tensors
    #pragma omp parallel for collapse(2) if(enable_parallelization)
    for (int i = 0; i < output_dims[0]; i++)
    {
        for (int j = 0; j < output_dims[1]; j++)
        {
            double sum{0.0};
            for (int ki = 0; ki < kernel.dimensions[0]; ki++)
            {
                for (int kj =0; kj < kernel.dimensions[1]; kj++)
                {
                    int x{i - ki};
                    int y{j - kj};
                    if (x >= 0 && x < dimensions[0] && y >= 0 && y < dimensions[1])
                    {
                        sum += get_value({x, y}) * kernel.get_value({ki, kj});
                    }
                }
            }
            result.set_value({i, j}, sum);
        }
    }
    return result;
}

// Performs standard convolution operation (flips kernel first).
// Mathematically equivalent to cross-correlation with flipped kernel.
// Output dimensions are: input_dims - kernel_dims + 1
// Returns a new tensor containing the convolution result.
TensorView TensorView::convolve(const TensorView &kernel, bool enable_parallelization) const
{
    if (rank != kernel.rank)
    {
        throw std::runtime_error("Tensor and kernel must have same rank");
    }

    if (rank != 2)
    {
        throw std::runtime_error("Convolution is only implemented for 2D tensors");
    }

    // Calculate output dimensions
    std::vector<int> output_dims;
    for (int i{0}; i < rank; i++)
    {
        output_dims.push_back(dimensions[i] - kernel.dimensions[i] + 1);
        if (output_dims[i] <= 0)
        {
            throw std::runtime_error("Kernel too large for dimension " + std::to_string(i));
        }
    }

    TensorView result(output_dims);

    // Perform 2D convolution for 2D tensors
    #pragma omp parallel for collapse(2) if(enable_parallelization)
    for (int i = 0; i < output_dims[0]; i++)
    {
        for (int j = 0; j < output_dims[1]; j++)
        {
            double sum{0.0};
            for (int ki = 0; ki < kernel.dimensions[0]; ki++)
            {
                for (int kj = 0; kj < kernel.dimensions[1]; kj++)
                {
                    sum += get_value({i + ki, j + kj}) *
                            kernel.get_value({kernel.dimensions[0] - 1 - ki,
                                                kernel.dimensions[1] - 1 - kj});
                }
            }
            result.set_value({i, j}, sum);
        }
    }
    

    return result;
}
