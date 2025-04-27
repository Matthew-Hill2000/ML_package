#include "tensor_view.h"

// Assigns another tensor's data to this tensor.
// creates a view that shares storage with the source tensor.
TensorView &TensorView::operator=(const TensorView &other)
{
    if (this != &other) 
    {
        storage = other.storage;
        dimensions = other.dimensions;
        strides = other.strides;
        offset = other.offset;
        n_values = other.n_values;
        rank = other.rank;
    }
    return *this;
}

// Sets all elements in the tensor to a single value.
TensorView &TensorView::operator=(double val)
{
    for (int i = 0; i < n_values; i++)
    {
        set_value_direct(i, val);
    }
    return *this;
}

// Element-wise addition
// Adds corresponding elements from two tensors with identical dimensions.
// Returns a new tensor containing the sum.
TensorView TensorView::operator+(const TensorView &other) const
{
    if (dimensions != other.dimensions)
    {
        throw std::invalid_argument("Tensor dimensions must match for addition");
    }

    TensorView result(dimensions);

    for (int i = 0; i < n_values; i++)
    {
        result.set_value_direct(i, get_value_direct(i) + other.get_value_direct(i));
    }

    return result;
}

// Element-wise subtraction
// Subtracts corresponding elements from two tensors with identical dimensions.
// Returns a new tensor containing the difference.
TensorView TensorView::operator-(const TensorView &other) const
{
    if (dimensions != other.dimensions)
    {
        throw std::invalid_argument("Tensor dimensions must match for subtraction");
    }

    TensorView result(dimensions);

    for (int i{0}; i < n_values; i++)
    {
        result.set_value_direct(i, get_value_direct(i) - other.get_value_direct(i));
    }

    return result;
}

// Element-wise multiplication
// Multiplies corresponding elements from two tensors.
// Returns a new tensor containing the element-wise product.
TensorView TensorView::operator*(const TensorView &other) const
{
    if (dimensions != other.dimensions)
    {
        throw std::invalid_argument("Tensor dimensions must match for element-wise multiplication");
    }

    TensorView result(dimensions);

    for (int i{0}; i < n_values; i++)
    {
        result.set_value_direct(i, get_value_direct(i) * other.get_value_direct(i));
    }

    return result;
}

// Element-wise division
TensorView TensorView::operator/(const TensorView &other) const
{
    if (dimensions != other.dimensions)
    {
        throw std::invalid_argument("Tensor dimensions must match for division");
    }

    TensorView result(dimensions);

    for (int i{0}; i < n_values; i++)
    {
        double divisor{other.get_value_direct(i)};
        if (divisor == 0.0)
        {
            throw std::invalid_argument("Division by zero");
        }
        result.set_value_direct(i, get_value_direct(i) / divisor);
    }

    return result;
}

// Element-wise addition assignment
TensorView &TensorView::operator+=(const TensorView &other)
{
    if (dimensions != other.dimensions)
    {
        throw std::invalid_argument("Tensor dimensions must match for += operation");
    }

    for (int i{0}; i < n_values; i++)
    {
        set_value_direct(i, get_value_direct(i) + other.get_value_direct(i));
    }
    return *this;
}

// Element-wise subtraction assignment
TensorView &TensorView::operator-=(const TensorView &other)
{
    if (dimensions != other.dimensions)
    {
        throw std::invalid_argument("Tensor dimensions must match for -= operation");
    }

    for (int i{0}; i < n_values; i++)
    {
        set_value_direct(i, get_value_direct(i) - other.get_value_direct(i));
    }

    return *this;
}

// Element-wise multiplication assignment
TensorView &TensorView::operator*=(const TensorView &other)
{
    if (dimensions != other.dimensions)
    {
        throw std::invalid_argument("Tensor dimensions must match for *= operation");
    }

    for (int i{0}; i < n_values; i++)
    {
        set_value_direct(i, get_value_direct(i) * other.get_value_direct(i));
    }

    return *this;
}

// Element-wise division assignment
TensorView &TensorView::operator/=(const TensorView &other)
{
    if (dimensions != other.dimensions)
    {
        throw std::invalid_argument("Tensor dimensions must match for /= operation");
    }

    for (int i{0}; i < n_values; i++)
    {
        double divisor{other.get_value_direct(i)};
        if (divisor == 0.0)
        {
            throw std::invalid_argument("Division by zero");
        }
        set_value_direct(i, get_value_direct(i) / divisor);
    }

    return *this;
}

// Scalar addition
TensorView TensorView::operator+(double scalar) const
{
    TensorView result(dimensions);

    for (int i{0}; i < n_values; i++)
    {
        result.set_value_direct(i, get_value_direct(i) + scalar);
    }

    return result;
}

// Scalar subtraction
TensorView TensorView::operator-(double scalar) const
{
    TensorView result(dimensions);

    for (int i{0}; i < n_values; i++)
    {
        result.set_value_direct(i, get_value_direct(i) - scalar);
    }

    return result;
}

// Scalar multiplication
TensorView TensorView::operator*(double scalar) const
{
    TensorView result(dimensions);

    for (int i{0}; i < n_values; i++)
    {
        result.set_value_direct(i, get_value_direct(i) * scalar);
    }

    return result;
}

// Scalar division
TensorView TensorView::operator/(double scalar) const
{
    if (scalar == 0.0)
    {
        throw std::invalid_argument("Division by zero");
    }

    TensorView result(dimensions);

    for (int i{0}; i < n_values; i++)
    {
        result.set_value_direct(i, get_value_direct(i) / scalar);
    }

    return result;
}

// Scalar addition assignment
TensorView &TensorView::operator+=(double scalar)
{
    for (int i{0}; i < n_values; i++)
    {
        set_value_direct(i, get_value_direct(i) + scalar);
    }

    return *this;
}

// Scalar subtraction assignment
TensorView &TensorView::operator-=(double scalar)
{
    for (int i{0}; i < n_values; i++)
    {
        set_value_direct(i, get_value_direct(i) - scalar);
    }

    return *this;
}

// Scalar multiplication assignment
TensorView &TensorView::operator*=(double scalar)
{
    for (int i{0}; i < n_values; i++)
    {
        set_value_direct(i, get_value_direct(i) * scalar);
    }

    return *this;
}

// Scalar division assignment
TensorView &TensorView::operator/=(double scalar)
{
    if (scalar == 0.0)
    {
        throw std::invalid_argument("Division by zero");
    }

    for (int i{0}; i < n_values; i++)
    {
        set_value_direct(i, get_value_direct(i) / scalar);
    }

    return *this;
}

// Equality check
bool TensorView::operator==(const TensorView &other) const
{
    if (dimensions != other.dimensions)
    {
        return false;
    }

    const double epsilon{1e-9};
    for (int i{0}; i < n_values; i++)
    {
        if (std::abs(get_value_direct(i) - other.get_value_direct(i)) > epsilon)
        {
            return false;
        }
    }

    return true;
}

// Inequality check
bool TensorView::operator!=(const TensorView &other) const
{
    return !(*this == other);
}
