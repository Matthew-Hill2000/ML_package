#include "tensor_view.h"

// initializes storage with specified size and fills with initial value
TensorStorage::TensorStorage(size_t size, double init_value)
    : data(size, init_value)
{
}

// Constructor that takes ownership of existing data using move semantics
TensorStorage::TensorStorage(std::vector<double> &&values)
    : data(std::move(values))
{
}

// Access element at specified index 
double &TensorStorage::at(int index)
{
    if (index < 0 || index >= static_cast<int>(data.size()))
    {
        throw std::out_of_range("Storage index out of bounds");
    }
    return data[index];
}

// Const access element at specific index
const double &TensorStorage::at(int index) const
{
    if (index < 0 || index >= static_cast<int>(data.size()))
    {
        throw std::out_of_range("Storage index out of bounds");
    }
    return data[index];
}

// Returns the number of elements in the storage
size_t TensorStorage::size() const
{
    return data.size();
}

// Provides mutable direct access to the underlying data vector
std::vector<double> &TensorStorage::get_data()
{
    return data;
}

// Provides const direct access to the underlying data vector
const std::vector<double> &TensorStorage::get_data() const
{
    return data;
}