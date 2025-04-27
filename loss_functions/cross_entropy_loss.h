#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include <cmath>
#include "../tensor/tensor_view.h"
#include "loss.h"

// The CrossEntropyLoss class implements the cross-entropy loss function
// Commonly used for classification problems with probability outputs
class CrossEntropyLoss : public Loss
{
private:
    bool enable_parallelization{false}; // Flag for parallelization
 
public:
    // Default constructor
    CrossEntropyLoss() = default;

    // Computes the cross-entropy loss between output and target
    double forward(const Tensor &output, const Tensor &target);

    // Computes the gradient of the loss with respect to the output
    Tensor backward(const Tensor &output, const Tensor &target);

    virtual void set_enable_parallelization(bool enable_parallelization) override
    {
        this->enable_parallelization = enable_parallelization;
    }

};

#endif