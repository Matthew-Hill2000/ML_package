#ifndef LOSS_H
#define LOSS_H

#include <cmath>
#include "../tensor/tensor_view.h"

class Loss
{
public:
    virtual double forward(const Tensor &output, const Tensor &target) = 0;

    virtual Tensor backward(const Tensor &output, const Tensor &target) = 0;

    virtual void set_enable_parallelization(bool enable_parallelization) = 0;

    virtual ~Loss() {};
};

#endif