#ifndef ReLU_H
#define ReLU_H

#include "../../layer.h"

class ReLU : public Layer
{
private:
    std::vector<int> input_shape;
    std::vector<int> output_shape;

    bool enable_parallelization; // Flag for parallelization

public:
    ReLU() {};

    Tensor forward(const Tensor &input) override;
    Tensor backward(Tensor &output_gradients) override;
    void update_weights(double learning_rate) override {};
    void reset_gradients() override;
    void set_enable_parallelization(bool enable_parallelization);
};

#endif