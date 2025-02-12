#include "tensor/tensor_new.h"
#include <cassert>
#include <iostream>

void test_construction_and_indexing() {
  // Test 2x3 matrix construction
  Tensor matrix({2, 3});

  // Set some values

  matrix[{0, 0}] = 1.0;
  matrix[{0, 1}] = 2.0;
  matrix[{0, 2}] = 3.0;
  matrix[{1, 0}] = 4.0;
  matrix[{1, 1}] = 5.0;
  matrix[{1, 2}] = 6.0;

  std::cout << "2x3 Matrix:" << std::endl;
  matrix.print();
}

void test_matrix_multiplication() {
  // Create 2x3 matrix A
  Tensor A({2, 2, 2});
  A[{0, 0, 0}] = 1;
  A[{0, 0, 1}] = 2;
  A[{0, 1, 0}] = 3;
  A[{0, 1, 1}] = 4;
  A[{1, 0, 0}] = 5;
  A[{1, 0, 1}] = 6;
  A[{1, 1, 0}] = 7;
  A[{1, 1, 1}] = 8;
  std::cout << "Matrix A:" << std::endl;
  A.print();
  A[0].print();
  A[1].print();

  Tensor C = A[0].matrix_multiplication(A[1]);
  std::cout << "Matrix multiplication result:" << std::endl;
  C.print();
}

void test_elementwise_operations() {
  Tensor A({2, 2});
  A[{0, 0}] = 1.0;
  A[{0, 1}] = 2.0;
  A[{1, 0}] = 3.0;
  A[{1, 1}] = 4.0;

  Tensor B({2, 2});
  B[{0, 0}] = 2.0;
  B[{0, 1}] = 3.0;
  B[{1, 0}] = 4.0;
  B[{1, 1}] = 5.0;

  std::cout << "Elementwise multiplication:" << std::endl;
  Tensor C = A.elementwise_prod(B);
  C.print();
}

void test_assignment() {
  Tensor A({2, 2});
  A = 3.0;
  std::cout << "Matrix after scalar assignment:" << std::endl;
  A.print();

  Tensor B({2, 2});
  B[{0, 0}] = 1.0;
  B[{0, 1}] = 2.0;
  B[{1, 0}] = 3.0;
  B[{1, 1}] = 4.0;

  A = B;
  std::cout << "Matrix after copy assignment:" << std::endl;
  A.print();
}

void test_high_dimensions() {
  // 3D tensor
  Tensor tensor3d({2, 2, 2});
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        tensor3d[{i, j, k}] = i + j + k;
      }
    }
  }
  std::cout << "3D Tensor:" << std::endl;
  tensor3d.print();
}

void test_edge_cases() {
  // Single element tensor
  Tensor scale({1});
  scale[{0}] = 42.0;
  std::cout << "Scalar tensor:" << std::endl;
  scale.print();

  // Test large dimensions
  try {
    Tensor large({1000000, 1000000});
  } catch (const std::exception &e) {
    std::cout << "Expected error for large tensor: " << e.what() << std::endl;
  }
}

void test_error_handling() {
  Tensor A({2, 2});

  // Test invalid index
  try {
    A[{2, 0}] = 1.0;
  } catch (const std::out_of_range &e) {
    std::cout << "Expected out of range error: " << e.what() << std::endl;
  }

  // Test dimension mismatch in operations
  Tensor B({2, 3});
  try {
    A.elementwise_prod(B);
  } catch (const std::runtime_error &e) {
    std::cout << "Expected dimension mismatch error: " << e.what() << std::endl;
  }
}

void test_convolution_and_correlation() {
  // Create a 4x4 input tensor
  Tensor input({4, 4});
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      input[{i, j}] = i + j;
    }
  }

  // Create a 2x2 kernel
  Tensor kernel({2, 2});
  kernel[{0, 0}] = 1;
  kernel[{0, 1}] = 2;
  kernel[{1, 0}] = 3;
  kernel[{1, 1}] = -1;

  std::cout << "Input tensor:" << std::endl;
  input.print();
  std::cout << "\nKernel:" << std::endl;
  kernel.print();

  Tensor conv_result = input.convolve(kernel);
  std::cout << "\nConvolution result:" << std::endl;
  conv_result.print();

  Tensor corr_result = input.cross_correlate(kernel);
  std::cout << "\nCross-correlation result:" << std::endl;
  corr_result.print();
}

int main() {
  /*std::cout << "\n=== Basic Construction and Indexing ===\n";*/
  /*test_construction_and_indexing();*/

  /*std::cout << "\n=== Matrix Multiplication ===\n";*/
  /*test_matrix_multiplication();*/

  /*std::cout << "\n=== Elementwise Operations ===\n";*/
  /*test_elementwise_operations();*/

  // std::cout << "\n=== Assignment Operations ===\n";
  // test_assignment();

  /*std::cout << "\n=== Higher Dimensional Tensors ===\n";*/
  /*test_high_dimensions();*/

  // std::cout << "\n=== Edge Cases ===\n";
  // test_edge_cases();

  // std::cout << "\n=== Error Handling ===\n";
  // test_error_handling();

  std::cout << "\n=== Convolution and Correlation ===\n";
  test_convolution_and_correlation();

  /*Tensor test({2, 4, 3, 3});*/
  /*test.print();*/

  return 0;
}
