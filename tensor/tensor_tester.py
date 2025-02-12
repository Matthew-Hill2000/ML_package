import numpy as np
from scipy.signal import convolve2d, correlate2d


def cross_correlation(matrix, kernel):
    """Computes the cross-correlation using scipy."""
    return correlate2d(matrix, kernel, mode="valid")


def convolution(matrix, kernel):
    """Computes the convolution using scipy."""
    return convolve2d(matrix, kernel, mode="valid")


# Example usage
if __name__ == "__main__":

    matrix = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    kernel = np.array([[1, 2], [3, -1]])

    print("Cross-Correlation Result:")
    print(cross_correlation(matrix, kernel))

    print("\nConvolution Result:")
    print(convolution(matrix, kernel))
