# coding=utf-8
import numpy as np


def standardize_matrix(A):
    """standardize matrix along column axis

    Args:
        A: matrix to be standardized (2-dimensional np.float64 array)

    Returns:
        standardized matrix
    """
    try:
        if type(A) is not np.ndarray:
            msg = "unexpected type of weight vector\n" \
                  "expected: numpy.ndarray, actual: " + str(type(A))
            raise ValueError(msg)

        elif len(A.shape) is not 2:
            msg = "unexpected dimension of input matrix\n" \
                  " expected: dim A == 2\n" \
                  " actual: dim A = {0}".format(len(A.shape))
            raise ValueError(msg)

    except ValueError as e:
        print(e)
        print()
        raise

    standardized_matrix = A.copy()

    for index in range(standardized_matrix.shape[1]):
        average = np.average(A[:, index])
        std = np.std(A[:, index], ddof=1)
        standardized_matrix[:, index] = (A[:, index] - average) / std

    return standardized_matrix
