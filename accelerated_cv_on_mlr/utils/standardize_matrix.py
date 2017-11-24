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

    average = np.average(A, axis=0)
    std = np.std(A, axis=0, ddof=1)
    standardized_matrix = (A - average) / std

    return standardized_matrix
