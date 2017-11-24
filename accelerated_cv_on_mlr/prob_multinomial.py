# coding=utf-8
import numpy as np


def prob_multinomial(uV):
    """calc softmax probability from weight vectors x

    Args:
        uV: M p-dimensional vectors ((M, p)-shape np.float64 object)

    Returns:
        M p-dimensional probability vectors ((M, p)-shape np.float64 object)
    """
    try:
        if len(uV.shape) <= 1:
            raise ValueError("unexpected shape of input vectors matrix \n expected: >=2, actual: " +
                             str(len(uV.shape)))
    except ValueError as e:
        print(e)
        print()
        raise

    # exp(uV)
    exp_weight_vectors = np.exp(uV)
    return np.diag(1.0 / np.sum(exp_weight_vectors, axis=1)).dot(exp_weight_vectors)
