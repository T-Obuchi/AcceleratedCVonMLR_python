# coding=utf-8
import numpy as np


def prob_logit(x):
    """calc accelerated_cv_on_mlr probability from weight vector x

    Args:
        x: M dimensional vector ((M,)-shape np.float64 object)

    Returns:
        probability ((M, 2)-shape np.float64 object)
    """
    try:
        if len(x.shape) != 1:
            raise ValueError("unexpected shape of input vector\nexpected:" + str(1) + ", actual: " + str(len(x.shape)))
    except ValueError as e:
        print(e)
        print()
        raise

    x = 1.0 * np.exp(-x)

    probability = np.concatenate(
        (
            (x / (1.0 + x)).reshape(x.shape[0], 1),
            (1.0 / (1.0 + x)).reshape(x.shape[0], 1)
        ),
        axis=1
    )

    return probability
