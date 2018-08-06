# coding=utf-8
import numpy as np
from accelerated_cv_on_mlr.prob_logit import prob_logit


def acv_logit(w, X, Ycode, lambda2=0.0):
    """ An approximate leave-one-out estimator of predictive likelihood
    for accelerated_cv_on_mlr regression with elastic net regularization[1]

    Compute and return an approximate leave-one-out estimator (LOOE) and
    its standard error of predivtive likelihood for accelerated_cv_on_mlr regression
    penalized by elastic net regularization.

    Args:
        w: weight vector ((1,N)-shape np.float64 array)
        X: input feature matrix ((M, N)-shape np.float64 array)
        Ycode: binary matrix representing the class to which the corresponding feature vector belongs((M, 2)-shape np.int64 array)
        lambda2: coefficient of the l2 regularization term (float value). Default value is zero.

    Returns:
        LOOE, ERR (float, float)

    References:
        [1]: T. Obuchi and Y. Kabashima, arXiv:1711.05420

    Note:
        In this method, expected shape is different from the MATLAB implementation.
        (MATLAB -> w is a (N,1) matrix, Python -> w is (1,N)-shape np.float64 array)
    """
    try:

        # type check
        if type(w) is not np.ndarray:
            msg = "unexpected type of weight vector\n" \
                  "expected: numpy.ndarray, actual: " + str(type(w))
            raise ValueError(msg)
        elif type(X) is not np.ndarray:
            msg = "unexpected type of input feature matrix\n" \
                  "expected: numpy.ndarray, actual: " + str(type(X))
            raise ValueError(msg)
        elif type(Ycode) is not np.ndarray:
            msg = "unexpected type of claass representative matrix\n" \
                  "expected: numpy.ndarray, actual: " + str(type(Ycode))
            raise ValueError(msg)

        # check length of shape
        if len(w.shape) <= 1 or 3 <= len(w.shape):
            raise ValueError("unexpected length of shape of weight vector\n expected: 2, actual: " + str(len(w.shape)))
        elif len(X.shape) <= 1 or 3 <= len(X.shape):
            raise ValueError("unexpected length of shape of feature matrix\n expected: 2, actual: " + str(len(X.shape)))
        elif len(Ycode.shape) <= 1 or 3 <= len(Ycode.shape):
            raise ValueError("unexpected length of shape of feature matrix\n "
                             "expected: 2, actual: " + str(len(Ycode.shape)))
        # check shape matching
        if w.shape[1] != X.shape[1]:
            msg = "unexpected shape combination\n" \
                  " expected: w.shape[1] == X.shape[1]\n" \
                  " actual: w.shape[1] = " + str(w.shape[1]) + ", X.shape[1] =" + str(X.shape[1])
            raise ValueError(msg)
        elif w.shape[0] != 1:
            msg = "unexpected shape of weight vectors \n" \
                  " expected: w.shape[0] == 1, actual: w.shape[0] = " + str(w.shape[0])
            raise ValueError(msg)
        elif X.shape[0] != Ycode.shape[0]:
            msg = "unexpected shape combination\n" \
                  " expected: X.shape[0] == Ycode.shape[0]\n" \
                  " actual: X.shape[0] = " + str(X.shape[0]) + ", Ycode.shape[0] = " + str(Ycode.shape[0])
            raise ValueError(msg)
        elif Ycode.shape[1] != 2:
            msg = "unexpected shape of Ycode\n" \
                  " expected: Ycode.shape[1] == 2," \
                  " actual: Ycode.shape[1] = " + str(Ycode.shape[1])
            raise ValueError(msg)

    except ValueError as e:
        print(e)
        print()
        raise

    w = w.reshape(w.shape[1], )
    u_all = X.dot(w)
    p_all = prob_logit(u_all)
    F_all = p_all.prod(axis=1)

    # active set
    threshold = 1e-6
    A = np.abs(w) > threshold

    # Hessian
    G = X[:, A].transpose().dot(np.diag(F_all)).dot(X[:, A])
    G += lambda2 * np.eye(G.shape[0])  # add l2 norm term

    # LOO factor
    C = np.einsum('ij,ji->i', X[:, A], np.linalg.solve(G, X[:, A].transpose()))

    # Gradientx
    b_all = Ycode[:, 0] - p_all[:, 0]

    # LOOE
    u_all_loo = u_all + C * b_all / (1.0 - F_all * C)
    p_all_loo = prob_logit(u_all_loo)

    LOOE = -1.0 * np.mean(np.log(np.sum(Ycode * p_all_loo, axis=1)))
    ERR = np.std(np.log(np.sum(Ycode * p_all_loo, axis=1))) / np.sqrt(X.shape[0] - 1)

    return LOOE, ERR
