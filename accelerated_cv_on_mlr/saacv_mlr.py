# coding=utf-8
import numpy as np
import math

from accelerated_cv_on_mlr.prob_multinomial import prob_multinomial


def saacv_mlr(wV, X, Ycode, Np=None, lambda2=0.0):
    """ A further simplified approximation of
    a leave-one-out estimator of predictive likelihood
    for multinomial accelerated_cv_on_mlr regression with elastic net regularization[1]

    Compute and return an very simplified approximation of
    a leave-one-out estimator (LOOE) and its standard error
    of predictive likelihood for multinomial accelerated_cv_on_mlr regression
    penalized by l1 norm.

    The following logistic regression penalized
    by the elastic net regularization (l1 + l2 norm) is considered:

                \hat{w}=argmin_{w}
                        { -\sum_{\mu}llkh(w|(y_{\mu},x_{\mu}))
                          + lambda*||w||_1 + (1/2)lambda2*||w||_2^2 },

    where llkh=log\phi is the log likelihood of logit model:

                \phi(w|(y,x))=(delta_{y,1}+delta_{y,2}e^{u})/(1+e^{u})

    where

                 u=x.w

    The leave-one-out estimator (LOOE) of a predictive likelihood is
    defined as the

                LOOE=-\sum_{\mu}llkh(\hat{w}^{\backslash \mu}|(y_{\mu},x_{\mu}))/M,

    where \hat{w}^{\backslash \mu} is the solution of the above
    minimization problem without the mu-th llkh term.
    This LOO solution \hat{w}^{\backslash \mu} is approximated
    from the full solution \hat{w}, yielding an approximate LOOE.


    Args:
        wV: weight vectors (p, N)-shape np.float64 array
        X: input feature matrix (M, N)-shape np.float64 array
        Ycode: class representative matrix (M, p)-shape np.int64 array
        Np: number of classes
        lambda2: Coefficient of the l2 regularization term

    Returns:
        LOOE, ERR (float, float)

    References:
        [1]: T. Obuchi and Y. Kabashima, arXiv:1711.05420

    Note:
        In this method, expected shape is different from the MATLAB implementation.
        (MATLAB -> w is a (N, p) matrix, Python -> w is (p, N)-shape np.float64 array)
        (This is due to sklearn package output form.)
    """
    try:
        # type check
        if type(wV) is not np.ndarray:
            msg = "unexpected type of weight vector\n" \
                  "expected: numpy.ndarray, actual: " + str(type(wV))
            raise ValueError(msg)
        elif type(X) is not np.ndarray:
            msg = "unexpected type of input feature matrix\n" \
                  "expected: numpy.ndarray, actual: " + str(type(X))
            raise ValueError(msg)
        elif type(Ycode) is not np.ndarray:
            msg = "unexpected type of claass representative matrix\n" \
                  "expected: numpy.ndarray, actual: " + str(type(Ycode))
            raise ValueError(msg)
        elif Np and (type(Np) is not int):
            msg = "unexpected type of claass representative matrix\n" \
                  "expected: int, actual: " + str(type(Np))
            raise ValueError(msg)

        # check length of shape
        if len(wV.shape) <= 1 or 3 <= len(wV.shape):
            raise ValueError("unexpected length of shape of weight vector\n expected: 2, actual: " + str(len(wV.shape)))
        elif len(X.shape) <= 1 or 3 <= len(X.shape):
            raise ValueError("unexpected length of shape of feature matrix\n expected: 2, actual: " + str(len(X.shape)))
        elif len(Ycode.shape) <= 1 or 3 <= len(Ycode.shape):
            raise ValueError("unexpected length of shape of feature matrix\n "
                             "expected: 2, actual: " + str(len(Ycode.shape)))

        if wV.shape[1] != X.shape[1]:
            msg = "unexpected shape combination\n" \
                  " expected: w.shape[1] == X.shape[1]\n" \
                  " actual: w.shape[1] = " + str(wV.shape[1]) + ", X.shape[1] =" + str(X.shape[1])
            raise ValueError(msg)
        elif X.shape[0] != Ycode.shape[0]:
            msg = "unexpected shape combination\n" \
                  " expected: X.shape[0] == Ycode.shape[0]\n" \
                  " actual: X.shape[0] = " + str(X.shape[0]) + ", Ycode.shape[0] = " + str(Ycode.shape[0])
            raise ValueError(msg)
        elif Ycode.shape[1] != wV.shape[0]:
            msg = "unexpected shape combination\n" \
                  " expected: Ycode.shape[1] == wV.shape[0]\n" \
                  " actual: Ycode.shape[1] = " + str(Ycode.shape[1]) + \
                  ", wV.shape[0] = " + str(wV.shape[0])
            raise ValueError(msg)

    except ValueError as e:
        print(e)
        print()
        raise

    if Np is None:
        Np = Ycode.shape[1]

    wV = wV.transpose()

    # Parameter
    M, N = X.shape
    Nparam = N * Np

    X_square = X * X
    mean_X_square = np.mean(X_square)

    # Preparation
    u_all = X.dot(wV)
    p_all = prob_multinomial(u_all)
    F = np.einsum('ab,ka->kab', np.eye(Np), p_all) - np.einsum('ka,kb->kab', p_all, p_all)

    # Active set
    A = (wV != 0)
    activated_positions = np.einsum('ka,kb->kab', A, A)

    # SA Approximation of LOO factor C
    # Initialization
    gamma = 0.5
    ERR = 100
    stack_I = np.einsum('k,ab->kab', np.ones(M), np.eye(Np))
    C_SA = np.zeros((Np, Np))
    chi = np.zeros((N, Np, Np))
    for i in range(N):
        chi[i][activated_positions[i]] = 1.0 / mean_X_square

    while ERR > 1e-6:
        chi_pre = chi.copy()

        # Compute R
        C_SA = mean_X_square * np.sum(chi, axis=0)

        R = np.sum(np.linalg.solve(stack_I + F.dot(C_SA), F), axis=0)

        # Update chi
        for index in range(N):
            sub_vector = R[activated_positions[index]]
            if len(sub_vector):
                length = int(math.sqrt(len(sub_vector)))
                [D, V] = np.linalg.eigh(sub_vector.reshape(length, length))
                A_rel = D > 1e-8

                Rinv_zmr = np.einsum('ij,j,mj->im', V[:, A_rel], 1.0 / D[A_rel], V[:, A_rel])

                chi[index][activated_positions[index]] = \
                    gamma * chi_pre[index][activated_positions[index]] + \
                    (1.0 - gamma) / mean_X_square * Rinv_zmr.reshape(length * length, )

        ERR = np.sum(np.linalg.norm(chi_pre - chi, ord='fro', axis=(1, 2))) / N

    # gradient
    b_all = np.zeros((Np, M))
    for ip in range(Np):
        b_all[ip, :] = p_all[:, ip].transpose() - Ycode[:, ip].transpose()

    # LOOE
    u_all_loo = np.zeros((M, Np))
    for mu in range(M):
        u_all_loo[mu, :] = u_all[mu, :] + (C_SA.dot(b_all[:, mu])).transpose()

    p_all_loo = prob_multinomial(u_all_loo)

    LOOE = -1.0 * np.mean(np.log(np.sum(Ycode * p_all_loo, axis=1)))
    ERR = np.std(np.log(np.sum(Ycode * p_all_loo, axis=1))) / np.sqrt(M - 1)

    return LOOE, ERR
