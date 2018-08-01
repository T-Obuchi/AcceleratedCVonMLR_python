# coding=utf-8
import numpy as np
import time

from accelerated_cv_on_mlr.prob_multinomial import prob_multinomial


def acv_mlr(wV, X, Ycode, Np=None, lambda2=0.0):
    """ An approximate leave-one-out estimator of predictive likelihood
    for multinomial accelerated_cv_on_mlr regression with elastic norm regularization[1]

    Compute and return an very simplified approximation of
    a leave-one-out estimator (LOOE) and its standard error
    of predivtive likelihood for multinomial accelerated_cv_on_mlr regression
    penalized by elastic net regularization.

    The following multinomial logistic regression penalized
    by the l1 + l2 norms (elastic net) is considered:

                \hat{w}=argmin_{{w_a}_a^{Np}}
                        { -\sum_{\mu}llkh({w_a}_a^{Np}|(y_{\mu},x_{\mu}))
                                 + lambda*\sum_{a}^{Np}||w_a||_1
                                 + (1/2)*lambda_2*\sum_{a}^{Np}||w_a||_2^2},

    where llkh=log\phi is the log likelihood of multinomial logistic map
    \phi:

                \phi(w|(y,x))=e^{u_{y}}/\sum_a e^{u_{a}}

    where

                 u_{a}=x.w_{a}.

    The leave-one-out estimator (LOOE) of a predictive likelihood is
    defined as the

                LOOE=-\sum_{\mu}llkh({\hat{w}^{\backslash \mu}_a}_a^{Np}|(y_{\mu},x_{\mu}))/M,

    where \hat{w}^{\backslash \mu}_a is the solution of the above
    minimization problem without the mu-th llkh term.
    This LOO solution \hat{w}^{\backslash \mu}_a is approximated
    from the full solution \hat{w}_a, yielding an approximate LOOE.

    Args:
        wV: weight vectors (p, N)-shape np.float64 array
        X: input feature matrix (M, N)-shape np.float64 array
        Ycode: class representative matrix (M, p)-shape np.int64 array
        Np: number of classes
        lambda2: coefficient of the l2 regularization term

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

    # Preparation
    u_all = X.dot(wV)
    p_all = prob_multinomial(u_all)
    W = wV.transpose().reshape(N * Np, )  # extended representation of weight vectors

    F_all = {}  # inter class hessian
    for i_p in range(Np):
        F_all[i_p] = {}
        for j_p in range(Np):
            F_all[i_p][j_p] = ((i_p == j_p) * p_all[:, i_p] - p_all[:, i_p] * p_all[:, j_p]).reshape(M, 1)

    # active set
    A = np.arange(len(W))[(W != 0)]  # position of active components
    A_ipt = np.mod(A, N)  # original column position
    A_cla = np.floor(A / N).astype(dtype=np.int64)  # original class index
    ORDER = np.arange(len(A))

    As_ord = {}
    As_ipt = {}
    for ip in range(Np):
        As_ord[ip] = ORDER[A_cla == ip]
        As_ipt[ip] = A_ipt[A_cla == ip]

    X_expand = __expand_X(A_cla, A_ipt, Np, X)

    F_expand = __calculate_F(A, A_cla, F_all, M)
    G = np.einsum('mk,ml,klm->kl', X_expand, X_expand, F_expand)

    # inverse hessian with zero mode removal
    [D, V] = np.linalg.eigh(G)
    threshold = 1e-8
    A_rel = D > threshold

    Ginv_zmr = V[:, A_rel].dot(np.linalg.inv(np.diag(D[A_rel]))).dot(V[:, A_rel].transpose())

    # LOO factor
    C = __calculate_C(As_ipt, As_ord, Ginv_zmr, M, Np, X)

    # gradient
    b_all = np.zeros((Np, M))
    for ip in range(Np):
        b_all[ip, :] = p_all[:, ip].transpose() - Ycode[:, ip].transpose()

    # LOOE
    F = np.zeros((Np, Np, M))
    I = np.eye(Np)
    for ip in range(Np):
        for jp in range(Np):
            F[ip, jp, :] = F_all[ip][jp].reshape(M, )

    u_all_loo = np.zeros((M, Np))
    for mu in range(M):
        temp = np.linalg.solve((I - F[:, :, mu].dot(C[:, :, mu])), b_all[:, mu])
        u_all_loo[mu, :] = u_all[mu, :] + (C[:, :, mu].dot(temp)).transpose()

    p_all_loo = prob_multinomial(u_all_loo)

    LOOE = -1.0 * np.mean(np.log(np.sum(Ycode * p_all_loo, axis=1)))
    ERR = np.std(np.log(np.sum(Ycode * p_all_loo, axis=1))) / np.sqrt(M - 1)

    return LOOE, ERR


def __expand_X(A_cla, A_ipt, Np, X):
    X_expand = np.concatenate([X[:, A_ipt[A_cla == p]] for p in range(Np)], axis=1)
    return X_expand


def __calculate_C(As_ipt, As_ord, Ginv_zmr, M, Np, X):
    """ calc C """
    C = np.zeros((Np, Np, M))

    for mu in range(M):
        for ip in range(Np):
            for jp in range(ip + 1, Np):
                C[ip, jp, mu] = X[mu, As_ipt[ip]].dot(
                    Ginv_zmr[As_ord[ip], :][:, As_ord[jp]]
                ).dot(
                    X[mu, As_ipt[jp]].transpose()
                )
        C[:, :, mu] = C[:, :, mu] + C[:, :, mu].transpose()
        for ip in range(Np):
            C[ip, ip, mu] = X[mu, As_ipt[ip]].dot(
                Ginv_zmr[As_ord[ip], :][:, As_ord[ip]]
            ).dot(
                X[mu, As_ipt[ip]].transpose()
            )

    return C


def __calculate_F(A, A_cla, F_all, M):
    """ calculate F """
    F_expand = np.array([F_all[A_cla[k]][A_cla[l]] for k in range(len(A)) for l in range(len(A))],
                        dtype=np.float32).reshape(len(A), len(A), M)
    return F_expand
