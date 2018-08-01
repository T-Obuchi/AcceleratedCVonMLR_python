# coding=utf-8
import unittest
import numpy as np
from numpy.testing import assert_allclose

import csv
from accelerated_cv_on_mlr.acv_logit import acv_logit


class TestAcvLogit(unittest.TestCase):
    def test_calculation(self):
        """ compare calculation result with that of matlab version """

        # read test data
        X = []
        X_reader = csv.reader(open("./sample_data_logit/dummy_x.csv", "r"))
        for data in X_reader:
            X.append(data)
        X = np.array(X, dtype=np.float64)

        Ycode = []
        Ycode_reader = csv.reader(open("./sample_data_logit/dummy_Ycode.csv", "r"))
        for data in Ycode_reader:
            Ycode.append(data)
        Ycode = np.array(Ycode, dtype=np.int64)

        wV = []
        wV_reader = csv.reader(open("./sample_data_logit/dummy_wV.csv", "r"))
        for data in wV_reader:
            wV.append(data)
        wV = np.array(wV, dtype=np.float64)

        expected = np.array([0.226749338949696, 0.022556893001450])
        actual = acv_logit(wV, X, Ycode)

        assert_allclose(actual, expected, rtol=1e-3)

    def test_type_checker(self):
        """ test type checker """
        # make ideal dummy data
        N = 10
        M = 20
        w = np.random.rand(1, N)
        X = np.random.rand(M, N)
        Ycode = np.random.binomial(1, 0.5, (N, 1))
        Ycode = np.concatenate((Ycode, np.mod(Ycode - 1, 2)), axis=1)

        self.assertRaises(ValueError, acv_logit, 1.0, X, Ycode)
        self.assertRaises(ValueError, acv_logit, w, 1.0, Ycode)
        self.assertRaises(ValueError, acv_logit, w, X, 1.0)

    def test_shape_length_check(self):
        """ test shape length checker"""

        # make ideal dummy data
        N = 10
        M = 20
        w = np.random.rand(1, N)
        X = np.random.rand(M, N)
        Ycode = np.random.binomial(1, 0.5, (N, 1))
        Ycode = np.concatenate((Ycode, np.mod(Ycode - 1, 2)), axis=1)

        # self.assertRaises(ValueError, acv_logit, w, X, Ycode)  # ideal case

        # shape length w
        self.assertRaises(ValueError, acv_logit, np.random.rand(1, 2, 3), X, Ycode)
        self.assertRaises(ValueError, acv_logit, np.random.rand(3), X, Ycode)

        # shape length X
        self.assertRaises(ValueError, acv_logit, w, np.random.rand(1, 2, 3), Ycode)
        self.assertRaises(ValueError, acv_logit, w, np.random.rand(3), Ycode)

        # shape length Y
        self.assertRaises(ValueError, acv_logit, w, X, np.random.rand(1, 2, 3))
        self.assertRaises(ValueError, acv_logit, w, X, np.random.rand(3))

    def test_shape_matching_check(self):
        """ test shape matching checker"""

        # make ideal dummy data
        N = 10
        M = 20
        w = np.random.rand(1, N)
        X = np.random.rand(M, N)
        Ycode = np.random.binomial(1, 0.5, (M, 1))
        Ycode = np.concatenate((Ycode, np.mod(Ycode - 1, 2)), axis=1)

        # self.assertRaises(ValueError, acv_logit, w, X, Ycode)  # ideal case

        # shape unmatching w and X
        self.assertRaises(ValueError, acv_logit, np.random.rand(1, N - 1), X, Ycode)
        self.assertRaises(ValueError, acv_logit, w, np.random.rand(M, N - 1), Ycode)

        # shape unmaching w
        self.assertRaises(ValueError, acv_logit, np.random.rand(2, N), X, Ycode)

        # shape unmatching X and Ycode
        self.assertRaises(ValueError, acv_logit, w, np.random.rand(M - 1, N), Ycode)
        self.assertRaises(ValueError, acv_logit, w, X, np.random.rand(M - 1, 2))

        # shape unmatching Ycode
        self.assertRaises(ValueError, acv_logit, w, X, np.random.rand(M, 3))
        self.assertRaises(ValueError, acv_logit, w, X, np.random.rand(M, 1))


if __name__ == '__main__':
    unittest.main()
