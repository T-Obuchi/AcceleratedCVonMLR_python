# coding=utf-8
import unittest
import csv
import numpy as np
from numpy.testing import assert_allclose

from accelerated_cv_on_mlr.utils.standardize_matrix import standardize_matrix


class TestStandardizeMatrix(unittest.TestCase):
    def test_calculation(self):
        a = np.arange(9).reshape(3, 3)
        actual = standardize_matrix(a)
        desired = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        assert_allclose(desired, actual)

        a = np.arange(12).reshape(3, 4)
        actual = standardize_matrix(a)
        desired = np.array([[-1, -1, -1, -1], [0, 0, 0, 0], [1, 1, 1, 1]])
        assert_allclose(desired, actual)

    def test_type_checker(self):
        a = 1.0
        self.assertRaises(ValueError, standardize_matrix, a)

    def test_shape_checker(self):
        a = np.arange(10)
        self.assertRaises(ValueError, standardize_matrix, a)

        b = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        self.assertRaises(ValueError, standardize_matrix, b)


if __name__ == '__main__':
    unittest.main()
