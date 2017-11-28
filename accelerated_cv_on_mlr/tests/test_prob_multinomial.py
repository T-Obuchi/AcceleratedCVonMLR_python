# coding=utf-8
import unittest

import numpy as np
from numpy.testing import assert_allclose

from accelerated_cv_on_mlr.prob_multinomial import prob_multinomial


class MyProbMultinomial(unittest.TestCase):
    def test_calculation(self):
        """ calculation value check """
        x = np.log(np.array([[1, 2, 3], [4, 5, 6]]))
        expected = np.array([[1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0], [4.0 / 15.0, 5.0 / 15.0, 6.0 / 15.0]])
        actual = prob_multinomial(x)
        assert_allclose(actual, expected, rtol=1e-10)

    def test_type_check(self):
        """ type check test"""
        x = np.array([1.0])
        self.assertRaises(ValueError, prob_multinomial, x)

        x = np.array(1.0)
        self.assertRaises(ValueError, prob_multinomial, x)


if __name__ == '__main__':
    unittest.main()
