import unittest

import numpy as np

from bulk_recycling_model import coefficients
from bulk_recycling_model.cases import Wind


class Test_Coefficients(unittest.TestCase):
    def setUp(self):
        # there values are completely un-physical
        Fx_left = np.array([[1, 1], [1, 1]])
        Fx_right = np.array([[2, 2], [2, 2]])
        Fy_bottom = np.array([[3, 3], [3, 3]])
        Fy_top = np.array([[4, 4], [4, 4]])
        E = np.array([[6, 6], [6, 6]])
        P = np.array([[5, 5], [5, 5]])
        dx = dy = 1
        # force the classification
        classification = np.array([[Wind.SW, Wind.NW], [Wind.NE, Wind.SE]], dtype=np.int8)
        self.coefficients = coefficients.Coefficients(
            Fx_left, Fx_right, Fy_bottom, Fy_top, E, P, dx, dy, classification
        )

    def test_A_0(self):
        expected = np.array(
            [
                [2 * 5 + 2 + 4, 2 * 5 + 2 - 3],
                [2 * 5 - 1 - 3, 2 * 5 - 1 + 4],
            ]
        )
        np.testing.assert_array_equal(self.coefficients.A_0, expected)

    def test_alpha_1(self):
        expected = np.array([[12, 12], [12, 12]])
        np.testing.assert_array_equal(self.coefficients.alpha_1, expected)

    def test_alpha_C(self):
        expected = np.array([[1 + 3, 1 - 4], [-2 - 4, -2 + 3]])
        np.testing.assert_array_equal(self.coefficients.alpha_C, expected)

    def test_alpha_U(self):
        expected = np.array([[-4, -4], [-4, -4]])
        np.testing.assert_array_equal(self.coefficients.alpha_U, expected)

    def test_alpha_R(self):
        expected = np.array([[-2, -2], [-2, -2]])
        np.testing.assert_array_equal(self.coefficients.alpha_R, expected)

    def test_alpha_D(self):
        expected = np.array([[3, 3], [3, 3]])
        np.testing.assert_array_equal(self.coefficients.alpha_D, expected)

    def test_alpha_L(self):
        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(self.coefficients.alpha_L, expected)
