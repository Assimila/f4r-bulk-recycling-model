import unittest

import numpy as np

from bulk_recycling_model import numerical_stability


class Test_identify_hot_pixel(unittest.TestCase):
    def test_identify_hot_pixel(self):
        instability_heuristic = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.9, 0.6],
                [0.7, 0.8, 0.5],
            ]
        )
        i_hot, j_hot = numerical_stability.identify_hot_pixel(instability_heuristic)
        self.assertEqual((i_hot, j_hot), (1, 1))


class Test_smooth_hot_pixel(unittest.TestCase):
    def test_ok(self):
        # total E = 18
        E = np.array(
            [
                [18/8, 18/8, 18/8],
                [18/8, 0.0, 18/8],
                [18/8, 18/8, 18/8],
            ]
        )
        i_hot, j_hot = 1, 1
        E_smoothed = numerical_stability.smooth_hot_pixel(E, i_hot, j_hot, weight=0.5)
        expected_E_smoothed = np.array(
            [
                [18/8 - 0.05, 18/8 - 0.2, 18/8 - 0.05],
                [18/8 - 0.2, 1.0, 18/8 - 0.2],
                [18/8 - 0.05, 18/8 - 0.2, 18/8 - 0.05],
            ]
        )
        np.testing.assert_array_almost_equal(E_smoothed, expected_E_smoothed)

    def test_non_zero(self):
        # same test as above but shifted up by 1.0
        E = np.array(
            [
                [18/8, 18/8, 18/8],
                [18/8, 0.0, 18/8],
                [18/8, 18/8, 18/8],
            ]
        )
        E += 1.0  # shift all values up by 1
        i_hot, j_hot = 1, 1
        E_smoothed = numerical_stability.smooth_hot_pixel(E, i_hot, j_hot, weight=0.5)
        expected_E_smoothed = np.array(
            [
                [18/8 - 0.05, 18/8 - 0.2, 18/8 - 0.05],
                [18/8 - 0.2, 1.0, 18/8 - 0.2],
                [18/8 - 0.05, 18/8 - 0.2, 18/8 - 0.05],
            ]
        )
        expected_E_smoothed += 1.0  # shift all values up by 1
        np.testing.assert_array_almost_equal(E_smoothed, expected_E_smoothed)

    def test_weight_0(self):
        # total E = 18
        E = np.array(
            [
                [18/8, 18/8, 18/8],
                [18/8, 0.0, 18/8],
                [18/8, 18/8, 18/8],
            ]
        )
        i_hot, j_hot = 1, 1
        E_smoothed = numerical_stability.smooth_hot_pixel(E, i_hot, j_hot, weight=0)
        np.testing.assert_array_almost_equal(E_smoothed, E)

    def test_weight_1(self):
        # total E = 18
        E = np.array(
            [
                [18/8, 18/8, 18/8],
                [18/8, 0.0, 18/8],
                [18/8, 18/8, 18/8],
            ]
        )
        i_hot, j_hot = 1, 1
        E_smoothed = numerical_stability.smooth_hot_pixel(E, i_hot, j_hot, weight=1)
        expected_E_smoothed = np.array(
            [
                [18/8 - 0.1, 18/8 - 0.4, 18/8 - 0.1],
                [18/8 - 0.4, 2.0, 18/8 - 0.4],
                [18/8 - 0.1, 18/8 - 0.4, 18/8 - 0.1],
            ]
        )
        np.testing.assert_array_almost_equal(E_smoothed, expected_E_smoothed)

    def test_corner(self):
        # total E = 8
        E = np.array(
            [
                [0.0, 8/3],
                [8/3, 8/3],
            ]
        )
        i_hot, j_hot = 0, 0
        E_smoothed = numerical_stability.smooth_hot_pixel(E, i_hot, j_hot, weight=0.5)
        expected_E_smoothed = np.array(
            [
                [1.0, 8/3 - 4/9],
                [8/3 - 4/9, 8/3 - 1/9]
            ]
        )
        np.testing.assert_array_almost_equal(E_smoothed, expected_E_smoothed)

    def test_edge(self):
        # total E = 12
        E = np.array(
            [
                [12/5, 0.0, 12/5],
                [12/5, 12/5, 12/5],
            ]
        )
        i_hot, j_hot = 0, 1
        E_smoothed = numerical_stability.smooth_hot_pixel(E, i_hot, j_hot, weight=0.5)
        expected_E_smoothed = np.array(
            [
                [12/5 - 4/14, 1.0, 12/5 - 4/14],
                [12/5 - 1/14, 12/5 - 4/14, 12/5 - 1/14],
            ]
        )
        np.testing.assert_array_almost_equal(E_smoothed, expected_E_smoothed)

    def test_gaussian_sigma(self):
        # total E = 18
        E = np.array(
            [
                [18/8, 18/8, 18/8],
                [18/8, 0.0, 18/8],
                [18/8, 18/8, 18/8],
            ]
        )
        i_hot, j_hot = 1, 1
        E_smoothed = numerical_stability.smooth_hot_pixel(E, i_hot, j_hot, weight=0.0, gaussian_scale=0.5)
        # assert conservation
        self.assertAlmostEqual(np.sum(E_smoothed), np.sum(E))
        # assert that orthogonal pixels are equal
        v = E_smoothed[0, 1]
        self.assertAlmostEqual(E_smoothed[1, 0], v)
        self.assertAlmostEqual(E_smoothed[1, 2], v)
        self.assertAlmostEqual(E_smoothed[2, 1], v)
        # assert that diagonal pixels are equal
        v = E_smoothed[0, 0]
        self.assertAlmostEqual(E_smoothed[0, 2], v)
        self.assertAlmostEqual(E_smoothed[2, 0], v)
        self.assertAlmostEqual(E_smoothed[2, 2], v)
