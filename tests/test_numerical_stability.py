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


class Test_nudge_hot_pixel(unittest.TestCase):
    def test_nudge(self):
        E = np.array(
            [
                [2, 2, 2],
                [2, 0, 2],
                [2, 2, 2],
            ]
        )
        i_hot, j_hot = 1, 1
        E_nudged = numerical_stability.nudge_hot_pixel(E, i_hot, j_hot, 1.0)
        # kernel should look like:
        # [[0.707, 1.0, 0.707],
        #  [1.0,   0,   1.0  ],
        #  [0.707, 1.0, 0.707]]
        d11 = 1 / np.sqrt(2)
        w = d11 * 4 + 1.0 * 4
        expected_E_nudged = np.array(
            [
                [2 - d11 / w, 2 - 1.0 / w, 2 - d11 / w],
                [2 - 1.0 / w, 1.0, 2 - 1.0 / w],
                [2 - d11 / w, 2 - 1.0 / w, 2 - d11 / w],
            ]
        )
        np.testing.assert_array_almost_equal(E_nudged, expected_E_nudged)

    def test_negative_nudge(self):
        E = np.array(
            [
                [2, 2, 2],
                [2, 4, 2],
                [2, 2, 2],
            ]
        )
        i_hot, j_hot = 1, 1
        E_nudged = numerical_stability.nudge_hot_pixel(E, i_hot, j_hot, -1.0)
        # kernel should look like:
        # [[0.707, 1.0, 0.707],
        #  [1.0,   0,   1.0  ],
        #  [0.707, 1.0, 0.707]]
        d11 = 1 / np.sqrt(2)
        w = d11 * 4 + 1.0 * 4
        expected_E_nudged = np.array(
            [
                [2 + d11 / w, 2 + 1.0 / w, 2 + d11 / w],
                [2 + 1.0 / w, 3.0, 2 + 1.0 / w],
                [2 + d11 / w, 2 + 1.0 / w, 2 + d11 / w],
            ]
        )
        np.testing.assert_array_almost_equal(E_nudged, expected_E_nudged)

    def test_kernel_size(self):
        """
        test on 5x5 kernel
        """
        E = np.array(
            [
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [2, 2, 0, 2, 2],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
            ]
        )
        i_hot, j_hot = 2, 2
        E_nudged = numerical_stability.nudge_hot_pixel(E, i_hot, j_hot, 1.0, kernel_size=5)
        d10 = 1 / 1
        d11 = 1 / np.sqrt(2)
        d20 = 1 / 2
        d12 = 1 / np.sqrt(5)
        d22 = 1 / np.sqrt(8)
        w = d10 * 4 + d11 * 4 + d20 * 4 + d12 * 8 + d22 * 4
        expected_E_nudged = np.array(
            [
                [2 - d22 / w, 2 - d12 / w, 2 - d20 / w, 2 - d12 / w, 2 - d22 / w],
                [2 - d12 / w, 2 - d11 / w, 2 - d10 / w, 2 - d11 / w, 2 - d12 / w],
                [2 - d20 / w, 2 - d10 / w, 1.0, 2 - d10 / w, 2 - d20 / w],
                [2 - d12 / w, 2 - d11 / w, 2 - d10 / w, 2 - d11 / w, 2 - d12 / w],
                [2 - d22 / w, 2 - d12 / w, 2 - d20 / w, 2 - d12 / w, 2 - d22 / w],
            ]
        )
        np.testing.assert_array_almost_equal(E_nudged, expected_E_nudged)

    def test_corner(self):
        E = np.array(
            [
                [0.0, 2.0],
                [2.0, 2.0],
            ]
        )
        i_hot, j_hot = 0, 0
        E_nudged = numerical_stability.nudge_hot_pixel(E, i_hot, j_hot, 1.0)
        d11 = 1 / np.sqrt(2)
        w = d11 * 1 + 1.0 * 2
        expected_E_nudged = np.array(
            [
                [1.0, 2.0 - 1.0 / w],
                [2.0 - 1.0 / w, 2.0 - d11 / w],
            ]
        )
        np.testing.assert_array_almost_equal(E_nudged, expected_E_nudged)

    def test_edge(self):
        E = np.array(
            [
                [2.0, 0.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        )
        i_hot, j_hot = 0, 1
        E_nudged = numerical_stability.nudge_hot_pixel(E, i_hot, j_hot, 1.0)
        d11 = 1 / np.sqrt(2)
        w = 1 * 3 + d11 * 2
        expected_E_nudged = np.array(
            [
                [2.0 - 1.0 / w, 1.0, 2.0 - 1.0 / w],
                [2.0 - d11 / w, 2.0 - 1.0 / w, 2.0 - d11 / w],
            ]
        )
        np.testing.assert_array_almost_equal(E_nudged, expected_E_nudged)
