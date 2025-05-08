import unittest

import numpy as np

from bulk_recycling_model import plotting


class Test_build_uv_fluxes(unittest.TestCase):
    def test_1x1(self):
        Fx_left = np.array([[1]])
        Fx_right = np.array([[2]])
        Fy_bottom = np.array([[3]])
        Fy_top = np.array([[4]])

        U, V = plotting.build_uv_fluxes(Fx_left, Fx_right, Fy_bottom, Fy_top)

        self.assertEqual(U.shape, (3, 3))
        self.assertEqual(V.shape, (3, 3))

        expected_U = np.array([[np.nan, 1, np.nan], [0, np.nan, 0], [np.nan, 2, np.nan]])
        expected_V = np.array([[np.nan, 0, np.nan], [3, np.nan, 4], [np.nan, 0, np.nan]])

        np.testing.assert_equal(U, expected_U)
        np.testing.assert_equal(V, expected_V)

    def test_2x2(self):
        Fx_left = np.array([[1, 2], [3, 4]])
        Fx_right = np.array([[3, 4], [5, 6]])
        Fy_bottom = np.array([[11, 12], [13, 14]])
        Fy_top = np.array([[12, 15], [14, 16]])

        U, V = plotting.build_uv_fluxes(Fx_left, Fx_right, Fy_bottom, Fy_top)

        self.assertEqual(U.shape, (5, 5))
        self.assertEqual(V.shape, (5, 5))

        expected_U = np.array(
            [
                [np.nan, 1, np.nan, 2, np.nan],
                [0, np.nan, 0, np.nan, 0],
                [np.nan, 3, np.nan, 4, np.nan],
                [0, np.nan, 0, np.nan, 0],
                [np.nan, 5, np.nan, 6, np.nan],
            ]
        )
        expected_V = np.array(
            [
                [np.nan, 0, np.nan, 0, np.nan],
                [11, np.nan, 12, np.nan, 15],
                [np.nan, 0, np.nan, 0, np.nan],
                [13, np.nan, 14, np.nan, 16],
                [np.nan, 0, np.nan, 0, np.nan],
            ]
        )

        np.testing.assert_equal(U, expected_U)
        np.testing.assert_equal(V, expected_V)
