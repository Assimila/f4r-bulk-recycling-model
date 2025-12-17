import unittest

import numpy as np

from bulk_recycling_model.cases import Wind
from bulk_recycling_model.coefficients import Coefficients
from bulk_recycling_model.preprocess import calculate_precipitation


class Test_handle_inflow_boundaries(unittest.TestCase):
    def test_ok(self):
        Fx_left = np.array([[1, -1], [2, 2]])
        Fx_right = np.array([[2, 2], [1, -1]])
        Fy_bottom = np.array([[1, 2], [-1, 2]])
        Fy_top = np.array([[2, 1], [2, -1]])
        l, r, b, t = Coefficients.handle_inflow_boundaries(Fx_left, Fx_right, Fy_bottom, Fy_top)  # noqa: E741
        # test for zeros
        assert np.all(l == np.array([[0, -1], [2, 2]]))
        assert np.all(r == np.array([[2, 2], [1, 0]]))
        assert np.all(b == np.array([[0, 2], [-1, 2]]))
        assert np.all(t == np.array([[2, 1], [2, 0]]))


class Test_Coefficients(unittest.TestCase):
    def test_SW(self):
        Fx_left = Fx_right = np.full((3, 3), 1)
        Fy_bottom = Fy_top = np.full((3, 3), 1)
        E = np.full((3, 3), 6)
        P = np.full((3, 3), 5)
        dx = dy = 1
        coefficients = Coefficients(Fx_left, Fx_right, Fy_bottom, Fy_top, E, P, dx, dy)

        # check the center of the grid (no boundary adjustments)
        assert coefficients.classification[1, 1] == Wind.SW
        assert coefficients.A_0[1, 1] == 2 * 5 + 1 + 1
        assert coefficients.alpha_1[1, 1] == 2 * 6
        assert coefficients.alpha_C[1, 1] == 1 + 1
        assert coefficients.alpha_U[1, 1] == -1
        assert coefficients.alpha_R[1, 1] == -1
        assert coefficients.alpha_D[1, 1] == 1
        assert coefficients.alpha_L[1, 1] == 1

    def test_NW(self):
        Fx_left = Fx_right = np.full((3, 3), 1)
        Fy_bottom = Fy_top = np.full((3, 3), -1)
        E = np.full((3, 3), 6)
        P = np.full((3, 3), 5)
        dx = dy = 1
        coefficients = Coefficients(Fx_left, Fx_right, Fy_bottom, Fy_top, E, P, dx, dy)

        # check the center of the grid (no boundary adjustments)
        assert coefficients.classification[1, 1] == Wind.NW
        assert coefficients.A_0[1, 1] == 2 * 5 + 1 + 1
        assert coefficients.alpha_1[1, 1] == 2 * 6
        assert coefficients.alpha_C[1, 1] == 1 + 1
        assert coefficients.alpha_U[1, 1] == 1
        assert coefficients.alpha_R[1, 1] == -1
        assert coefficients.alpha_D[1, 1] == -1
        assert coefficients.alpha_L[1, 1] == 1

    def test_NE(self):
        Fx_left = Fx_right = np.full((3, 3), -1)
        Fy_bottom = Fy_top = np.full((3, 3), -1)
        E = np.full((3, 3), 6)
        P = np.full((3, 3), 5)
        dx = dy = 1
        coefficients = Coefficients(Fx_left, Fx_right, Fy_bottom, Fy_top, E, P, dx, dy)

        # check the center of the grid (no boundary adjustments)
        assert coefficients.classification[1, 1] == Wind.NE
        assert coefficients.A_0[1, 1] == 2 * 5 + 1 + 1
        assert coefficients.alpha_1[1, 1] == 2 * 6
        assert coefficients.alpha_C[1, 1] == 1 + 1
        assert coefficients.alpha_U[1, 1] == 1
        assert coefficients.alpha_R[1, 1] == 1
        assert coefficients.alpha_D[1, 1] == -1
        assert coefficients.alpha_L[1, 1] == -1

    def test_SE(self):
        Fx_left = Fx_right = np.full((3, 3), -1)
        Fy_bottom = Fy_top = np.full((3, 3), 1)
        E = np.full((3, 3), 6)
        P = np.full((3, 3), 5)
        dx = dy = 1
        coefficients = Coefficients(Fx_left, Fx_right, Fy_bottom, Fy_top, E, P, dx, dy)

        # check the center of the grid (no boundary adjustments)
        assert coefficients.classification[1, 1] == Wind.SE
        assert coefficients.A_0[1, 1] == 2 * 5 + 1 + 1
        assert coefficients.alpha_1[1, 1] == 2 * 6
        assert coefficients.alpha_C[1, 1] == 1 + 1
        assert coefficients.alpha_U[1, 1] == -1
        assert coefficients.alpha_R[1, 1] == 1
        assert coefficients.alpha_D[1, 1] == 1
        assert coefficients.alpha_L[1, 1] == -1


class Test_instability_heuristic(unittest.TestCase):
    def test_regression(self):
        """
        regression test, using a 3x3 slice of ERA data,
        which is know to have numerical instability.
        """
        Fx_left = np.array(
            [
                [2.56019604, -0.71216823, -3.01511097],
                [2.04025812, -0.91973988, -2.89132212],
                [2.58103255, -0.26332458, -2.13375434],
            ]
        )
        Fx_right = np.array(
            [
                [2.04025812, -0.91973988, -2.89132212],
                [2.58103255, -0.26332458, -2.13375434],
                [2.39412323, -0.62477035, -2.56121831],
            ]
        )
        Fy_bottom = np.array(
            [
                [2.02432782, 1.22458342, 1.08957919],
                [1.04683338, 0.52464728, 0.8318316],
                [0.31467802, 0.36302492, 0.63753953],
            ]
        )
        Fy_top = np.array(
            [
                [1.22458342, 1.08957919, 0.56159528],
                [0.52464728, 0.8318316, 0.25773877],
                [0.36302492, 0.63753953, 0.58617741],
            ]
        )
        E = np.array(
            [
                [2.94533699, 2.8531073, 2.87521196],
                [2.93569207, 2.90742972, 2.91058806],
                [2.8570064, 2.87345821, 2.91699079],
            ]
        )
        dx = 0.03490644223630295
        dy = 0.034482758620689655

        # preprocess
        P = calculate_precipitation(
            Fx_left, Fx_right, Fy_bottom, Fy_top, E, dx, dy
        )

        coefficients = Coefficients(Fx_left, Fx_right, Fy_bottom, Fy_top, E, P, dx, dy)

        instability_heuristic = coefficients.instability_heuristic

        # central cell is >> 1
        self.assertGreater(instability_heuristic[1, 1], 30)
