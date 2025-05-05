import unittest

import numpy as np

from bulk_recycling_model import coefficients
from bulk_recycling_model.cases import Wind


class Test_handle_inflow_boundaries(unittest.TestCase):
    def test_ok(self):
        Fx_left = np.array([[1, -1], [2, 2]])
        Fx_right = np.array([[2, 2], [1, -1]])
        Fy_bottom = np.array([[1, 2], [-1, 2]])
        Fy_top = np.array([[2, 1], [2, -1]])
        l, r, b, t = coefficients.Coefficients.handle_inflow_boundaries(Fx_left, Fx_right, Fy_bottom, Fy_top)  # noqa: E741
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
        self.coefficients = coefficients.Coefficients(Fx_left, Fx_right, Fy_bottom, Fy_top, E, P, dx, dy)

        # check the center of the grid (no boundary adjustments)
        assert self.coefficients.classification[1, 1] == Wind.SW
        assert self.coefficients.A_0[1, 1] == 2 * 5 + 1 + 1
        assert self.coefficients.alpha_1[1, 1] == 2 * 6
        assert self.coefficients.alpha_C[1, 1] == 1 + 1
        assert self.coefficients.alpha_U[1, 1] == -1
        assert self.coefficients.alpha_R[1, 1] == -1
        assert self.coefficients.alpha_D[1, 1] == 1
        assert self.coefficients.alpha_L[1, 1] == 1

    def test_NW(self):
        Fx_left = Fx_right = np.full((3, 3), 1)
        Fy_bottom = Fy_top = np.full((3, 3), -1)
        E = np.full((3, 3), 6)
        P = np.full((3, 3), 5)
        dx = dy = 1
        self.coefficients = coefficients.Coefficients(Fx_left, Fx_right, Fy_bottom, Fy_top, E, P, dx, dy)

        # check the center of the grid (no boundary adjustments)
        assert self.coefficients.classification[1, 1] == Wind.NW
        assert self.coefficients.A_0[1, 1] == 2 * 5 + 1 + 1
        assert self.coefficients.alpha_1[1, 1] == 2 * 6
        assert self.coefficients.alpha_C[1, 1] == 1 + 1
        assert self.coefficients.alpha_U[1, 1] == 1
        assert self.coefficients.alpha_R[1, 1] == -1
        assert self.coefficients.alpha_D[1, 1] == -1
        assert self.coefficients.alpha_L[1, 1] == 1

    def test_NE(self):
        Fx_left = Fx_right = np.full((3, 3), -1)
        Fy_bottom = Fy_top = np.full((3, 3), -1)
        E = np.full((3, 3), 6)
        P = np.full((3, 3), 5)
        dx = dy = 1
        self.coefficients = coefficients.Coefficients(Fx_left, Fx_right, Fy_bottom, Fy_top, E, P, dx, dy)

        # check the center of the grid (no boundary adjustments)
        assert self.coefficients.classification[1, 1] == Wind.NE
        assert self.coefficients.A_0[1, 1] == 2 * 5 + 1 + 1
        assert self.coefficients.alpha_1[1, 1] == 2 * 6
        assert self.coefficients.alpha_C[1, 1] == 1 + 1
        assert self.coefficients.alpha_U[1, 1] == 1
        assert self.coefficients.alpha_R[1, 1] == 1
        assert self.coefficients.alpha_D[1, 1] == -1
        assert self.coefficients.alpha_L[1, 1] == -1

    def test_SE(self):
        Fx_left = Fx_right = np.full((3, 3), -1)
        Fy_bottom = Fy_top = np.full((3, 3), 1)
        E = np.full((3, 3), 6)
        P = np.full((3, 3), 5)
        dx = dy = 1
        self.coefficients = coefficients.Coefficients(Fx_left, Fx_right, Fy_bottom, Fy_top, E, P, dx, dy)

        # check the center of the grid (no boundary adjustments)
        assert self.coefficients.classification[1, 1] == Wind.SE
        assert self.coefficients.A_0[1, 1] == 2 * 5 + 1 + 1
        assert self.coefficients.alpha_1[1, 1] == 2 * 6
        assert self.coefficients.alpha_C[1, 1] == 1 + 1
        assert self.coefficients.alpha_U[1, 1] == -1
        assert self.coefficients.alpha_R[1, 1] == 1
        assert self.coefficients.alpha_D[1, 1] == 1
        assert self.coefficients.alpha_L[1, 1] == -1
