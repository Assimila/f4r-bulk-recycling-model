import unittest

import numpy as np

from bulk_recycling_model.cases import Wind, classify_cells


class TestClassifyCells(unittest.TestCase):
    def test_classify_cells_all_SW(self):
        Fx_left = np.array([[1, 1], [1, 1]])
        Fy_bottom = np.array([[1, 1], [1, 1]])
        expected = np.array([[Wind.SW, Wind.SW], [Wind.SW, Wind.SW]], dtype=np.int8)
        result = classify_cells(Fx_left, Fy_bottom)
        np.testing.assert_array_equal(result, expected)

    def test_classify_cells_all_NW(self):
        Fx_left = np.array([[1, 1], [1, 1]])
        Fy_bottom = np.array([[-1, -1], [-1, -1]])
        expected = np.array([[Wind.NW, Wind.NW], [Wind.NW, Wind.NW]], dtype=np.int8)
        result = classify_cells(Fx_left, Fy_bottom)
        np.testing.assert_array_equal(result, expected)

    def test_classify_cells_all_NE(self):
        Fx_left = np.array([[-1, -1], [-1, -1]])
        Fy_bottom = np.array([[-1, -1], [-1, -1]])
        expected = np.array([[Wind.NE, Wind.NE], [Wind.NE, Wind.NE]], dtype=np.int8)
        result = classify_cells(Fx_left, Fy_bottom)
        np.testing.assert_array_equal(result, expected)

    def test_classify_cells_all_SE(self):
        Fx_left = np.array([[-1, -1], [-1, -1]])
        Fy_bottom = np.array([[1, 1], [1, 1]])
        expected = np.array([[Wind.SE, Wind.SE], [Wind.SE, Wind.SE]], dtype=np.int8)
        result = classify_cells(Fx_left, Fy_bottom)
        np.testing.assert_array_equal(result, expected)

    def test_classify_cells_mixed(self):
        Fx_left = np.array([[1, 1], [-1, -1]])
        Fy_bottom = np.array([[1, -1], [-1, 1]])
        expected = np.array([[Wind.SW, Wind.NW], [Wind.NE, Wind.SE]], dtype=np.int8)
        result = classify_cells(Fx_left, Fy_bottom)
        np.testing.assert_array_equal(result, expected)
