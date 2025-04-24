import unittest

import numpy as np

from bulk_recycling_model import preprocess


class Test_check_input_array(unittest.TestCase):
    def test_ok(self):
        array = np.ones((3, 3), dtype=float)
        preprocess._check_input_array(array)

    def test_not_2d(self):
        array = np.ones((3,), dtype=float)
        with self.assertRaises(ValueError):
            preprocess._check_input_array(array)

    def test_nans(self):
        array = np.ones((3, 3), dtype=float)
        array[0, 0] = np.nan
        with self.assertRaises(ValueError):
            preprocess._check_input_array(array)

    def test_too_small(self):
        array = np.ones((2, 3), dtype=float)
        with self.assertRaises(ValueError):
            preprocess._check_input_array(array)


# np.arange(9).reshape(3, 3) = [[0,1,2],[3,4,5],[6,7,8]]
# with indexing [lon, lat], in xy looks like:
# 2 5 8
# 1 4 7
# 0 3 6


class Test_prepare_E(unittest.TestCase):
    def test_ok(self):
        E = np.arange(9).reshape(3, 3)
        expected = np.array([[2, 3], [5, 6]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess.prepare_E(E), expected)


class Test_prepare_Fx_left(unittest.TestCase):
    def test_ok(self):
        Fx = np.arange(9).reshape(3, 3)
        expected = np.array([[0.5, 1.5], [3.5, 4.5]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess.prepare_Fx_left(Fx), expected)


class Test_prepare_Fx_right(unittest.TestCase):
    def test_ok(self):
        Fx = np.arange(9).reshape(3, 3)
        expected = np.array([[3.5, 4.5], [6.5, 7.5]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess.prepare_Fx_right(Fx), expected)


class Test_prepare_Fy_bottom(unittest.TestCase):
    def test_ok(self):
        Fy = np.arange(9).reshape(3, 3)
        expected = np.array([[1.5, 2.5], [4.5, 5.5]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess.prepare_Fy_bottom(Fy), expected)


class Test_prepare_Fy_top(unittest.TestCase):
    def test_ok(self):
        Fy = np.arange(9).reshape(3, 3)
        expected = np.array([[2.5, 3.5], [5.5, 6.5]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess.prepare_Fy_top(Fy), expected)
