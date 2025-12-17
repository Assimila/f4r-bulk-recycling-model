import unittest

import numpy as np

from bulk_recycling_model import preprocess_low_resolution
from tests.data.load_data import load_data


class Test_check_input_array(unittest.TestCase):
    def test_ok(self):
        array = np.ones((5, 5), dtype=float)
        preprocess_low_resolution._check_input_array(array)

    def test_not_2d(self):
        array = np.ones((5,), dtype=float)
        with self.assertRaises(ValueError):
            preprocess_low_resolution._check_input_array(array)

    def test_nans(self):
        array = np.ones((5, 5), dtype=float)
        array[0, 0] = np.nan
        with self.assertRaises(ValueError):
            preprocess_low_resolution._check_input_array(array)

    def test_too_small(self):
        array = np.ones((4, 4), dtype=float)
        with self.assertRaises(ValueError):
            preprocess_low_resolution._check_input_array(array)


# np.arange(25).reshape(5, 5)
# with indexing [lon, lat], in xy looks like:
# 4 9 14 19 24
# 3 8 13 18 23
# 2 7 12 17 22
# 1 6 11 16 21
# 0 5 10 15 20


class Test_primary_to_secondary(unittest.TestCase):
    def test_ok(self):
        arr = np.arange(25).reshape(5, 5)
        expected = np.array([[6, 8], [16, 18]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess_low_resolution._primary_to_secondary(arr), expected)

    def test_real_data(self):
        E = load_data()["E"]
        # prepare_E is a wrapper for the function _primary_to_secondary
        preprocess_low_resolution.prepare_E(E)


class Test_prepare_Fx_left(unittest.TestCase):
    def test_ok(self):
        Fx = np.arange(25).reshape(5, 5)
        expected = np.array([[1, 3], [11, 13]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess_low_resolution.prepare_Fx_left(Fx), expected)

    def test_real_data(self):
        Fx = load_data()["Fx"]
        preprocess_low_resolution.prepare_Fx_left(Fx)


class Test_prepare_Fx_right(unittest.TestCase):
    def test_ok(self):
        Fx = np.arange(25).reshape(5, 5)
        expected = np.array([[11, 13], [21, 23]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess_low_resolution.prepare_Fx_right(Fx), expected)

    def test_real_data(self):
        Fx = load_data()["Fx"]
        preprocess_low_resolution.prepare_Fx_right(Fx)


class Test_prepare_Fy_bottom(unittest.TestCase):
    def test_ok(self):
        Fy = np.arange(25).reshape(5, 5)
        expected = np.array([[5, 7], [15, 17]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess_low_resolution.prepare_Fy_bottom(Fy), expected)

    def test_real_data(self):
        Fy = load_data()["Fy"]
        preprocess_low_resolution.prepare_Fy_bottom(Fy)


class Test_prepare_Fy_top(unittest.TestCase):
    def test_ok(self):
        Fy = np.arange(25).reshape(5, 5)
        expected = np.array([[7, 9], [17, 19]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess_low_resolution.prepare_Fy_top(Fy), expected)

    def test_real_data(self):
        Fy = load_data()["Fy"]
        preprocess_low_resolution.prepare_Fy_top(Fy)
