import unittest

import numpy as np

from bulk_recycling_model import utils


class Test_buffer(unittest.TestCase):
    def test_1d(self):
        a = np.array([1, 2, 3], dtype=float)
        expected = np.array([np.nan, 1, 2, 3, np.nan])
        result = utils.buffer(a)
        np.testing.assert_array_equal(result, expected)

        buffer_value = 0
        expected = np.array([0, 1, 2, 3, 0])
        result = utils.buffer(a, value=buffer_value)
        np.testing.assert_array_equal(result, expected)

    def test_2d(self):
        a = np.array([[1, 2], [3, 4]], dtype=float)
        expected = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, 1, 2, np.nan],
                [np.nan, 3, 4, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )
        result = utils.buffer(a)
        np.testing.assert_array_equal(result, expected)

        buffer_value = 0
        expected = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
        result = utils.buffer(a, value=buffer_value)
        np.testing.assert_array_equal(result, expected)


class Test_unbuffer(unittest.TestCase):
    def test_too_small(self):
        a = np.array([1, 2])
        with self.assertRaises(ValueError):
            utils.unbuffer(a)

    def test_1d(self):
        a = np.array([np.nan, 1, 2, 3, np.nan])
        expected = np.array([1, 2, 3])
        result = utils.unbuffer(a)
        np.testing.assert_array_equal(result, expected)

    def test_2d(self):
        a = np.arange(9).reshape(3, 3)
        expected = np.array([[4]])
        result = utils.unbuffer(a)
        np.testing.assert_array_equal(result, expected)

    def test_is_view(self):
        a = np.arange(9).reshape(3, 3)
        result = utils.unbuffer(a)
        self.assertIsNotNone(result.base)


class Test_check_lr_flux(unittest.TestCase):
    def test_check_lr_flux(self):
        Fx_left = np.array([[1, 2], [3, 4]])
        Fx_right = np.array([[3, 4], [5, 6]])
        utils.check_lr_flux(Fx_left, Fx_right)

    def test_inconsistent_flux(self):
        Fx_left = np.array([[1, 2], [3, -1]])
        Fx_right = np.array([[3, 4], [5, 6]])
        with self.assertRaises(ValueError):
            utils.check_lr_flux(Fx_left, Fx_right)


class Test_check_tb_flux(unittest.TestCase):
    def test_check_tb_flux(self):
        Fy_bottom = np.array([[1, 2], [3, 4]])
        Fy_top = np.array([[2, 5], [4, 6]])
        utils.check_tb_flux(Fy_top, Fy_bottom)

    def test_inconsistent_flux(self):
        Fy_bottom = np.array([[1, 2], [3, -1]])
        Fy_top = np.array([[2, 5], [4, 6]])
        with self.assertRaises(ValueError):
            utils.check_tb_flux(Fy_top, Fy_bottom)


class Test_outflow_mask(unittest.TestCase):
    def test_ok(self):
        Fx_left = np.array([[-1, 0, 1], [0, 0, 0], [0, 0, 0]])
        Fx_right = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 1]])
        Fy_bottom = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        Fy_top = np.array([[0, 0, -1], [0, 0, 0], [0, 0, 1]])
        mask = utils.outflow_mask(Fx_left, Fx_right, Fy_bottom, Fy_top)
        expected = np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
            ]
        )
        np.testing.assert_array_equal(mask, expected)


class Test_inflow_mask(unittest.TestCase):
    def test_ok(self):
        Fx_left = np.array([[-1, 0, 1], [0, 0, 0], [0, 0, 0]])
        Fx_right = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 1]])
        Fy_bottom = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        Fy_top = np.array([[0, 0, -1], [0, 0, 0], [0, 0, 1]])
        mask = utils.inflow_mask(Fx_left, Fx_right, Fy_bottom, Fy_top)
        expected = np.array(
            [
                [1, 0, 1, 1, 1],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 0, 1],
            ]
        )
        np.testing.assert_array_equal(mask, expected)
