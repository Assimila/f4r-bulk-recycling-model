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
