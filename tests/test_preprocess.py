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
