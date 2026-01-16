import unittest

import numpy as np

from bulk_recycling_model.clamping import clamp, smooth


class Test_smooth(unittest.TestCase):
    def test_zero_deviation(self):
        deviation = np.array([0.0])
        tolerance = 0.1
        expected = np.array([0.0])
        np.testing.assert_array_almost_equal(smooth(deviation, tolerance), expected)

    def test_small_deviation(self):
        deviation = np.array([0.05])
        tolerance = 0.1
        expected = tolerance * (1 - np.exp(-deviation / tolerance))
        np.testing.assert_array_almost_equal(smooth(deviation, tolerance), expected)

    def test_large_deviation(self):
        deviation = np.array([1e6])
        tolerance = 0.1
        expected = np.array([tolerance])  # asymptotically approaches tolerance
        np.testing.assert_array_almost_equal(smooth(deviation, tolerance), expected)

    def test_negative_deviation(self):
        deviation = np.array([-1.0])
        tolerance = 0.1
        with self.assertRaises(ValueError):
            smooth(deviation, tolerance)

    def test_gradient_at_zero(self):
        # should have derivative = 1 at deviation = 0
        tolerance = 0.1
        x = [0, 1e-6]
        y = smooth(np.array(x), tolerance)
        derivative = (y[1] - y[0]) / (x[1] - x[0])
        self.assertAlmostEqual(derivative, 1.0, places=5)

    def test_on_2d_array(self):
        deviation = np.array([[0.0, 0.1], [0.5, 1.0]])
        tolerance = 0.2
        expected = tolerance * (1 - np.exp(-deviation / tolerance))
        np.testing.assert_array_almost_equal(smooth(deviation, tolerance), expected)


class Test_clamp(unittest.TestCase):
    def test_within_bounds(self):
        x = np.arange(0, 1, 0.01)
        tolerance = 0.1
        np.testing.assert_array_almost_equal(clamp(x, tolerance=tolerance), x)

    def test_above_upper_bound(self):
        x = np.array([1.05, 1.5, 2.0])
        deviation = x - 1.0
        tolerance = 0.1
        expected = 1 + tolerance * (1 - np.exp(-deviation / tolerance))
        np.testing.assert_array_almost_equal(clamp(x, tolerance=tolerance), expected)

    def test_below_lower_bound(self):
        x = np.array([-2.0, -1.5, -0.05])
        tolerance = 0.1
        deviation = -x
        expected = -tolerance * (1 - np.exp(-deviation / tolerance))
        np.testing.assert_array_almost_equal(clamp(x, tolerance=tolerance), expected)

    def test_extreme_lower_bound(self):
        x = np.array([-1e6])
        tolerance = 0.1
        expected = np.array([-tolerance])  # asymptotically approaches -tolerance
        np.testing.assert_array_almost_equal(clamp(x, tolerance=tolerance), expected)

    def test_on_2d_array(self):
        x = np.array([[-0.3, -0.2, -0.1], [0.0, 0.5, 1.0], [1.1, 1.2, 1.3]])
        tolerance = 0.1
        expected = np.array(
            [
                [
                    -tolerance * (1 - np.exp(-0.3 / tolerance)),
                    -tolerance * (1 - np.exp(-0.2 / tolerance)),
                    -tolerance * (1 - np.exp(-0.1 / tolerance)),
                ],
                [0.0, 0.5, 1.0],
                [
                    1 + tolerance * (1 - np.exp(-0.1 / tolerance)),
                    1 + tolerance * (1 - np.exp(-0.2 / tolerance)),
                    1 + tolerance * (1 - np.exp(-0.3 / tolerance)),
                ],
            ]
        )
        np.testing.assert_array_almost_equal(clamp(x, tolerance=tolerance), expected)

    def test_custom_upper_bound(self):
        x = np.array([2.0, 2.5, 3.0])
        upper_bound = 2.0
        tolerance = 0.1
        deviation = x - upper_bound
        expected = upper_bound + tolerance * (1 - np.exp(-deviation / tolerance))
        np.testing.assert_array_almost_equal(clamp(x, upper_bound=upper_bound, tolerance=tolerance), expected)

    def test_custom_lower_bound(self):
        x = np.array([-2.0, -1.5, -1.0])
        lower_bound = -1.0
        tolerance = 0.1
        deviation = lower_bound - x
        expected = lower_bound - tolerance * (1 - np.exp(-deviation / tolerance))
        np.testing.assert_array_almost_equal(clamp(x, lower_bound=lower_bound, tolerance=tolerance), expected)

    def test_custom_bounds(self):
        x = np.array([-1.5, -1.0, 0.0, 1.0, 1.5])
        lower_bound = -1.0
        upper_bound = 1.0
        tolerance = 0.1
        expected = np.array([
            lower_bound - tolerance * (1 - np.exp(-(lower_bound - x[0]) / tolerance)),
            -1.0,
            0.0,
            1.0,
            upper_bound + tolerance * (1 - np.exp(-(x[4] - upper_bound) / tolerance))
        ])
        np.testing.assert_array_almost_equal(
            clamp(x, lower_bound=lower_bound, upper_bound=upper_bound, tolerance=tolerance), expected
        )
