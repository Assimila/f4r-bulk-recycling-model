import unittest

import numpy as np

from bulk_recycling_model import axis


class TestAxis(unittest.TestCase):

    def test_invalid_n_points(self):
        with self.assertRaises(ValueError):
            axis.Axis(0, 1, 1)
        with self.assertRaises(ValueError):
            axis.Axis(0, 1, 0)
        with self.assertRaises(ValueError):
            axis.Axis(0, 1, -1)

    def test_primary(self):
        np.testing.assert_allclose(
            axis.Axis(0, 1, 5).primary,
            [0, 1, 2, 3, 4]
        )
        np.testing.assert_allclose(
            axis.Axis(0, 1.1, 5).primary,
            [0, 1.1, 2.2, 3.3, 4.4]
        )

    def test_secondary(self):
        np.testing.assert_allclose(
            axis.Axis(0, 1, 5).secondary,
            [0.5, 1.5, 2.5, 3.5]
        )
        np.testing.assert_allclose(
            axis.Axis(0, 1.1, 5).secondary,
            [0.55, 1.65, 2.75, 3.85]
        )

    def test_secondary_buffered(self):
        np.testing.assert_allclose(
            axis.Axis(0, 1, 5).secondary_buffered,
            [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        )
        np.testing.assert_allclose(
            axis.Axis(0, 1.1, 5).secondary_buffered,
            [-0.55, 0.55, 1.65, 2.75, 3.85, 4.95]
        )

    def test_half_step(self):
        np.testing.assert_allclose(
            axis.Axis(0, 1, 5).half_step,
            [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        )
        np.testing.assert_allclose(
            axis.Axis(0, 1.1, 5).half_step,
            [0, 0.55, 1.1, 1.65, 2.2, 2.75, 3.3, 3.85, 4.4]
        )
