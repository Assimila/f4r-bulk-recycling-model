import unittest

import numpy as np

from bulk_recycling_model import rotate

# numpy indexing (i, j) = (lon, lat)
# 0,2 1,2 2,2 3,2
# 0,1 1,1 2,1 3,1
# 0,0 1,0 2,0 3,0

# after rotating 90 degrees
# 0,3 1,3 2,3
# 0,2 1,2 2,2
# 0,1 1,1 2,1
# 0,0 1,0 2,0


class Test_rot90(unittest.TestCase):
    def test_ok(self):
        arr = np.zeros((4, 3))
        arr[0, 0] = 1
        out = rotate.rot90(arr)
        self.assertEqual(out.shape, (3, 4))
        self.assertEqual(out[2, 0], 1)

    def test_full_rotation(self):
        arr = np.arange(12).reshape(4, 3)
        out = rotate.rot90(arr, k=4)
        np.testing.assert_array_equal(out, arr)

    def test_rotate_and_back(self):
        arr = np.arange(12).reshape(4, 3)
        out = rotate.rot90(arr)
        out = rotate.rot90(out, k=-1)
        np.testing.assert_array_equal(out, arr)


class Test_rot90_flux(unittest.TestCase):
    def test_ok(self):
        Fx = np.zeros((4, 3))
        Fx[0, 0] = 1
        Fy = np.zeros((4, 3))
        Fy[0, 0] = 1

        Fxx, Fyy = rotate.rot90_flux(Fx, Fy)

        self.assertEqual(Fxx.shape, (3, 4))
        self.assertEqual(Fxx[2, 0], -1)

        self.assertEqual(Fyy.shape, (3, 4))
        self.assertEqual(Fyy[2, 0], 1)

    def test_full_rotation(self):
        Fx = np.zeros((4, 3))
        Fx[0, 0] = 1
        Fy = np.zeros((4, 3))
        Fy[0, 0] = 1

        Fxx, Fyy = rotate.rot90_flux(Fx, Fy, k=4)

        np.testing.assert_array_equal(Fx, Fxx)
        np.testing.assert_array_equal(Fy, Fyy)

    def test_rotate_and_back(self):
        Fx = np.zeros((4, 3))
        Fx[0, 0] = 1
        Fy = np.zeros((4, 3))
        Fy[0, 0] = 1

        Fxx, Fyy = rotate.rot90_flux(Fx, Fy)
        Fxx, Fyy = rotate.rot90_flux(Fxx, Fyy, k=-1)

        np.testing.assert_array_equal(Fx, Fxx)
        np.testing.assert_array_equal(Fy, Fyy)


class Test_rot90_flux_lrbt(unittest.TestCase):

    def test_ok(self):
        Fx_left = np.zeros((4, 3))
        Fx_left[0, 0] = 1
        Fx_right = np.zeros((4, 3))
        Fx_right[0, 0] = 2
        Fy_bottom = np.zeros((4, 3))
        Fy_bottom[0, 0] = 3
        Fy_top = np.zeros((4, 3))
        Fy_top[0, 0] = 4

        Fxx_left, Fxx_right, Fyy_bottom, Fyy_top = rotate.rot90_flux_lrbt(
            Fx_left, Fx_right, Fy_bottom, Fy_top
        )

        self.assertEqual(Fxx_left.shape, (3, 4))
        self.assertEqual(Fxx_left[2, 0], -4)

        self.assertEqual(Fxx_right.shape, (3, 4))
        self.assertEqual(Fxx_right[2, 0], -3)

        self.assertEqual(Fyy_bottom.shape, (3, 4))
        self.assertEqual(Fyy_bottom[2, 0], 1)

        self.assertEqual(Fyy_top.shape, (3, 4))
        self.assertEqual(Fyy_top[2, 0], 2)
