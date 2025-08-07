import unittest

import numpy as np

from bulk_recycling_model import preprocess
from bulk_recycling_model.scaling import Scaling, UnitSystem
from tests.data.load_data import load_data


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


class Test_primary_to_secondary(unittest.TestCase):
    def test_ok(self):
        arr = np.arange(9).reshape(3, 3)
        expected = np.array([[2, 3], [5, 6]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess._primary_to_secondary(arr), expected)

    def test_real_data(self):
        E = load_data()["E"]
        # prepare_E is a wrapper for the function _primary_to_secondary
        preprocess.prepare_E(E)


class Test_prepare_Fx_left(unittest.TestCase):
    def test_ok(self):
        Fx = np.arange(9).reshape(3, 3)
        expected = np.array([[0.5, 1.5], [3.5, 4.5]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess.prepare_Fx_left(Fx), expected)

    def test_real_data(self):
        Fx = load_data()["Fx"]
        preprocess.prepare_Fx_left(Fx)


class Test_prepare_Fx_right(unittest.TestCase):
    def test_ok(self):
        Fx = np.arange(9).reshape(3, 3)
        expected = np.array([[3.5, 4.5], [6.5, 7.5]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess.prepare_Fx_right(Fx), expected)

    def test_real_data(self):
        Fx = load_data()["Fx"]
        preprocess.prepare_Fx_right(Fx)


class Test_prepare_Fy_bottom(unittest.TestCase):
    def test_ok(self):
        Fy = np.arange(9).reshape(3, 3)
        expected = np.array([[1.5, 2.5], [4.5, 5.5]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess.prepare_Fy_bottom(Fy), expected)

    def test_real_data(self):
        Fy = load_data()["Fy"]
        preprocess.prepare_Fy_bottom(Fy)


class Test_prepare_Fy_top(unittest.TestCase):
    def test_ok(self):
        Fy = np.arange(9).reshape(3, 3)
        expected = np.array([[2.5, 3.5], [5.5, 6.5]], dtype=float)
        np.testing.assert_array_almost_equal(preprocess.prepare_Fy_top(Fy), expected)

    def test_real_data(self):
        Fy = load_data()["Fy"]
        preprocess.prepare_Fy_top(Fy)


class Test_calculate_precipitation(unittest.TestCase):
    def test_constant_flux(self):
        # with no divergence in flux, precipitation = evaporation
        Fx_left = Fx_right = np.full((2, 2), 1)
        Fy_bottom = Fy_top = np.full((2, 2), -1)
        E = np.array([[1, 2], [3, 4]])
        dx = dy = 1
        expected_precipitation = E
        np.testing.assert_array_almost_equal(
            preprocess.calculate_precipitation(Fx_left, Fx_right, Fy_bottom, Fy_top, E, dx, dy),
            expected_precipitation,
        )

    def test_p_lower_e(self):
        # div flux > 0 implies precipitation < evaporation
        Fx_left = np.array([[0, 1], [1, 2]])
        Fx_right = Fx_left + 1
        Fy_bottom = Fy_top = np.zeros((2, 2))
        E = np.array([[2, 2], [2, 2]])
        dx = dy = 1
        expected_precipitation = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_almost_equal(
            preprocess.calculate_precipitation(Fx_left, Fx_right, Fy_bottom, Fy_top, E, dx, dy),
            expected_precipitation,
        )

    def test_p_higher_e(self):
        # div flux < 0 implies precipitation > evaporation
        Fx_left = np.array([[0, 1], [1, 2]])
        Fx_right = Fx_left - 1
        Fy_bottom = Fy_top = np.zeros((2, 2))
        E = np.array([[2, 2], [2, 2]])
        dx = dy = 1
        expected_precipitation = np.array([[3, 3], [3, 3]])
        np.testing.assert_array_almost_equal(
            preprocess.calculate_precipitation(Fx_left, Fx_right, Fy_bottom, Fy_top, E, dx, dy),
            expected_precipitation,
        )

    def test_real_data(self):
        dat = load_data()
        lon = dat["lon"]
        lat = dat["lat"]

        L = lon[-1] - lon[0]  # degrees
        # convert to meters
        L = L * 111e3 * np.cos(np.deg2rad(lat.mean()))

        dx = L / (len(lon) - 1)  # meters

        H = lat[-1] - lat[0]  # degrees
        # convert to meters
        H = H * 111e3

        dy = H / (len(lat) - 1)  # meters

        # get a scaling object to convert from natural to scaled units
        scaling = Scaling(H)

        dx = scaling.distance.convert(dx, UnitSystem.SI, UnitSystem.scaled)
        dy = scaling.distance.convert(dy, UnitSystem.SI, UnitSystem.scaled)

        Fx = dat["Fx"]
        Fx = scaling.water_vapor_flux.convert(Fx, UnitSystem.natural, UnitSystem.scaled)
        Fx_left = preprocess.prepare_Fx_left(Fx)
        Fx_right = preprocess.prepare_Fx_right(Fx)

        Fy = dat["Fy"]
        Fy = scaling.water_vapor_flux.convert(Fy, UnitSystem.natural, UnitSystem.scaled)
        Fy_bottom = preprocess.prepare_Fy_bottom(Fy)
        Fy_top = preprocess.prepare_Fy_top(Fy)

        E = dat["E"]
        E = scaling.evaporation.convert(E, UnitSystem.natural, UnitSystem.scaled)
        E = preprocess.prepare_E(E)

        _ = preprocess.calculate_precipitation(Fx_left, Fx_right, Fy_bottom, Fy_top, E, dx, dy)


class Test_fix_negative_precipitation(unittest.TestCase):
    def test_positive_precipitation(self):
        P = np.array([[1, 2], [3, 4]])
        expected = P.copy()
        np.testing.assert_array_almost_equal(preprocess.fix_negative_precipitation(P), expected)

    def test_negative_precipitation(self):
        P = np.array([[1, -2], [-3, 4]])
        expected = np.array([[1, 0], [0, 4]])
        np.testing.assert_array_almost_equal(preprocess.fix_negative_precipitation(P), expected)

    def test_zero_precipitation(self):
        P = np.array([[0, 0], [0, 0]])
        expected = P.copy()
        np.testing.assert_array_almost_equal(preprocess.fix_negative_precipitation(P), expected)
