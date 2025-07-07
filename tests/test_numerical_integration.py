import unittest

import numpy as np
import xarray as xr

from bulk_recycling_model import numerical_integration


class Test_np_trapz_no_extrapolation(unittest.TestCase):
    def test_ok(self):
        integrand = np.array([4, 2, 1, 0])
        surface_pressure = 1000
        pressure_levels = np.array([1000, 900, 800, 100])
        result = numerical_integration.np_trapz_no_extrapolation(integrand, surface_pressure, pressure_levels)
        expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result, expected)

    def test_surface_pressure_not_in_levels(self):
        integrand = np.array([4, 2, 1, 0])
        surface_pressure = 990
        pressure_levels = np.array([1000, 900, 800, 100])
        result = numerical_integration.np_trapz_no_extrapolation(integrand, surface_pressure, pressure_levels)
        expected = -1 * (1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result, expected)


class Test_np_trapz_with_extrapolation(unittest.TestCase):
    def test_ok(self):
        integrand = np.array([4, 2, 1, 0])
        surface_pressure = 1000
        pressure_levels = np.array([1000, 900, 800, 100])
        result = numerical_integration.np_trapz_with_extrapolation(integrand, surface_pressure, pressure_levels)
        expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result, expected)

    def test_surface_pressure_not_in_levels(self):
        integrand = np.array([4, 2, 1, 0])
        surface_pressure = 990
        pressure_levels = np.array([1000, 900, 800, 100])
        result = numerical_integration.np_trapz_with_extrapolation(integrand, surface_pressure, pressure_levels)
        expected = -1 * (2 * 90 + 1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result, expected)


class Test_np_trapz_with_surface_value(unittest.TestCase):
    def test_ok(self):
        integrand = np.array([4, 2, 1, 0])
        surface_pressure = 1000
        surface_value = 4
        pressure_levels = np.array([1000, 900, 800, 100])
        result = numerical_integration.np_trapz_with_surface_value(
            integrand, surface_pressure, surface_value, pressure_levels
        )
        expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result, expected)

    def test_surface_pressure_not_in_levels(self):
        integrand = np.array([4, 2, 1, 0])
        surface_pressure = 990
        surface_value = 3
        pressure_levels = np.array([1000, 900, 800, 100])
        result = numerical_integration.np_trapz_with_surface_value(
            integrand, surface_pressure, surface_value, pressure_levels
        )
        expected = -1 * (2.5 * 90 + 1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result, expected)


class Test_integrate_no_extrapolation(unittest.TestCase):
    def test_ok(self):
        integrand = xr.DataArray([4, 2, 1, 0], coords={"level": [1000, 900, 800, 100]})
        surface_pressure = xr.DataArray(1000)
        result = numerical_integration.integrate_no_extrapolation(integrand, surface_pressure)
        expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result.item(), expected)

    def test_surface_pressure_not_in_levels(self):
        integrand = xr.DataArray([4, 2, 1, 0], coords={"level": [1000, 900, 800, 100]})
        surface_pressure = xr.DataArray(990)
        result = numerical_integration.integrate_no_extrapolation(integrand, surface_pressure)
        expected = -1 * (1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result.item(), expected)

    def test_with_more_dims(self):
        """
        also include latitude and longitude dimensions
        """
        lat = [0, 10, 20]
        lon = [30, 40]
        levels = [1000, 900, 800, 100]
        _integrand = [4, 2, 1, 0]
        integrand = xr.DataArray(
            np.tile(_integrand, (len(lat), len(lon), 1)),
            dims=["lat", "lon", "level"],
            coords={
                "level": levels,
                "lat": lat,
                "lon": lon,
            },
        )
        surface_pressure = xr.DataArray(
            np.tile(1000, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        result = numerical_integration.integrate_no_extrapolation(integrand, surface_pressure)
        _expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        expected = xr.DataArray(
            np.tile(_expected, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        xr.testing.assert_equal(result, expected)

    def test_transposed_dims(self):
        """
        screw around with the dimension ordering to check that xarray handles it correctly
        """
        lat = [0, 10, 20]
        lon = [30, 40]
        levels = [1000, 900, 800, 100]
        _integrand = [4, 2, 1, 0]
        integrand = xr.DataArray(
            np.tile(_integrand, (len(lat), len(lon), 1)),
            dims=["lat", "lon", "level"],
            coords={
                "level": levels,
                "lat": lat,
                "lon": lon,
            },
        )
        integrand = integrand.transpose("lat", "level", "lon")
        surface_pressure = xr.DataArray(
            np.tile(1000, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        surface_pressure = surface_pressure.transpose("lon", "lat")
        result = numerical_integration.integrate_no_extrapolation(integrand, surface_pressure)
        _expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        expected = xr.DataArray(
            np.tile(_expected, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        xr.testing.assert_equal(result, expected, check_dim_order=False)


class Test_integrate_with_extrapolation(unittest.TestCase):
    def test_ok(self):
        integrand = xr.DataArray([4, 2, 1, 0], coords={"level": [1000, 900, 800, 100]})
        surface_pressure = xr.DataArray(1000)
        result = numerical_integration.integrate_with_extrapolation(integrand, surface_pressure)
        expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result.item(), expected)

    def test_surface_pressure_not_in_levels(self):
        integrand = xr.DataArray([4, 2, 1, 0], coords={"level": [1000, 900, 800, 100]})
        surface_pressure = xr.DataArray(990)
        result = numerical_integration.integrate_with_extrapolation(integrand, surface_pressure)
        expected = -1 * (2 * 90 + 1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result.item(), expected)

    def test_with_more_dims(self):
        """
        also include latitude and longitude dimensions
        """
        lat = [0, 10, 20]
        lon = [30, 40]
        levels = [1000, 900, 800, 100]
        _integrand = [4, 2, 1, 0]
        integrand = xr.DataArray(
            np.tile(_integrand, (len(lat), len(lon), 1)),
            dims=["lat", "lon", "level"],
            coords={
                "level": levels,
                "lat": lat,
                "lon": lon,
            },
        )
        surface_pressure = xr.DataArray(
            np.tile(1000, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        result = numerical_integration.integrate_with_extrapolation(integrand, surface_pressure)
        _expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        expected = xr.DataArray(
            np.tile(_expected, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        xr.testing.assert_equal(result, expected)

    def test_transposed_dims(self):
        """
        screw around with the dimension ordering to check that xarray handles it correctly
        """
        lat = [0, 10, 20]
        lon = [30, 40]
        levels = [1000, 900, 800, 100]
        _integrand = [4, 2, 1, 0]
        integrand = xr.DataArray(
            np.tile(_integrand, (len(lat), len(lon), 1)),
            dims=["lat", "lon", "level"],
            coords={
                "level": levels,
                "lat": lat,
                "lon": lon,
            },
        )
        integrand = integrand.transpose("lat", "level", "lon")
        surface_pressure = xr.DataArray(
            np.tile(1000, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        surface_pressure = surface_pressure.transpose("lon", "lat")
        result = numerical_integration.integrate_with_extrapolation(integrand, surface_pressure)
        _expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        expected = xr.DataArray(
            np.tile(_expected, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        xr.testing.assert_equal(result, expected, check_dim_order=False)


class Test_integrate_with_surface_value(unittest.TestCase):
    def test_ok(self):
        integrand = xr.DataArray([4, 2, 1, 0], coords={"level": [1000, 900, 800, 100]})
        surface_pressure = xr.DataArray(1000)
        surface_value = xr.DataArray(4)
        result = numerical_integration.integrate_with_surface_value(integrand, surface_pressure, surface_value)
        expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result.item(), expected)

    def test_surface_pressure_not_in_levels(self):
        integrand = xr.DataArray([4, 2, 1, 0], coords={"level": [1000, 900, 800, 100]})
        surface_pressure = xr.DataArray(990)
        surface_value = xr.DataArray(3)
        result = numerical_integration.integrate_with_surface_value(integrand, surface_pressure, surface_value)
        expected = -1 * (2.5 * 90 + 1.5 * 100 + 0.5 * 700)
        self.assertAlmostEqual(result.item(), expected)

    def test_with_more_dims(self):
        """
        also include latitude and longitude dimensions
        """
        lat = [0, 10, 20]
        lon = [30, 40]
        levels = [1000, 900, 800, 100]
        _integrand = [4, 2, 1, 0]
        integrand = xr.DataArray(
            np.tile(_integrand, (len(lat), len(lon), 1)),
            dims=["lat", "lon", "level"],
            coords={
                "level": levels,
                "lat": lat,
                "lon": lon,
            },
        )
        surface_pressure = xr.DataArray(
            np.tile(1000, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        surface_value = xr.DataArray(
            np.tile(4, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        result = numerical_integration.integrate_with_surface_value(integrand, surface_pressure, surface_value)
        _expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        expected = xr.DataArray(
            np.tile(_expected, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        xr.testing.assert_equal(result, expected)

    def test_transposed_dims(self):
        """
        screw around with the dimension ordering to check that xarray handles it correctly
        """
        lat = [0, 10, 20]
        lon = [30, 40]
        levels = [1000, 900, 800, 100]
        _integrand = [4, 2, 1, 0]
        integrand = xr.DataArray(
            np.tile(_integrand, (len(lat), len(lon), 1)),
            dims=["lat", "lon", "level"],
            coords={
                "level": levels,
                "lat": lat,
                "lon": lon,
            },
        )
        integrand = integrand.transpose("lat", "level", "lon")
        surface_pressure = xr.DataArray(
            np.tile(1000, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        surface_pressure = surface_pressure.transpose("lon", "lat")
        surface_value = xr.DataArray(
            np.tile(4, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        result = numerical_integration.integrate_with_surface_value(integrand, surface_pressure, surface_value)
        _expected = -1 * (3 * 100 + 1.5 * 100 + 0.5 * 700)
        expected = xr.DataArray(
            np.tile(_expected, (len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        xr.testing.assert_equal(result, expected, check_dim_order=False)
