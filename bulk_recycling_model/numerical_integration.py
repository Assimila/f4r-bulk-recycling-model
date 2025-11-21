"""
The utilities in this module are to assist with the vertical integration of atmospheric moisture,
to attain column densities of water vapour depth.

Generally a dataset comes in the form of a gridded 3+ dimensional array.
The pressure dimension may have coordinates [1000, 900, 800, 600, 400, 100] in millibars,
but the integral that we want to compute is from the surface (ie. surface pressure) to the top of the atmosphere,
which will vary from point to point, and in time.

If surface pressure at some (lat, lon) is 960 millibars,
then we want to make the integral from 960 to 0 millibars.
The gridded data value at 1000 millibars (below surface level) is usually NaN.

In addition to the gridded dataset, the user must provide a surface pressure dataset,
and they may also provide a third dataset that measures the integrand at surface level.
"""

import numpy as np
import xarray as xr


def np_trapz_no_extrapolation(integrand: np.ndarray, surface_pressure: float, pressure_levels: np.ndarray) -> float:
    """
    Integrate from surface pressure to top of atmosphere without extrapolation.

    This function is suitable for use with xarray `apply_ufunc`.

    Simply truncates the integrand to discard pressure levels greater than surface pressure (ie. below surface level).

    Args:
        integrand: 1-dimensional array of values (observations) to integrate.
            should be ordered from surface to top of atmosphere.
        surface_pressure: scalar value of surface pressure.
        pressure_levels: 1-dimensional array of pressure levels.
            should be ordered from surface to top of atmosphere, and the same length as integrand.

    Returns:
        A scalar value of the integral from surface pressure to top of atmosphere,
        with units defined by the inputs (integrand * pressure_levels).
        Be careful with the sign, which may be opposite to what you expect.
    """
    if integrand.size != pressure_levels.size:
        raise ValueError(
            "integrand and pressure_levels must have the same length. "
            f"Got integrand of length {integrand.size} and pressure_levels of length {pressure_levels.size}."
        )
    mask = pressure_levels <= surface_pressure
    y = integrand[mask]
    x = pressure_levels[mask]
    result = np.trapezoid(y, x)
    if isinstance(result, np.ndarray):
        raise ValueError("Result of integration should be a scalar, but got an array.")
    return result


def np_trapz_with_extrapolation(integrand: np.ndarray, surface_pressure: float, pressure_levels: np.ndarray) -> float:
    """
    Integrate from surface pressure to top of atmosphere with extrapolation.

    This function is suitable for use with xarray `apply_ufunc`.

    Truncates the integrand to discard pressure levels greater than surface pressure (ie. below surface level).
    Creates a point at the surface pressure,
    using the value of the integrand at the lowest pressure level above the surface pressure.

    Args:
        integrand: 1-dimensional array of values (observations) to integrate.
            should be ordered from surface to top of atmosphere.
        surface_pressure: scalar value of surface pressure.
        pressure_levels: 1-dimensional array of pressure levels.
            should be ordered from surface to top of atmosphere, and the same length as integrand.

    Returns:
        A scalar value of the integral from surface pressure to top of atmosphere,
        with units defined by the inputs (integrand * pressure_levels).
        Be careful with the sign, which may be opposite to what you expect.
    """
    if integrand.size != pressure_levels.size:
        raise ValueError(
            "integrand and pressure_levels must have the same length. "
            f"Got integrand of length {integrand.size} and pressure_levels of length {pressure_levels.size}."
        )
    mask = pressure_levels <= surface_pressure
    y = integrand[mask]
    x = pressure_levels[mask]
    # add a point @ surface pressure
    y_at_surface = y[0]
    y = np.concatenate(([y_at_surface], y))
    x = np.concatenate(([surface_pressure], x))
    result = np.trapezoid(y, x)
    if isinstance(result, np.ndarray):
        raise ValueError("Result of integration should be a scalar, but got an array.")
    return result


def np_trapz_with_surface_value(
    integrand: np.ndarray, surface_pressure: float, surface_value: float, pressure_levels: np.ndarray
) -> float:
    """
    Integrate from surface pressure to top of atmosphere with extrapolation.

    This function is suitable for use with xarray `apply_ufunc`.

    Truncates the integrand to discard pressure levels greater than surface pressure (ie. below surface level).
    Creates a point at the surface pressure, using the provided surface_value.

    Args:
        integrand: 1-dimensional array of values (observations) to integrate.
            should be ordered from surface to top of atmosphere.
        surface_pressure: scalar value of surface pressure.
        surface_value: scalar value of the integrand at the surface pressure.
        pressure_levels: 1-dimensional array of pressure levels.
            should be ordered from surface to top of atmosphere, and the same length as integrand.

    Returns:
        A scalar value of the integral from surface pressure to top of atmosphere,
        with units defined by the inputs (integrand * pressure_levels).
        Be careful with the sign, which may be opposite to what you expect.
    """
    if integrand.size != pressure_levels.size:
        raise ValueError(
            "integrand and pressure_levels must have the same length. "
            f"Got integrand of length {integrand.size} and pressure_levels of length {pressure_levels.size}."
        )
    mask = pressure_levels <= surface_pressure
    y = integrand[mask]
    x = pressure_levels[mask]
    # add a point @ surface pressure
    y = np.concatenate(([surface_value], y))
    x = np.concatenate(([surface_pressure], x))
    result = np.trapezoid(y, x)
    if isinstance(result, np.ndarray):
        raise ValueError("Result of integration should be a scalar, but got an array.")
    return result


def integrate_no_extrapolation(
    gridded_data: xr.DataArray,
    surface_pressure: xr.DataArray,
    pressure_levels_dim: str = "level",
    check_finite: bool = True,
) -> xr.DataArray:
    """
    Integrate gridded data from surface pressure to top of atmosphere without extrapolation.

    Simply truncates the integrand to discard pressure levels greater than surface pressure (ie. below surface level).

    Args:
        gridded_data: The integrand.
            A DataArray with dimensions including `pressure_levels_dim`,
            which should be ordered from surface to top of atmosphere.
        surface_pressure: Surface pressure (lower bound of the integral).
            A DataArray with the same dimensions as `gridded_data` excluding `pressure_levels_dim`.
        pressure_levels_dim: The name of the dimension in gridded_data that represents pressure levels.
        check_finite: If True, checks that the result of the integration is finite.

    Returns:
        A DataArray with the same dimensions as `gridded_data` excluding `pressure_levels_dim`.
        With units defined by the inputs.
        Be careful with the sign, which may be opposite to what you expect.
    """
    # checks
    gridded_dims = set(gridded_data.dims)
    if pressure_levels_dim not in gridded_dims:
        raise ValueError(f"gridded_data does not contain dimension {pressure_levels_dim}")
    broadcast_dims = gridded_dims - {pressure_levels_dim}
    surface_pressure_dims = set(surface_pressure.dims)
    if surface_pressure_dims != broadcast_dims:
        raise ValueError(f"surface_pressure has dimensions {surface_pressure_dims}, expected {broadcast_dims}")

    #ED added dask allowed
    da = xr.apply_ufunc(
        np_trapz_no_extrapolation,
        gridded_data,
        surface_pressure,
        input_core_dims=[[pressure_levels_dim], []],
        vectorize=True,
        #dask="allowed",
        kwargs={
            # pass the gridded pressure levels directly to the function as kwargs
            "pressure_levels": gridded_data.coords[pressure_levels_dim].values,
        },
    )

    if check_finite:
        if not xr.ufuncs.isfinite(da).all():
            raise ValueError("Integration result is not finite")
        
    return da


def integrate_with_extrapolation(
    gridded_data: xr.DataArray,
    surface_pressure: xr.DataArray,
    pressure_levels_dim: str = "level",
    check_finite: bool = True,
) -> xr.DataArray:
    """
    Integrate gridded data from surface pressure to top of atmosphere with extrapolation.

    Truncates the integrand to discard pressure levels greater than surface pressure (ie. below surface level).
    Creates a point at the surface pressure,
    using the value of the integrand at the lowest pressure level above the surface pressure.

    Args:
        gridded_data: The integrand.
            A DataArray with dimensions including `pressure_levels_dim`,
            which should be ordered from surface to top of atmosphere.
        surface_pressure: Surface pressure (lower bound of the integral).
            A DataArray with the same dimensions as `gridded_data` excluding `pressure_levels_dim`.
        pressure_levels_dim: The name of the dimension in gridded_data that represents pressure levels.
        check_finite: If True, checks that the result of the integration is finite.

    Returns:
        A DataArray with the same dimensions as `gridded_data` excluding `pressure_levels_dim`.
        With units defined by the inputs.
        Be careful with the sign, which may be opposite to what you expect.
    """
    # checks
    gridded_dims = set(gridded_data.dims)
    if pressure_levels_dim not in gridded_dims:
        raise ValueError(f"gridded_data does not contain dimension {pressure_levels_dim}")
    broadcast_dims = gridded_dims - {pressure_levels_dim}
    surface_pressure_dims = set(surface_pressure.dims)
    if surface_pressure_dims != broadcast_dims:
        raise ValueError(f"surface_pressure has dimensions {surface_pressure_dims}, expected {broadcast_dims}")

    #ED added dask allowed
    da = xr.apply_ufunc(
        np_trapz_with_extrapolation,
        gridded_data,
        surface_pressure,
        input_core_dims=[[pressure_levels_dim], []],
        vectorize=True,
        #dask="allowed",
        kwargs={
            # pass the gridded pressure levels directly to the function as kwargs
            "pressure_levels": gridded_data.coords[pressure_levels_dim].values,
        },
    )

    if check_finite:
        if not xr.ufuncs.isfinite(da).all():
            #print("INTEGRATION NOT FINITE")
            raise ValueError("Integration result is not finite")
        
    return da


def integrate_with_surface_value(
    gridded_data: xr.DataArray,
    surface_pressure: xr.DataArray,
    surface_value: xr.DataArray,
    pressure_levels_dim: str = "level",
    check_finite: bool = True,
) -> xr.DataArray:
    """
    Integrate gridded data from surface pressure to top of atmosphere with extrapolation.

    Truncates the integrand to discard pressure levels greater than surface pressure (ie. below surface level).
    Creates a point at the surface pressure, using the provided surface_value.

    Args:
        gridded_data: The integrand.
            A DataArray with dimensions including `pressure_levels_dim`,
            which should be ordered from surface to top of atmosphere.
        surface_pressure: Surface pressure (lower bound of the integral).
            A DataArray with the same dimensions as `gridded_data` excluding `pressure_levels_dim`.
        surface_value: Value of the integrand at the surface pressure.
            A DataArray with the same dimensions as `gridded_data` excluding `pressure_levels_dim`.
        pressure_levels_dim: The name of the dimension in gridded_data that represents pressure levels.
        check_finite: If True, checks that the result of the integration is finite.

    Returns:
        A DataArray with the same dimensions as `gridded_data` excluding `pressure_levels_dim`.
        With units defined by the inputs.
        Be careful with the sign, which may be opposite to what you expect.
    """
    # checks
    gridded_dims = set(gridded_data.dims)
    if pressure_levels_dim not in gridded_dims:
        raise ValueError(f"gridded_data does not contain dimension {pressure_levels_dim}")
    broadcast_dims = gridded_dims - {pressure_levels_dim}
    surface_pressure_dims = set(surface_pressure.dims)
    if surface_pressure_dims != broadcast_dims:
        raise ValueError(f"surface_pressure has dimensions {surface_pressure_dims}, expected {broadcast_dims}")
    surface_value_dims = set(surface_value.dims)
    if surface_value_dims != broadcast_dims:
        raise ValueError(f"surface_value has dimensions {surface_value_dims}, expected {broadcast_dims}")

    #ED added dask allowed
    da = xr.apply_ufunc(
        np_trapz_with_surface_value,
        gridded_data,
        surface_pressure,
        surface_value,
        input_core_dims=[[pressure_levels_dim], [], []],
        vectorize=True,
        #dask="allowed",
        kwargs={
            # pass the gridded pressure levels directly to the function as kwargs
            "pressure_levels": gridded_data.coords[pressure_levels_dim].values,
        },
    )

    if check_finite:
        if not xr.ufuncs.isfinite(da).all():
            raise ValueError("Integration result is not finite")

    return da
