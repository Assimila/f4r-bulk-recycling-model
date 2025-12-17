"""
Here we provide preprocessing utilities for working with a low-resolution secondary grid.
This grid is defined directly on top of the primary grid, at half resolution.

Given a primary grid (o):

    o . o . o
    . . . . .
    o . o . o
    . . . . .
    o . o . o

Instead of defining the secondary grid (x) normally like this:

    o . o . o
    . x . x .
    o . o . o
    . x . x .
    o . o . o

Define the secondary grid directly on top of the primary grid at half resolution:

    o . o . o
    . . . . .
    o . x . o
    . . . . .
    o . o . o

This may be useful when high resolution data is numerically unstable,
or for a quick look at model results at lower resolution,
without needing to do any spatial interpolation.
"""

import numpy as np

from .preprocess import calculate_precipitation  # NOQA: F401


def _check_input_array(array: np.ndarray) -> None:
    """
    Basic sanity checks on input data
    on the primary grid.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")
    if array.shape[0] < 5 or array.shape[1] < 5:
        # need at least 5 points on the primary grid
        # -> 2 points on the secondary grid
        # for the boundary conditions to work
        raise ValueError("Input array must have at least 5 rows and 5 columns")
    if np.isnan(array).any():
        raise ValueError("Input array must not contain NaN values")


def _primary_to_secondary(
    array: np.ndarray,
    i_start: int = 1,
    j_start: int = 1,
) -> np.ndarray:
    """
    Convert a 2D array from primary grid to secondary grid.

    Args:
        array:
            (N, M) array on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.
        i_start:
            defines the x index of the first cell of the secondary grid
        j_start:
            defines the y index of the first cell of the secondary grid

    Returns:
        array on the secondary grid.
    """
    _check_input_array(array)

    return array[slice(i_start, None, 2), slice(j_start, None, 2)]


def _trim(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """
    Trim the secondary grid array to match the shape of the primary grid array.

    Args:
        primary:
            (N, M) array on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.
        secondary:
    """
    N, M = primary.shape

    # shape of array on secondary grid
    # 5 -> 2
    # 6 -> 2
    # 7 -> 3
    # etc.
    _N = (N - 1) // 2
    _M = (M - 1) // 2

    return secondary[:_N, :_M]



def prepare_E(E: np.ndarray) -> np.ndarray:
    """
    Compute the evaporation on the secondary grid.

    Args:
        E:
            (N, M) array of evaporation on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.

    Returns:
        array of evaporation on the secondary grid.
    """
    return _trim(E, _primary_to_secondary(E))


def prepare_P(P: np.ndarray) -> np.ndarray:
    """
    Compute the precipitation on the secondary grid.

    Not that this is not normally necessary, as precipitation is not used in the model.
    However, it may be useful to compare input precipitation data with the calculated precipitation data.

    Args:
        P:
            (N, M) array of precipitation on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.

    Returns:
        array of precipitation on the secondary grid.
    """
    return _trim(P, _primary_to_secondary(P))

def prepare_Fx_left(Fx: np.ndarray) -> np.ndarray:
    """
    Compute the longitudinal water vapour flux on the left hand side of each cell on the secondary grid.

    Args:
        Fx:
            (N, M) array of longitudinal water vapor flux on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.

    Returns:
        array of longitudinal water vapor flux on the secondary grid.
    """
    return _trim(Fx, _primary_to_secondary(Fx, i_start=0, j_start=1))

def prepare_Fx_right(Fx: np.ndarray) -> np.ndarray:
    """
    Compute the longitudinal water vapour flux on the right hand side of each cell on the secondary grid.

    Args:
        Fx:
            (N, M) array of longitudinal water vapor flux on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.

    Returns:
        array of longitudinal water vapor flux on the secondary grid.
    """
    return _trim(Fx, _primary_to_secondary(Fx, i_start=2, j_start=1))


def prepare_Fy_bottom(Fy: np.ndarray) -> np.ndarray:
    """
    Compute the latitudinal water vapour flux on the bottom side of each cell on the secondary grid.

    Args:
        Fy:
            (N, M) array of latitudinal water vapor flux on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.

    Returns:
        array of latitudinal water vapor flux on the secondary grid.
    """
    return _trim(Fy, _primary_to_secondary(Fy, i_start=1, j_start=0))


def prepare_Fy_top(Fy: np.ndarray) -> np.ndarray:
    """
    Compute the latitudinal water vapour flux on the top side of each cell on the secondary grid.

    Args:
        Fy:
            (N, M) array of latitudinal water vapor flux on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.

    Returns:
        array of latitudinal water vapor flux on the secondary grid.
    """
    return _trim(Fy, _primary_to_secondary(Fy, i_start=1, j_start=2))
