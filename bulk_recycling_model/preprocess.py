import numpy as np


def _check_input_array(array: np.ndarray) -> None:
    """
    Basic sanity checks on input data.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")
    if array.shape[0] < 3 or array.shape[1] < 3:
        # need at least 3 points on the primary grid
        # -> 2 points on the secondary grid
        # for the boundary conditions to work
        raise ValueError("Input array must have at least 3 rows and 3 columns")
    if np.isnan(array).any():
        raise ValueError("Input array must not contain NaN values")


def prepare_E(E: np.ndarray) -> np.ndarray:
    """
    Compute the evaporation on the secondary grid.
    This is the average of the surrounding four points on the primary grid.

    Args:
        E:
            (N, M) array of evaporation on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.
            Should have units of mm/day.

    Returns:
        (N-1, M-1) array of evaporation on the secondary grid.
        Units: mm/day.
    """
    _check_input_array(E)

    # Compute the average of the four points on the primary grid
    E_secondary = (
        (
            E[:-1, :-1]  # (i, j)
            + E[:-1, 1:]  # (i, j+1)
            + E[1:, :-1]  # (i+1, j)
            + E[1:, 1:]  # (i+1, j+1)
        )
        / 4
    )

    return E_secondary


def prepare_Fx_left(Fx: np.ndarray) -> np.ndarray:
    """
    Compute the longitudinal water vapour flux on the left hand side of each cell on the secondary grid.

    If a cell has central point (i+1/2, j+1/2) on the secondary grid,
    this function computes fluxes at (i, j+1/2),
    as an average of (i, j) and (i, j+1).

    Args:
        Fx:
            (N, M) array of longitudinal water vapor flux on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.
            Should have units of centimetre x m/s.

    Returns:
        (N-1, M-1) array of longitudinal water vapor flux on the secondary grid.
        Units: centimetre x m/s.
    """
    _check_input_array(Fx)

    # Compute the average of the two points on the primary grid
    Fx_left = (
        (
            Fx[:-1, :-1]  # (i, j)
            + Fx[:-1, 1:]  # (i, j+1)
        )
        / 2
    )

    return Fx_left


def prepare_Fx_right(Fx: np.ndarray) -> np.ndarray:
    """
    Compute the longitudinal water vapour flux on the right hand side of each cell on the secondary grid.

    If a cell has central point (i+1/2, j+1/2) on the secondary grid,
    this function computes fluxes at (i+1, j+1/2),
    as an average of (i+1, j) and (i+1, j+1).

    Args:
        Fx:
            (N, M) array of longitudinal water vapor flux on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.
            Should have units of centimetre x m/s.

    Returns:
        (N-1, M-1) array of longitudinal water vapor flux on the secondary grid.
        Units: centimetre x m/s.
    """
    _check_input_array(Fx)

    # Compute the average of the two points on the primary grid
    Fx_right = (
        (
            Fx[1:, :-1]  # (i+1, j)
            + Fx[1:, 1:]  # (i+1, j+1)
        )
        / 2
    )

    return Fx_right


def prepare_Fy_bottom(Fy: np.ndarray) -> np.ndarray:
    """
    Compute the latitudinal water vapour flux on the bottom side of each cell on the secondary grid.

    If a cell has central point (i+1/2, j+1/2) on the secondary grid,
    this function computes fluxes at (i+1/2, j),
    as an average of (i, j) and (i+1, j).

    Args:
        Fy:
            (N, M) array of latitudinal water vapor flux on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.
            Should have units of centimetre x m/s.

    Returns:
        (N-1, M-1) array of latitudinal water vapor flux on the secondary grid.
        Units: centimetre x m/s.
    """
    _check_input_array(Fy)

    # Compute the average of the two points on the primary grid
    Fy_bottom = (
        (
            Fy[:-1, :-1]  # (i, j)
            + Fy[1:, :-1]  # (i+1, j)
        )
        / 2
    )

    return Fy_bottom


def prepare_Fy_top(Fy: np.ndarray) -> np.ndarray:
    """
    Compute the latitudinal water vapour flux on the top side of each cell on the secondary grid.

    If a cell has central point (i+1/2, j+1/2) on the secondary grid,
    this function computes fluxes at (i+1/2, j+1),
    as an average of (i, j+1) and (i+1, j+1).

    Args:
        Fy:
            (N, M) array of latitudinal water vapor flux on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.
            Should have units of centimetre x m/s.

    Returns:
        (N-1, M-1) array of latitudinal water vapor flux on the secondary grid.
        Units: centimetre x m/s.
    """
    _check_input_array(Fy)

    # Compute the average of the two points on the primary grid
    Fy_top = (
        (
            Fy[:-1, 1:]  # (i, j+1)
            + Fy[1:, 1:]  # (i+1, j+1)
        )
        / 2
    )

    return Fy_top
