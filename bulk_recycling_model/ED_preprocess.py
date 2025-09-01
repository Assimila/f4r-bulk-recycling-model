import numpy as np


def _check_input_array(array: np.ndarray) -> None:
    """
    Basic sanity checks on input data
    on the primary grid.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")
    if array.shape[0] < 3 or array.shape[1] < 3:
        # need at least 3 points on the primary grid
        # -> 2 points on the secondary grid
        # for the boundary conditions to work
        raise ValueError("Input array must have at least 3 rows and 3 columns")
    #ED edit: can't filter for nans because of clipping with shapefile
    #if np.isnan(array).any():
    #    raise ValueError("Input array must not contain NaN values")


def _primary_to_secondary(array: np.ndarray) -> np.ndarray:
    """
    Convert a 2D array from primary grid to secondary grid.
    This is the average of the surrounding four points on the primary grid.

    Args:
        array:
            (N, M) array on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.

    Returns:
        (N-1, M-1) array on the secondary grid.
    """
    _check_input_array(array)

    return (
        (
            array[:-1, :-1]  # (i, j)
            + array[:-1, 1:]  # (i, j+1)
            + array[1:, :-1]  # (i+1, j)
            + array[1:, 1:]  # (i+1, j+1)
        )
        / 4
    )


def prepare_E(E: np.ndarray) -> np.ndarray:
    """
    Compute the evaporation on the secondary grid.
    This is the average of the surrounding four points on the primary grid.

    Args:
        E:
            (N, M) array of evaporation on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.

    Returns:
        (N-1, M-1) array of evaporation on the secondary grid.
    """
    return _primary_to_secondary(E)


def prepare_P(P: np.ndarray) -> np.ndarray:
    """
    Compute the precipitation on the secondary grid.
    This is the average of the surrounding four points on the primary grid.

    Not that this is not normally necessary, as precipitation is not used in the model.
    However, it may be useful to compare input precipitation data with the calculated precipitation data.

    Args:
        P:
            (N, M) array of precipitation on the primary grid.
            N = number of points in longitude.
            M = number of points in latitude.

    Returns:
        (N-1, M-1) array of precipitation on the secondary grid.
    """
    return _primary_to_secondary(P)


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

    Returns:
        (N-1, M-1) array of longitudinal water vapor flux on the secondary grid.
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

    Returns:
        (N-1, M-1) array of longitudinal water vapor flux on the secondary grid.
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

    Returns:
        (N-1, M-1) array of latitudinal water vapor flux on the secondary grid.
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

    Returns:
        (N-1, M-1) array of latitudinal water vapor flux on the secondary grid.
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


def calculate_precipitation(
    Fx_left: np.ndarray,
    Fx_right: np.ndarray,
    Fy_bottom: np.ndarray,
    Fy_top: np.ndarray,
    E: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    We could drive the model using both precipitation and evaporation data.
    However, if there are biases in the input data, this would perturb the equation of state,
    resulting in errors in the recycling ratio.

    Here we disregard the precipitation data,
    and use the equation of state + evaporation data to calculate a consistent precipitation.

    The level of bias in the input data can be estimated by comparing this precipitation
    with the input precipitation data.

    Units: It is important to use consistent units here.
    Preferably scaled units for all variables.

    Inputs and output should have shape (N, M) on the secondary grid.
    N = number of points in longitude.
    M = number of points in latitude.

    Args:
        Fx_left: longitudinal water vapor flux on the left hand side of each cell
        Fx_right: longitudinal water vapor flux on the right hand side of each cell
        Fy_bottom: latitudinal water vapor flux on the bottom side of each cell
        Fy_top: latitudinal water vapor flux on the top side of each cell
        E: evaporation
        dx: longitudinal grid spacing
        dy: latitudinal grid spacing
    """
    return E - (Fx_right - Fx_left) / dx - (Fy_top - Fy_bottom) / dy
