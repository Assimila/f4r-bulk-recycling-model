import numpy as np


def buffer(a: np.ndarray, value = np.nan) -> np.ndarray:
    """
    Add a buffer of some constant value around the array.
    """
    if np.isnan(value) and not np.issubdtype(a.dtype, np.floating):
        raise ValueError(f"{a.dtype} does not support NaN")
    padded_shape = tuple(dim + 2 for dim in a.shape)
    padded_array = np.full(padded_shape, value, dtype=a.dtype)
    slices = tuple(slice(1, -1) for _ in a.shape)
    padded_array[slices] = a
    return padded_array


def unbuffer(a: np.ndarray) -> np.ndarray:
    """
    Remove a buffer of 1 cell from around the array.

    Returns: a view of the original array with the buffer removed.
    """
    if any(dim < 3 for dim in a.shape):
        raise ValueError("All dimensions must have size >= 3 to unbuffer.")
    slices = tuple(slice(1, -1) for _ in a.shape)
    return a[slices]


def check_lr_flux(
    Fx_left: np.ndarray,
    Fx_right: np.ndarray,
) -> None:
    """
    Check that the fluxes are consistent across the grid.
    
    The x flux on the right hand side of each cell should be equal to the x flux
    on the left hand side of the next cell.

    Inputs and output should have shape (N, M) on the secondary grid.
    N = number of points in longitude.
    M = number of points in latitude.

    Args:
        Fx_left: longitudinal water vapor flux on the left hand side of each cell
        Fx_right: longitudinal water vapor flux on the right hand side of each cell

    Raises:
        ValueError: if the fluxes are not consistent
    """
    if not np.all(Fx_right[:-1, :] == Fx_left[1:, :]):
        raise ValueError("Inconsistent Fx")


def check_tb_flux(
    Fy_top: np.ndarray,
    Fy_bottom: np.ndarray,
) -> None:
    """
    Check that the fluxes are consistent across the grid.
    
    The y flux on the top side of each cell should be equal to the y flux
    on the bottom side of the next cell.

    Inputs and output should have shape (N, M) on the secondary grid.
    N = number of points in longitude.
    M = number of points in latitude.

    Args:
        Fy_top: latitudinal water vapor flux on the top side of each cell
        Fy_bottom: latitudinal water vapor flux on the bottom side of each cell

    Raises:
        ValueError: if the fluxes are not consistent
    """
    if not np.all(Fy_top[:, :-1] == Fy_bottom[:, 1:]):
        raise ValueError("Inconsistent Fy")
