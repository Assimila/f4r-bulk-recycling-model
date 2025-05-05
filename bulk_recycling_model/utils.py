from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt


def buffer(a: np.ndarray, value=np.nan) -> np.ndarray:
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


def outflow_mask(
    Fx_left: np.ndarray,
    Fx_right: np.ndarray,
    Fy_bottom: np.ndarray,
    Fy_top: np.ndarray,
) -> npt.NDArray[np.bool_]:
    """
    Create a mask for cells in the buffered region around the secondary grid.
    These are cells where the water vapor flux flows outwards from the model domain.
    These cells are solved by extrapolation.

    Inputs and outputs should have shape (N, M) on the secondary grid.
    N = number of points in longitude.
    M = number of points in latitude.

    Args:
        Fx_left: longitudinal water vapor flux on the left hand side of each cell
        Fx_right: longitudinal water vapor flux on the right hand side of each cell
        Fy_bottom: latitudinal water vapor flux on the bottom side of each cell
        Fy_top: latitudinal water vapor flux on the top side of each cell

    Returns: boolean array of shape (N+2, M+2)
    """
    (N, M) = Fx_left.shape
    mask = np.full((N + 2, M + 2), False, dtype=bool)
    # this slice cuts off the buffer
    _no_buffer = slice(1, -1)
    # left
    mask[0, _no_buffer] = Fx_left[0, :] < 0
    # right
    mask[-1, _no_buffer] = Fx_right[-1, :] > 0
    # bottom
    mask[_no_buffer, 0] = Fy_bottom[:, 0] < 0
    # top
    mask[_no_buffer, -1] = Fy_top[:, -1] > 0
    return mask


def inflow_mask(
    Fx_left: np.ndarray,
    Fx_right: np.ndarray,
    Fy_bottom: np.ndarray,
    Fy_top: np.ndarray,
) -> npt.NDArray[np.bool_]:
    """
    Inverse of outflow_mask in the buffered region around the secondary grid.
    No water vapor of local origin $w_m$ is present in these cells.

    Inputs and outputs should have shape (N, M) on the secondary grid.
    N = number of points in longitude.
    M = number of points in latitude.

    Args:
        Fx_left: longitudinal water vapor flux on the left hand side of each cell
        Fx_right: longitudinal water vapor flux on the right hand side of each cell
        Fy_bottom: latitudinal water vapor flux on the bottom side of each cell
        Fy_top: latitudinal water vapor flux on the top side of each cell

    Returns: boolean array of shape (N+2, M+2)
    """
    outflow = outflow_mask(Fx_left, Fx_right, Fy_bottom, Fy_top)
    # this slice cuts off the buffer
    _no_buffer = slice(1, -1)
    # using the outflow mask, first set all interior cells to True
    outflow[_no_buffer, _no_buffer] = True
    # then invert
    return ~outflow


def diagonal(
    N: int,
    M: int,
    diagonal: int = 0,
) -> Generator[tuple[int, int]]:
    """
    Interpret (N, M) as a 2D array where:
    N = number of points in longitude.
    M = number of points in latitude.

    Iterate over the diagonal of the 2D array in physical space.

    Diagonal 0 runs from top left corner (0, M-1), (1, M-2), ...

    Diagonal 1 is the first super-diagonal.
    Diagonal 1 runs from (1, M-1), (2, M-2), ...

    Arguments:
        N: number of points in longitude.
        M: number of points in latitude.
        diagonal: diagonal to iterate over.
            0 is the main diagonal.
            range of values is (-M+1, N-1) inclusive.

    Yields:
        (i, j): pairs of indices
    """
    for i in range(N):
        j = M - 1 - i + diagonal
        if 0 <= j < M:
            yield i, j


def drop_first[T](gen: Iterable[T]) -> Generator[T]:
    """
    Drop the first element
    """
    first = True
    for item in gen:
        if first:
            first = False
            continue
        yield item


def drop_last[T](gen: Iterable[T]) -> Generator[T]:
    """
    Drop the last element
    """
    prev = None
    for item in gen:
        if prev is not None:
            yield prev
        prev = item
