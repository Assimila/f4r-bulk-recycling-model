import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.quiver import Quiver

from bulk_recycling_model.axis import Axis


def pcolormesh(ax: Axes, data: np.ndarray, lon: Axis, lat: Axis, **kwargs) -> QuadMesh:
    """
    Create a pcolormesh plot of the given data on the specified axes.

    Args:
        data: 2D array of data to plot of the shape (N, M).
            N = number of points in longitude.
            M = number of points in latitude.
            May be on the primary grid, secondary grid, or buffered secondary grid.
        lon: Axis object for the x-axis (longitude).
        lat: Axis object for the y-axis (latitude).
        **kwargs: Additional keyword arguments to pass to pcolormesh.
    """
    # figure out if we're on the primary, secondary, or buffered secondary grid
    N, M = data.shape
    if N == lon.n_points and M == lat.n_points:
        # primary grid
        x = lon.primary
        y = lat.primary
    elif N == lon.n_points - 1 and M == lat.n_points - 1:
        # secondary grid
        x = lon.secondary
        y = lat.secondary
    elif N == lon.n_points + 1 and M == lat.n_points + 1:
        # buffered secondary grid
        x = lon.secondary_buffered
        y = lat.secondary_buffered
    else:
        raise ValueError("Array shape does not match axes")

    # use indexing="ij" to account for the [lon, lat] indexing
    X, Y = np.meshgrid(x, y, indexing="ij")

    return ax.pcolormesh(X, Y, data, shading="nearest", **kwargs)


def build_uv_fluxes(
    Fx_left: np.ndarray, Fx_right: np.ndarray, Fy_bottom: np.ndarray, Fy_top: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the U and V fluxes suitable for plotting with matplotlib quiver.

    Inputs should have shape (N, M) on the secondary grid.
    N = number of points in longitude.
    M = number of points in latitude.

    For a 1x1 grid with a single cell, returns (u, v) like:
    ```
    (nan, nan)   (0, Fy_top)    (nan, nan)
    (Fx_left, 0) (nan, nan)     (Fx_right, 0)
    (nan, nan)   (0, Fy_bottom) (nan, nan)
    ```
    using [x, y] indexing.

    Args:
        Fx_left: Longitudinal water vapor flux on the left hand side of each cell.
        Fx_right: Longitudinal water vapor flux on the right hand side of each cell.
        Fy_bottom: Latitudinal water vapor flux on the bottom side of each cell.
        Fy_top: Latitudinal water vapor flux on the top side of each cell.

    Returns:
        Tuple of (U, V) with shape (2*N + 1, 2*M + 1).
        This is a half-step grid.
    """
    # Ensure all input arrays have the same shape
    if not (Fx_left.shape == Fx_right.shape == Fy_bottom.shape == Fy_top.shape):
        raise ValueError("All input arrays must have the same shape")

    N, M = Fx_left.shape

    # Initialize U and V arrays with NaNs
    U = np.full((2 * N + 1, 2 * M + 1), np.nan)
    V = np.full((2 * N + 1, 2 * M + 1), np.nan)

    U[0:-1:2, 1::2] = Fx_left
    # fill the right edge
    U[-1, 1::2] = Fx_right[-1, :]
    # assign zeros for top and bottom cell edges where there is no x-flux
    U[1::2, 0::2] = 0

    V[1::2, 0:-1:2] = Fy_bottom
    # fill the top edge
    V[1::2, -1] = Fy_top[:, -1]
    # assign zeros for left and right cell edges where there is no y-flux
    V[0::2, 1::2] = 0

    return U, V


def quiver(
    ax: Axes,
    Fx_left: np.ndarray,
    Fx_right: np.ndarray,
    Fy_bottom: np.ndarray,
    Fy_top: np.ndarray,
    lon: Axis,
    lat: Axis,
    **kwargs,
) -> Quiver:
    """
    Create a quiver plot of water vapor fluxes,
    with one arrow per cell face on the secondary grid.

    Inputs should have shape (N, M) on the secondary grid.
    N = number of points in longitude.
    M = number of points in latitude.

    Args:
        ax: Axes object to plot on.
        Fx_left: Longitudinal water vapor flux on the left hand side of each cell.
        Fx_right: Longitudinal water vapor flux on the right hand side of each cell.
        Fy_bottom: Latitudinal water vapor flux on the bottom side of each cell.
        Fy_top: Latitudinal water vapor flux on the top side of each cell.
        lon: Axis object for the x-axis (longitude).
        lat: Axis object for the y-axis (latitude).
        **kwargs: Additional keyword arguments to pass to quiver.
    """
    U, V = build_uv_fluxes(Fx_left, Fx_right, Fy_bottom, Fy_top)
    # use indexing="ij" to account for the [lon, lat] indexing
    X, Y = np.meshgrid(lon.half_step, lat.half_step, indexing="ij")
    return ax.quiver(X, Y, U, V, **kwargs)
