from enum import IntEnum

import numpy as np
import numpy.typing as npt


class Wind(IntEnum):
    """
    These are the 4 cases for the wind direction.
    Implication is "wind from the ...".

    For each grid cell, the wind case determines the form of the iterative scheme.
    """
    SW = 1
    NW = 2
    NE = 3
    SE = 4


def classify_cells(Fx_left: np.ndarray, Fy_bottom: np.ndarray) -> npt.NDArray[np.int8]:
    """
    classify each grid cell.

    Args:
        Fx_left:
            (N, M) array of longitudinal water vapor flux on the left hand side of each cell.
            N = number of points in longitude.
            M = number of points in latitude.

        Fy_bottom:
            (N, M) array of latitudinal water vapor flux on the bottom of each cell.
            N = number of points in longitude.
            M = number of points in latitude.
    """
    # wind x from the west
    W = Fx_left >= 0
    # wind y from the south
    S = Fy_bottom >= 0

    arr = (S & W) * Wind.SW + (~S & W) * Wind.NW + (~S & ~W) * Wind.NE + (S & ~W) * Wind.SE
    return arr.astype(np.int8)
