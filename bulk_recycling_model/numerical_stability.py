import numpy as np


def identify_hot_pixel(instability_heuristic: np.ndarray) -> tuple[int, int]:
    """
    Identify the "hottest" pixel in the instability heuristic.

    Returns: tuple of indices (longitude, latitude).
    """
    i, j = np.unravel_index(np.argmax(instability_heuristic), instability_heuristic.shape)
    return int(i), int(j)


def nudge_hot_pixel(
    E: np.ndarray,
    i_hot: int,
    j_hot: int,
    offset: float,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Given the indices of a hot pixel,
    apply a nudge to the evaporation field E.

    new value = old value + offset

    Redistribute the evaporation lost/gained by the hot pixel
    to its immediate neighbours weighted by distance.

    Inputs and output should have shape (N, M).
    N = number of points in longitude.
    M = number of points in latitude.

    Arguments:
        E: Evaporation
        i_hot: index in longitude of the hot pixel
        j_hot: index in latitude of the hot pixel
        offset: amount to adjust the hot pixel by.
        kernel_size: size of the redistribution kernel (must be odd).
            Default is 3, which redistributes within a 3x3 grid.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    
    E = E.copy()
    N, M = E.shape

    # this is how we redistribute evaporation after adjusting the hot pixel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel_center = kernel_size // 2
    for ii in range(kernel_size):
        for jj in range(kernel_size):
            if ii == kernel_center and jj == kernel_center:
                kernel[ii, jj] = 0
            else:
                # 1 / distance
                d2 = (ii - kernel_center) ** 2 + (jj - kernel_center) ** 2
                kernel[ii, jj] = 1.0 / np.sqrt(d2)

    redistribution = np.zeros_like(E, dtype=float)  # full size kernel

    # ii, jj in kernel space
    for ii in range(kernel_size):
        for jj in range(kernel_size):
            # absolute indices in the full size E and redistribution arrays
            i = i_hot + ii - kernel_center
            j = j_hot + jj - kernel_center
            # handle boundaries
            # if this is a real cell...
            if 0 <= i < N and 0 <= j < M:
                redistribution[i, j] = kernel[ii, jj]

    # adjust hot pixel
    E[i_hot, j_hot] = E[i_hot, j_hot] + offset

    # use normalised redistribution array to adjust neighbours accordingly
    E = E - redistribution * offset / redistribution.sum()

    return E
