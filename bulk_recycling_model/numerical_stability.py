import numpy as np


def identify_hot_pixel(instability_heuristic: np.ndarray) -> tuple[int, int]:
    """
    Identify the "hottest" pixel in the instability heuristic.

    Returns: tuple of indices (longitude, latitude).
    """
    i, j = np.unravel_index(np.argmax(instability_heuristic), instability_heuristic.shape)
    return int(i), int(j)


def smooth_hot_pixel(
    E: np.ndarray,
    i_hot: int,
    j_hot: int,
    weight: float = 0.0,
    gaussian_loc: float = 0.0,
    gaussian_scale: float = 0.0,
) -> np.ndarray:
    """
    Given the indices of a hot pixel,
    apply a smoothing operation to the evaporation field E.

    We observe that the hot pixel often has a low evaporation E, 
    and thus a negative precipitation P.

    This operation is local to a 3x3 grid, centred on the hot pixel.
    It should move the value of the hot pixel closer to the 3x3 mean,
    whilst conserving total evaporation.

    Optionally adds some gaussian noise to the smoothing operation,
    which modifies the change applied to the hot pixel.

    Inputs and output should have shape (N, M).
    N = number of points in longitude.
    M = number of points in latitude.

    Arguments:
        E: Evaporation
        i_hot: index in longitude of the hot pixel
        j_hot: index in latitude of the hot pixel
        weight: smoothing strength.
            if weight = 0, no smoothing is applied.
            if weight = 1, the value of the hot pixel is set to the local 3x3 mean.
        gaussian_loc: Mean of optional gaussian noise to add to the smoothing.
        gaussian_scale: Standard deviation of optional gaussian noise to add to the smoothing.

    Returns: smoothed evaporation field
    """
    if not 0 <= weight <= 1:
        raise ValueError("weight must be in [0, 1]")
    
    if gaussian_scale < 0.0:
        raise ValueError("gaussian_scale must be non-negative")

    E = E.copy()
    N, M = E.shape

    # this is how we redistribute evaporation after adjusting the hot pixel
    kernel = np.array(
        [
            [0.05, 0.2, 0.05],
            [0.2, 0, 0.2],
            [0.05, 0.2, 0.05],
        ]
    )

    hot_pixel = E[i_hot, j_hot]
    redistribution = np.zeros_like(E)  # full size kernel
    total = 0.0
    n = 0

    # ii, jj in kernel space
    for ii in range(3):
        for jj in range(3):
            # absolute indices in the full size E and redistribution arrays
            i = i_hot + ii - 1
            j = j_hot + jj - 1
            # handle boundaries
            # if this is a real cell...
            if 0 <= i < N and 0 <= j < M:
                redistribution[i, j] = kernel[ii, jj]
                total += E[i, j]
                n += 1

    mean = total / n

    # delta is the amount of evaporation that we will shift around
    # new value for the hot pixel = value + delta
    delta = (mean - hot_pixel) * weight

    if gaussian_loc != 0.0 or gaussian_scale > 0.0:
        # optionally add some gaussian noise
        delta += np.random.normal(gaussian_loc, gaussian_scale)

    # adjust hot pixel
    E[i_hot, j_hot] = hot_pixel + delta

    # use normalised redistribution array to adjust neighbours accordingly
    E = E - redistribution * delta / redistribution.sum()

    return E
