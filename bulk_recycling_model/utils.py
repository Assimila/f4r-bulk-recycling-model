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
