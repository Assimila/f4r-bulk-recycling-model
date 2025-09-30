"""
The bulk recycling model is an implicit finite-difference model.
The implicit dependency is only left &/ down.
Therefore the system of equations can be solved from the bottom left corner of the grid to the top right corner.

Sometimes we encounter numerical instability, where the solution does not converge.

The functions in this module rotate the grid by 90 degrees
to test if the instability is due to the order in which the grid is solved.
"""

import numpy as np


def rot90(arr: np.ndarray, k:int = 1) -> np.ndarray:
    """
    Rotate a 2D array by 90 degrees counter-clockwise.
    
    This function could equivalently be thought of as rotating / re-labeling the axes in a clockwise direction.

    ```
      N         W
    W + E ==> S + N
      S         E
    ```

    Args:
        arr: 2D array with shape (N, M).
            N = number of points in longitude.
            M = number of points in latitude.
        k: Number of times the array is rotated by 90 degrees. May be negative.

    Returns:
        Rotated 2D array
    """
    return np.rot90(arr, k=k)


def rot90_flux(Fx: np.ndarray, Fy: np.ndarray, k:int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate flux arrays by 90 degrees counter-clockwise.
    
    This function could equivalently be thought of as rotating / re-labeling the axes in a clockwise direction.

    ```
      N         W
    W + E ==> S + N
      S         E
    ```

    Args:
        Fx: Longitudinal flux with shape (N, M).
            N = number of points in longitude.
            M = number of points in latitude.
        Fy: Latitudinal flux with shape (N, M).
            N = number of points in longitude.
            M = number of points in latitude.
        k: Number of times the array is rotated by 90 degrees. May be negative.

    Returns:
        Tuple of rotated (Fx, Fy)
    """
    # check shapes are the same
    if Fx.shape != Fy.shape:
        raise ValueError("Flux arrays have different shapes")

    # first rotate the data as if it is scalar
    Fx = rot90(Fx, k=k)
    Fy = rot90(Fy, k=k)

    # now worry about the flux directions
    for _ in range(k % 4):
        # for each 90 degrees clockwise rotation...

        # Fx was previously W -> E
        # is now S -> N
        
        # Fy was previously S -> N
        # is now E -> W
        
        Fx, Fy = -Fy, Fx

    return Fx, Fy


def rot90_flux_lrbt(
    Fx_left: np.ndarray, Fx_right: np.ndarray, Fy_bottom: np.ndarray, Fy_top: np.ndarray, k:int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate flux arrays by 90 degrees counter-clockwise.
    
    This function could equivalently be thought of as rotating / re-labeling the axes in a clockwise direction.

    ```
      N         W
    W + E ==> S + N
      S         E
    ```

    Args:
        Fx_left: Longitudinal flux on the left hand side of each cell.
        Fx_right: Longitudinal flux on the right hand side of each cell.
        Fy_bottom: Latitudinal flux on the bottom side of each cell.
        Fy_top: Latitudinal flux on the top side of each cell.
        k: Number of times the array is rotated by 90 degrees. May be negative.

    Returns:
        Tuple of rotated (Fx_left, Fx_right, Fy_bottom, Fy_top)
    """
    # check shapes are the same
    if not (Fx_left.shape == Fx_right.shape == Fy_bottom.shape == Fy_top.shape):
        raise ValueError("Flux arrays have different shapes")

    # first rotate the data as if it is scalar
    Fx_left = rot90(Fx_left, k=k)
    Fx_right = rot90(Fx_right, k=k)
    Fy_bottom = rot90(Fy_bottom, k=k)
    Fy_top = rot90(Fy_top, k=k)

    # now worry about the flux directions
    for _ in range(k % 4):
        # for each 90 degrees clockwise rotation...

        # Fx_left was previously W -> E
        # is now S -> N 
        # => Fy_bottom

        # Fx_right was previously W -> E
        # is now S -> N 
        # => Fy_top
        
        # Fy_bottom was previously S -> N
        # is now E -> W
        # ==> -Fx_right

        # Fy_top was previously S -> N
        # is now E -> W
        # ==> -Fx_left

        Fx_left, Fx_right, Fy_bottom, Fy_top = -Fy_top, -Fy_bottom, Fx_left, Fx_right

    return Fx_left, Fx_right, Fy_bottom, Fy_top
