"""
The bulk recycling model solves for the scalar field $\rho$ which has physical bounds of [0, 1].
However, the numerical scheme does not guarantee that the solution remains within these bounds.
And is sometimes unstable, leading to runaway non-convergent solutions.

To mitigate these issues, we apply clamping to the scalar field after each update step.
This is conceptually similar to a penalty term, or restorative force,
that pushes the solution back into the physical bounds.

This correction should be smooth and differentiable to avoid introducing numerical artifacts.
The map from un-clamped to clamped values should be monotonic and approach the identity function
within the physical bounds, to minimize distortion of valid solutions.
"""
import numpy as np


def smooth(deviation: np.ndarray, tolerance: float = 0.1) -> np.ndarray:
    """
    A smooth function that maps deviations [0, inf) to [0, tolerance).
    It should have value = 0 and first derivative = 1 at deviation = 0.
    Asymptotically approaches tolerance as deviation goes to infinity.
    """
    if np.any(deviation < 0):
        raise ValueError("Deviation must be >= 0")
    # 1st derivative = exp(-deviation / tolerance)
    return tolerance * (1 - np.exp(-deviation / tolerance))


def clamp(x: np.ndarray, *, lower_bound: float = 0.0, upper_bound: float = 1.0, tolerance: float = 0.1) -> np.ndarray:
    """
    Apply a smooth function to clamp values above upper_bound,
    asymptotically approaching an absolution maximum of upper_bound + tolerance.

    And similarly, clamp values below lower_bound,
    asymptotically approaching an absolute minimum of lower_bound - tolerance.

    Values in the range [lower_bound, upper_bound] are unchanged.
    """
    out = np.copy(x)

    deviation_upper = x - upper_bound
    mask_upper = deviation_upper > 0
    out[mask_upper] = upper_bound + smooth(deviation_upper[mask_upper], tolerance)

    deviation_lower = lower_bound - x
    mask_lower = deviation_lower > 0
    out[mask_lower] = lower_bound - smooth(deviation_lower[mask_lower], tolerance)

    return out
