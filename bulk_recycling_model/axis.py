import numpy as np


class Axis:
    """
    Utility class to handle axis labels (latitude, longitude)
    on the primary and secondary grids.
    """
    def __init__(self, min: float, step: float, n_points: int):
        """
        Args:
            min: Minimum value of the axis on the primary grid.
            step: Step size between points on the axis.
            n_points: Number of points on the primary grid.
        """
        if n_points < 2:
            # If there are less than 2 points, the secondary grid cannot be defined.
            raise ValueError("n_points must be at least 2.")
        self.min = min
        self.step = step
        self.n_points = n_points

    @property
    def primary(self) -> np.ndarray:
        """
        Returns the primary grid axis.
        """
        return np.linspace(self.min, self.min + self.step * (self.n_points - 1), self.n_points)
    
    @property
    def secondary(self) -> np.ndarray:
        """
        Returns the secondary grid axis.
        This has one less point, and is shifted by half a step.
        """
        min = self.min + self.step / 2
        n_points = self.n_points - 1
        return np.linspace(min, min + self.step * (n_points - 1), n_points)
    
    @property
    def secondary_buffered(self) -> np.ndarray:
        """
        Returns the secondary grid axis with a buffer of one step on each side.
        """
        min = self.min - self.step / 2
        n_points = self.n_points + 1
        return np.linspace(min, min + self.step * (n_points - 1), n_points)
