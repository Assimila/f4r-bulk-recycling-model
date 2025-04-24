from functools import cached_property

import numpy as np
import numpy.typing as npt

from . import cases


class Coefficients:
    """
    Utility class to compute model constants,
    which only vary by grid cell, not by iteration.
    """

    def __init__(
        self,
        Fx_left: np.ndarray,
        Fx_right: np.ndarray,
        Fy_bottom: np.ndarray,
        Fy_top: np.ndarray,
        E: np.ndarray,
        P: np.ndarray,
        dx: float,
        dy: float,
        classification: npt.NDArray[np.int8] | None = None,
    ):
        """
        Units: all variables must be provided in scaled units.

        Inputs and output should have shape (N, M) on the secondary grid.
        N = number of points in longitude.
        M = number of points in latitude.

        Args:
            Fx_left: longitudinal water vapor flux on the left hand side of each cell
            Fx_right: longitudinal water vapor flux on the right hand side of each cell
            Fy_bottom: latitudinal water vapor flux on the bottom side of each cell
            Fy_top: latitudinal water vapor flux on the top side of each cell
            E: evaporation
            P: precipitation
            dx: longitudinal grid spacing
            dy: latitudinal grid spacing
            classification: precomputed classification of the cells.
                Will be computed if not provided.
        """
        self.Fx_left = Fx_left
        self.Fx_right = Fx_right
        self.Fy_bottom = Fy_bottom
        self.Fy_top = Fy_top
        self.E = E
        self.P = P
        self.dx = dx
        self.dy = dy

        if classification is None:
            classification = cases.classify_cells(Fx_left, Fy_bottom)
        self.classification = classification

    @cached_property
    def A_0(self) -> np.ndarray:
        """
        Compute the constant A_0 across the secondary grid.
        
        This is the denominator of

        $$
        \rho^{k+1}_{i+1/2, j+1/2} = \frac{A_1}{A_0}
        $$
        """
        SW = 2 * self.P * self.dx * self.dy + self.Fx_right * self.dy + self.Fy_top * self.dx
        NW = 2 * self.P * self.dx * self.dy + self.Fx_right * self.dy - self.Fy_bottom * self.dx
        NE = 2 * self.P * self.dx * self.dy - self.Fx_left * self.dy - self.Fy_bottom * self.dx
        SE = 2 * self.P * self.dx * self.dy - self.Fx_left * self.dy + self.Fy_top * self.dx

        A_0 = np.zeros_like(self.P, dtype=float)

        A_0[self.classification == cases.Wind.SW] = SW[self.classification == cases.Wind.SW]
        A_0[self.classification == cases.Wind.NW] = NW[self.classification == cases.Wind.NW]
        A_0[self.classification == cases.Wind.NE] = NE[self.classification == cases.Wind.NE]
        A_0[self.classification == cases.Wind.SE] = SE[self.classification == cases.Wind.SE]

        if np.any(np.isclose(A_0, 0)):
            raise ZeroDivisionError("A0 contains values close to zero, which may lead to division by zero.")

        return A_0
    
    @cached_property
    def alpha_1(self) -> np.ndarray:
        """
        Compute the constant alpha_1 across the secondary grid.
        
        This is a component of the numerator A_1 of

        $$
        \rho^{k+1}_{i+1/2, j+1/2} = \frac{A_1}{A_0}
        $$
        """
        return 2 * self.E * self.dx * self.dy

    @cached_property
    def alpha_C(self) -> np.ndarray:
        """
        Compute the constant alpha_C across the secondary grid.
        
        This is a component of the numerator A_1 of

        $$
        \rho^{k+1}_{i+1/2, j+1/2} = \frac{A_1}{A_0}
        $$
        """
        SW = self.Fx_left * self.dy + self.Fy_bottom * self.dx
        NW = self.Fx_left * self.dy - self.Fy_top * self.dx
        NE = -self.Fx_right * self.dy - self.Fy_top * self.dx
        SE = -self.Fx_right * self.dy + self.Fy_bottom * self.dx

        alpha_C = np.zeros_like(self.P, dtype=float)

        alpha_C[self.classification == cases.Wind.SW] = SW[self.classification == cases.Wind.SW]
        alpha_C[self.classification == cases.Wind.NW] = NW[self.classification == cases.Wind.NW]
        alpha_C[self.classification == cases.Wind.NE] = NE[self.classification == cases.Wind.NE]
        alpha_C[self.classification == cases.Wind.SE] = SE[self.classification == cases.Wind.SE]

        return alpha_C
    
    @cached_property
    def alpha_U(self) -> np.ndarray:
        """
        Compute the constant alpha_U across the secondary grid.
        
        This is a component of the numerator A_1 of

        $$
        \rho^{k+1}_{i+1/2, j+1/2} = \frac{A_1}{A_0}
        $$
        """
        return -self.Fy_top * self.dx

    @cached_property
    def alpha_R(self) -> np.ndarray:
        """
        Compute the constant alpha_R across the secondary grid.
        
        This is a component of the numerator A_1 of

        $$
        \rho^{k+1}_{i+1/2, j+1/2} = \frac{A_1}{A_0}
        $$
        """
        return -self.Fx_right * self.dy
    
    @cached_property
    def alpha_D(self) -> np.ndarray:
        """
        Compute the constant alpha_D across the secondary grid.
        
        This is a component of the numerator A_1 of

        $$
        \rho^{k+1}_{i+1/2, j+1/2} = \frac{A_1}{A_0}
        $$
        """
        return self.Fy_bottom * self.dx
    
    @cached_property
    def alpha_L(self) -> np.ndarray:
        """
        Compute the constant alpha_L across the secondary grid.
        
        This is a component of the numerator A_1 of

        $$
        \rho^{k+1}_{i+1/2, j+1/2} = \frac{A_1}{A_0}
        $$
        """
        return self.Fx_left * self.dy
