import logging
from datetime import UTC, datetime, timedelta
from typing import Callable, TypedDict

import numpy as np

from .coefficients import Coefficients
from .rotate import rot90, rot90_flux_lrbt
from .utils import (
    check_lr_flux,
    check_tb_flux,
    diagonal,
    drop_first,
    drop_last,
    inflow_mask,
    outflow_mask,
    unbuffer,
)

logger = logging.getLogger(__name__)


class RunStatus(TypedDict):
    success: bool

    # solution for rho on the secondary grid.
    # shape (N, M)
    # N = number of points in longitude.
    # M = number of points in latitude.
    rho: np.ndarray

    # number of iterations
    k: int

    # delta per iteration
    deltas: list[float]

    # time taken
    time_taken: timedelta


def callback(rho: np.ndarray, k: int) -> None:
    """
    Noop callback function.

    Args:
        rho: the auxiliary variable rho, with shape (N, M) on the secondary grid.
            N = number of points in longitude.
            M = number of points in latitude.
        k: the iteration number
    """
    pass


# Fail early condition.
# If the convergence delta exceeds this value,
# we consider the solution to be diverging (running away).
DELTA_DIVERGENCE_THRESHOLD = 1e3


def run(
    Fx_left: np.ndarray,
    Fx_right: np.ndarray,
    Fy_bottom: np.ndarray,
    Fy_top: np.ndarray,
    E: np.ndarray,
    P: np.ndarray,
    dx: float,
    dy: float,
    rho_0: float = 0.0,
    R: float = 0.2,
    R_1: float = 0.2,
    max_iter: int = 1000,
    tol: float = 1e-3,
    divergence_tolerance: float = DELTA_DIVERGENCE_THRESHOLD,
    callback: Callable[[np.ndarray, int], None] = callback,
) -> RunStatus:
    """
    Solve for the auxiliary variable rho of the bulk recycling model.

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
        rho_0: initial guess for the auxiliary variable rho
        R: relaxation parameter within the model domain
        R_1: relaxation parameter external to the model domain.
            used for extrapolation of outflows.
        max_iter: maximum number of iterations
        tol: tolerance for convergence.
            The algorithm halts when the absolute difference between iterations (including relaxation) is less than tol.
        divergence_tolerance: tolerance for divergence (fail early).
        callback: A callback function, with arguments (rho, k).
            May be called multiple times per iteration.

    Raises:
        ValueError: if any of the inputs are not valid
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------------------------------------------------------

    if E.ndim != 2:
        raise ValueError("array must be 2D")

    (N, M) = E.shape

    if N < 2 or M < 2:
        # for boundary conditions to work
        raise ValueError("array too small")

    # check all arrays the same shape
    for arr in [Fx_left, Fx_right, Fy_bottom, Fy_top, E, P]:
        if arr.shape != (N, M):
            raise ValueError("arrays must be the same shape")

    for arr in [Fx_left, Fx_right, Fy_bottom, Fy_top, E, P]:
        if not np.isfinite(arr).all():
            raise ValueError("array values must be finite")

    check_lr_flux(Fx_left, Fx_right)
    check_tb_flux(Fy_top, Fy_bottom)

    # check dx, dy > 0
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")

    # check 0 <= rho_0 < 1
    if not (0 <= rho_0 < 1):
        raise ValueError("rho_0 must be between 0 and 1")

    # check 0 < R, R_1 <= 1
    if not (0 < R <= 1 and 0 < R_1 <= 1):
        raise ValueError("R and R_1 must be between 0 and 1")

    # max_iter > 0
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    # check tol > 0
    if tol <= 0:
        raise ValueError("tol must be positive")

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare for iteration
    # ------------------------------------------------------------------------------------------------------------------

    # Create coefficients object
    coeffs = Coefficients(Fx_left, Fx_right, Fy_bottom, Fy_top, E, P, dx, dy)

    # create mask of grid cells external to the model domain (on buffered secondary grid)
    m_outflow = outflow_mask(Fx_left, Fx_right, Fy_bottom, Fy_top)
    m_inflow = inflow_mask(Fx_left, Fx_right, Fy_bottom, Fy_top)

    # shape of the buffered secondary grid
    (N_buffered, M_buffered) = (N + 2, M + 2)

    # initial guess for rho (on buffered secondary grid)
    rho = np.full((N_buffered, M_buffered), rho_0, dtype=float)
    # in the buffered region external to the model domain, set zeros in place of inflows.
    # this is cosmetic, as these cells are not used.
    rho[m_inflow] = 0.0

    # keep track of the delta change per iteration
    deltas = []

    # loop variables
    k = 0
    iter_start = datetime.now(UTC)
    time_taken = timedelta(0)

    success = False

    # ------------------------------------------------------------------------------------------------------------------
    # Iterate
    # ------------------------------------------------------------------------------------------------------------------

    for k in range(1, max_iter + 1):
        # --------------------------------------------------------------------------------------------------------------
        # Take a step
        # --------------------------------------------------------------------------------------------------------------

        step_start = datetime.now(UTC)

        rho_next = rho.copy()

        # This is a nice debug trick: fill rho_next with NaNs.
        # This ensures that the implementation does not use k+1 data from
        # cells where it does not yet exist.
        # By the end of the iteration, no NaNs should be left.
        # unbuffer(rho_next).fill(np.nan)

        # --------------------------------------------------------------------------------------------------------------
        # Solve for rho within the model domain
        # --------------------------------------------------------------------------------------------------------------

        logger.debug(f"Iteration {k} of {max_iter}. Solving for rho within the model domain")

        # We must solve from the bottom left corner (lon/lat).
        # Iterating by the diagonal is cosmetic if we go cell by cell, but may be faster if vectorized.
        # TODO: try to vectorize the inner loop
        # skip the first and last 2 diagonals on the buffered grid, as these are outside the model domain
        for diag in range(-M_buffered + 3, N_buffered - 2):
            # ----------------------------------------------------------------------------------------------------------
            # Per diagonal
            # ----------------------------------------------------------------------------------------------------------

            logger.debug(f"Iteration {k} of {max_iter}, diagonal {diag}")

            # skip the first and last cell of the diagonal - 
            # these are in the buffer, outside the model domain, and updated later.
            for i, j in drop_first(drop_last(diagonal(N_buffered, M_buffered, diag))):
                # ------------------------------------------------------------------------------------------------------
                # Per cell within the model domain
                # ------------------------------------------------------------------------------------------------------

                logger.debug(f"Iteration {k} of {max_iter}, diagonal {diag}, cell ({i}, {j})")

                A_0 = coeffs.A_0_buffered[i, j]

                A_1 = (
                    coeffs.alpha_1_buffered[i, j]
                    + coeffs.alpha_C_buffered[i, j] * rho[i, j]
                    + coeffs.alpha_U_buffered[i, j] * rho[i, j + 1]
                    + coeffs.alpha_R_buffered[i, j] * rho[i + 1, j]
                    + coeffs.alpha_D_buffered[i, j] * rho_next[i, j - 1]
                    + coeffs.alpha_L_buffered[i, j] * rho_next[i - 1, j]
                )

                # apply the relaxation parameter
                rho_next[i, j] = rho[i, j] + R * (A_1 / A_0 - rho[i, j])

            callback(rho_next, k)

        # --------------------------------------------------------------------------------------------------------------
        # Solve for external outflow cells by extrapolation
        # --------------------------------------------------------------------------------------------------------------

        logger.debug(f"Iteration {k} of {max_iter}. Solving for external outflow cells by extrapolation")

        # left
        i = 0
        for j in range(M_buffered):
            if m_outflow[i, j]:
                rho_extr = 2 * rho_next[i + 1, j] - rho_next[i + 2, j]
                # apply the relaxation parameter
                rho_next[i, j] = rho[i, j] + R_1 * (rho_extr - rho[i, j])

        # right
        i = -1
        for j in range(M_buffered):
            if m_outflow[i, j]:
                rho_extr = 2 * rho_next[i - 1, j] - rho_next[i - 2, j]
                # apply the relaxation parameter
                rho_next[i, j] = rho[i, j] + R_1 * (rho_extr - rho[i, j])

        # bottom
        j = 0
        for i in range(N_buffered):
            if m_outflow[i, j]:
                rho_extr = 2 * rho_next[i, j + 1] - rho_next[i, j + 2]
                # apply the relaxation parameter
                rho_next[i, j] = rho[i, j] + R_1 * (rho_extr - rho[i, j])

        # top
        j = -1
        for i in range(N_buffered):
            if m_outflow[i, j]:
                rho_extr = 2 * rho_next[i, j - 1] - rho_next[i, j - 2]
                # apply the relaxation parameter
                rho_next[i, j] = rho[i, j] + R_1 * (rho_extr - rho[i, j])

        callback(rho_next, k)

        # --------------------------------------------------------------------------------------------------------------
        # Finished step
        # --------------------------------------------------------------------------------------------------------------

        logger.info(
            f"Iteration {k} of {max_iter}. Finished in {(datetime.now(UTC) - step_start).total_seconds()} seconds"
        )

        delta = np.abs(unbuffer(rho_next) - unbuffer(rho)).max()
        deltas.append(delta)
        logger.info(f"Iteration {k} of {max_iter}. delta = {delta}")

        # update rho for next iteration
        rho = rho_next

        if delta < tol:
            logger.info(f"Converged in {k} iterations")
            success = True
            break
        elif delta > divergence_tolerance:
            logger.warning(f"Diverged in {k} iterations")
            break
        elif not np.isfinite(delta):
            logger.warning(f"Diverged (non-finite delta) in {k} iterations")
            break

    else:
        logger.warning(f"Did not converge in {max_iter} iterations")

    # ------------------------------------------------------------------------------------------------------------------
    # Finished iteration
    # ------------------------------------------------------------------------------------------------------------------

    time_taken = datetime.now(UTC) - iter_start
    logger.info(f"Total time: {time_taken}")

    return RunStatus(
        success=success,
        rho=unbuffer(rho),
        k=k,
        deltas=deltas,
        time_taken=time_taken,
    )


def run_4_orientations(
    Fx_left: np.ndarray,
    Fx_right: np.ndarray,
    Fy_bottom: np.ndarray,
    Fy_top: np.ndarray,
    E: np.ndarray,
    P: np.ndarray,
    dx: float,
    dy: float,
    rho_0: float = 0.0,
    R: float = 0.2,
    R_1: float = 0.2,
    max_iter: int = 1000,
    tol: float = 1e-3,
    divergence_tolerance: float = DELTA_DIVERGENCE_THRESHOLD,
    callback: Callable[[np.ndarray, int], None] = callback,
) -> dict[int, RunStatus]:
    """
    Run the solver 4 times, rotating the grid by 90 degrees counter-clockwise each time.

    Returns:
        dict mapping from number of 90 degree rotations to RunStatus.
        The returned data is always in the original orientation.
    """
    # mapping from k to RunStatus
    run_status: dict[int, RunStatus] = {}  

    for k in range(4):
        logger.info(f"Attempt {k + 1} of 4 with rotation of {k * 90} degrees")

        # rotate by 90 degrees counter-clockwise k times
        Fx_left_k, Fx_right_k, Fy_bottom_k, Fy_top_k = rot90_flux_lrbt(
            Fx_left, Fx_right, Fy_bottom, Fy_top, k=k
        )
        E_k = rot90(E, k=k)
        P_k = rot90(P, k=k)
        if k % 2 == 1:
            # swap dx and dy for odd k
            dx_k = dy
            dy_k = dx
        else:
            dx_k = dx
            dy_k = dy

        status = run(
            Fx_left_k,
            Fx_right_k,
            Fy_bottom_k,
            Fy_top_k,
            E_k,
            P_k,
            dx_k,
            dy_k,
            rho_0=rho_0,
            R=R,
            R_1=R_1,
            max_iter=max_iter,
            tol=tol,
            divergence_tolerance=divergence_tolerance,
            callback=callback,
        )

        logger.info(
            f"Attempt finished with success={status['success']} after {status['k']} iterations. "
            f"Final delta={status['deltas'][-1]}."
        )

        # rotate the result (scalar field rho) back to original orientation
        status["rho"] = rot90(status["rho"], k=-k)

        run_status[k] = status

    return run_status
