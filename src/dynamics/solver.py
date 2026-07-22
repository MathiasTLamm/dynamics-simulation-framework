
import numpy as np

def euler(derivatives_fn: callable, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    Advances one timestep with Euler's method

    Requires the Equations of Motion (EoM) to be written as a 1st order system of differential equations.

    Parameters
    ------------
    derivatives_fn : callable
        Function computing the derivatives of the state with signature (t, y) -> np.ndarray.
    t : float
        Current simulation time.
    y : np.ndarray
        Starting state variable array with shape (N_state_var,).
    dt : float
        Time step.

    Returns
    -----------
    y_new : np.array
        Updated state variable array with shape (N_state_var,).
    """

    y_dot = derivatives_fn(t, y)
    y_new = y + dt * y_dot
    return y_new