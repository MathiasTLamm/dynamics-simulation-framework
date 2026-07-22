
import numpy as np
from dynamics.base import DynamicSystem

class SinglePendulum(DynamicSystem):
    """
    A simple single pendulum: a point mass on a massless rigid rod, 
    swinging under gravity with no damping or driving forces.

    Parameters
    ----------
    m : float
        Mass of the pendulum bob [kg]
    L : float
        Length of the rod [m]
    g : float, optional
        Gravtational acceleration [m/s^2]. Defaults to 9.81.

    Attributes
    ----------
    m : float
        Stored mass.
    L : float
        Stored length.
    g : float
        Stored gravitational acceleration.
    """

    def __init__(self, m: float, L: float, g=9.81):
        self.m = m
        self.L = L
        self.g = g

    def derivatives(self, t, y):
        """
        Computes the time derivatives of the pendulum's state.
        
        The Equations of Motion (EoM) are found based on the Lagrange method
        and rewritten as a 1st order system of differential equations.

        Parameters
        ----------
        t : float
            Current time [s] (unused - the system is autonomous).
        y : np.ndarray
            Currrent state, (theta, omega) given in [rad, rad/s].
            
        Returns
        -------
        np.ndarray
            Derivative of the state, (theta_dot, omega_dot) given in [rad/s, rad/s^2].
        """

        L = self.L
        g = self.g

        theta, omega = y
        theta_dot = omega
        omega_dot = -(g / L) * np.sin(theta)

        y_dot = np.array([theta_dot, omega_dot])
        return y_dot
    
    def state_labels(self):
        """
        Returns the names of the state variables in the same order as the state vector y.

        Returns
        -------
        labels : list of str
            State labels (theta, omega)

        """

        labels = ["theta", "omega"]
        return labels
        