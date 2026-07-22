
from abc import ABC, abstractmethod

class DynamicSystem(ABC):
    """
    Abstract Base Class (ABC) used to define each dynamic system

    Any given system (e.g. SinglePendulum) must inherit from this class and implement the methods: "derivatives" and "state_labels".
    This lets the solver and analysis function operate on any system without knowing the specific physics.
    """

    @abstractmethod
    def derivatives(self, t, y):
        """
        Calculates the time derivative of the state.

        Parameters
        ------------
        t : float
            Current simulation time [s].
        y : np.ndarray
            Starting state variable array with shape (N_state_var,).

        Returns
        -------
        np.ndarray
            Derivative of the state.
        """

        pass

    @abstractmethod
    def state_labels(self):
        """
        Returns the names of the state variables in the same order as the state vector y.

        Returns
        -------
        labels : list of str
            State labels
        """
        pass

    