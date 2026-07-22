
"""
Tests for the SinglePendulum system

Tests included
--------------
test_stay_at_equilibrium
    Tests if the pendulum stays at equilibrium.

"""

import numpy as np

import dynamics.systems.single_pendulum as SingPend
import dynamics.solver as slv

def test_stay_at_equilibrium():
    """
    The pendulum at rest at theta = 0 should remain at equilibrium after 1 timestep.
    """

    # Defining system parameters
    m = 1.0
    L = 1.0
    t = 0.0
    dt = 0.1

    pendulum = SingPend.SinglePendulum(m, L)
    
    # Sets the starting state to the equilibrium y = [0.0, 0.0]
    y = np.array([0.0, 0.0])

    # Advances one timestep and checks if the system is still at equilibrium
    y_new = slv.euler(pendulum.derivatives, t, y, dt)
    assert np.allclose(y_new, [0.0, 0.0])