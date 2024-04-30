# Will be trying to piece together Luc's code which pretty much has everything but object avoidance 

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 0.05  # Time step for synchronous mode

# Section to set up similuator (prob torcs)----------

# Section to generate waypoints to follow -----------
# currently based on spawn point locatio in carla so will change

# MPC paramaters 
params = {
    'L': 2.875  # Wheelbase of the vehicle. Source : https://www.tesla.com/ownersmanual/model3/en_us/GUID-56562137-FC31-4110-A13C-9A9FC6657BF0.html
}
T = 2.0  # Prediction horizon in seconds
N = int(T / TIME_STEP)  # Prediction horizon in time steps
dt = TIME_STEP  # Time step for discretization
state_dim = 4  # Dimension of the state [x, y, theta, v]
control_dim = 2  # Dimension of the control input [steering angle, acceleration]

# Initialize Opti object
opti = ca.Opti()

# Declare variables
X = opti.variable(state_dim, N + 1)  # state trajectory variables over prediction horizon
U = opti.variable(control_dim, N)  # control trajectory variables over prediction horizon
P = opti.parameter(state_dim)  # initial state parameter
Q_base = ca.MX.eye(state_dim)  # Base state penalty matrix (emphasizes position states)
weight_increase_factor = 1.00  # Increase factor for each step in the prediction horizon
R = ca.MX.eye(control_dim)  # control penalty matrix for objective function
W = opti.parameter(2, N)  # Reference trajectory parameter

# Objective
obj = 0
for k in range(N):
    Q = Q_base * (weight_increase_factor ** k)  # Increase weight for each step in the prediction horizon

    x_k = X[:, k]  # Current state
    u_k = U[:, k]  # Current control input
    x_next = X[:, k + 1]  # Next state

    x_ref = ca.vertcat(W[:, k],
                       ca.MX.zeros(state_dim - 2, 1))  # Reference state with waypoint and zero for other states

    dx = x_k - x_ref  # Deviation of state from reference state
    du = u_k  # Control input deviation (assuming a desired control input of zero)

    # Quadratic cost with reference state and control input
    obj += ca.mtimes([ca.mtimes(dx.T, Q), dx]) + ca.mtimes(
        [ca.mtimes(du.T, R), du])  # Minimize quadratic cost and deviation from reference state

opti.minimize(obj)




