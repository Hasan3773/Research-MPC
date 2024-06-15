from scipy.interpolate import CubicHermiteSpline
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import boxconstraint as bx
import math
import spline_dist
import get_spline

# Parameters
params = {
    'L': (1.05234 + 1.4166)/2.0 
}
# define arbatrary end objective
TIME_STEP = 0.05  # Time step
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
W = opti.parameter(2, 1)  # Reference trajectory parameter
A = None # list of start points
B = None # list of end points
x_ref = None

# Objective
obj = 0
for k in range(N):
    Q = Q_base * (weight_increase_factor ** k)  # Increase weight for each step in the prediction horizon

    x_k = X[:, k]  # Current state
    u_k = U[:, k]  # Current control input
    x_next = X[:, k + 1]  # Next state

    x_ref = ca.vertcat(W,
                        0, 1)  # Reference state with waypoint and zero for other states
    
    # get reference spline 
    x_ref = 

    # convert x_ref to A & B here or in lucs function

    dx = x_k - x_ref  # Deviation of state from reference state (get rid of)

    dx = spline_dist.min_lineseg_dist(x_k, A, B)

    du = u_k  # Control input deviation (assuming a desired control input of zero)

    # Quadratic cost with reference state and control input
    obj += ca.mtimes([ca.mtimes(dx.T, Q), dx]) + ca.mtimes(
        [ca.mtimes(du.T, R), du])  # Minimize quadratic cost and deviation from reference state

opti.minimize(obj)

# actual values for metadrive car
max_steering_angle_deg = 60
max_steering_angle_rad = 60 * ca.pi / 180

# Dynamics (Euler discretization using bicycle model)
for k in range(N):
    steering_angle_rad = U[0, k] * max_steering_angle_rad  # Convert normalized steering angle to radians

    opti.subject_to(X[:, k + 1] == X[:, k] + dt * ca.vertcat(
        X[3, k] * ca.cos(X[2, k]),
        X[3, k] * ca.sin(X[2, k]),
        (X[3, k] / params['L']) * ca.tan(steering_angle_rad),
        U[1, k]
    ))

# Constraints
opti.subject_to(X[:, 0] == P)  # Initial state constraint

# Input constraints
steering_angle_bounds = [-1.0, 1.0]
acceleration_bounds = [-1.0, 1.0]
lb = np.array([steering_angle_bounds[0], acceleration_bounds[0]]).reshape(-1, 1)
ub = np.array([steering_angle_bounds[1], acceleration_bounds[1]]).reshape(-1, 1)
action_space = bx.BoxConstraint(lb=lb, ub=ub)

# Apply constraints to optimization problem
for i in range(N):
    # Input constraints
    opti.subject_to(action_space.H_np @ U[:, i] <= action_space.b_np)

# Setup solver
acceptable_dual_inf_tol = 1e11
acceptable_compl_inf_tol = 1e-3
acceptable_iter = 15
acceptable_constr_viol_tol = 1e-3
acceptable_tol = 1e-6

opts = {"ipopt.acceptable_tol": acceptable_tol,
        "ipopt.acceptable_constr_viol_tol": acceptable_constr_viol_tol,
        "ipopt.acceptable_dual_inf_tol": acceptable_dual_inf_tol,
        "ipopt.acceptable_iter": acceptable_iter,
        "ipopt.acceptable_compl_inf_tol": acceptable_compl_inf_tol,
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": 'yes'}
opti.solver('ipopt', opts)

# Array to store closed-loop trajectory states (X and Y coordinates)
closed_loop_data = []
open_loop_data = []
residuals_data = []

# Initialize warm-start parameters
prev_sol_x = None
prev_sol_u = None

# Extend 2D coords into 3D space
def extendTo3D(arr, extend):
    ret = []
    for e in arr:
        ret.append([e[0], e[1], extend])
    return ret

# Distance between waypoints
DISTANCE = 2
# Height of waypoints above ground
HEIGHT = 0.5
def generate_spline(vehicle):
    if vehicle.lane in vehicle.navigation.current_ref_lanes:
        current_lane = vehicle.lane
    else:
        current_lane = vehicle.navigation.current_ref_lanes[0]
    long_pos = current_lane.local_coordinates(vehicle.position)[0]
    lane_theta = current_lane.heading_theta_at(long_pos)
    waypoint_x = vehicle.position[0] + np.cos(lane_theta) * DISTANCE *2
    waypoint_y = vehicle.position[1] + np.sin(lane_theta) * DISTANCE *2
    print(current_lane.get_polyline())
    return ca.vertcat(waypoint_x, waypoint_y), extendTo3D(current_lane.get_polyline(), HEIGHT)

def make_line(x_offset, height, y_dir=1, color=(1,105/255,180/255)):
    points = [(x*y_dir,x_offset+x,height*x/10+height) for x in range(10)]
    colors = [np.clip(np.array([*color,1])*(i+1)/11, 0., 1.0) for i in range(10)]
    if y_dir<0:
        points = points[::-1]
        colors = colors[::-1]
    return points, colors

def draw_waypoint(drawer, waypoints):
    drawer.reset()
    for k, waypoint in enumerate(waypoints):
        point = [((waypoint[0]), (waypoint[1]), HEIGHT)]
        color = [np.array([0.09090909, 0.03743316, 0.06417112, 0.09090909])]
        drawer.draw_points(point, color)


# cubic_spline references W symbol & position references P symbol
def find_action(vehicle, drawer=None, verbose=False):
    # if not verbose:
    #     def print(*args, **kwargs):
    #         pass
    global prev_sol_x
    global prev_sol_u

    waypoints, waypoint_arr = generate_spline(vehicle)
    if drawer is not None:
        draw_waypoint(drawer, waypoint_arr)

    #  Fetch initial state from MetaDrive
    x0 = vehicle.position[0]
    y0 = vehicle.position[1]
    theta0 = np.arctan2(vehicle.heading[1], vehicle.heading[0])
    v0 = ca.sqrt(vehicle.velocity[0] ** 2 + vehicle.velocity[1] ** 2)

    if verbose:
        print("Current x: ", x0)
        print("Current y: ", y0)
        print("Current theta: ", theta0)
        print("Current velocity: ", v0)
    
    # Store current state in the closed-loop trajectory data
    if i > 0:
        closed_loop_data.append([x0, y0, theta0, v0])

    # Set initial state for optimization problem
    initial_state = ca.vertcat(x0, y0, theta0, v0)
    opti.set_value(P, initial_state)

    # Set the reference trajectory for the current iteration
    opti.set_value(W, waypoints)

    if prev_sol_x is not None and prev_sol_u is not None:
        # Warm-starting the solver with the previous solution
        opti.set_initial(X, prev_sol_x)
        opti.set_initial(U, prev_sol_u)

    # Solve the optimization problem
    sol = opti.solve()

    # If the solver is successful, apply the first control input to the vehicle
    if sol.stats()['success']:
        u = sol.value(U[:, 0])

        # Bound acceleration and steering angle to [-1, 1]
        u[0] = np.clip(u[0], -1, 1)
        u[1] = np.clip(u[1], -1, 1)
        if verbose:
            print("Steering angle: ", u[0])
            print("Acceleration: ", u[1])

    # Update previous solution variables for warm-starting next iteration
    prev_sol_x = sol.value(X)
    prev_sol_u = sol.value(U)

    return [u[0], u[1]]

 