from scipy.interpolate import CubicHermiteSpline
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import boxconstraint as bx
import math

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

# Objective
obj = 0
for k in range(N):
    Q = Q_base * (weight_increase_factor ** k)  # Increase weight for each step in the prediction horizon

    x_k = X[:, k]  # Current state
    u_k = U[:, k]  # Current control input
    x_next = X[:, k + 1]  # Next state

    x_ref = ca.vertcat(W,
                        0, 1)  # Reference state with waypoint and zero for other states
    
    # find deviation from reference spline
    x_ref = 


    dx = x_k - x_ref  # Deviation of state from reference state
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


def get_splines_to_destination(vehicle, drawer):
    """
    Extract the entirety of splines from current position to end state.

    Args:
        vehicle: The vehicle object which contains navigation information, including the road network and checkpoints.
        drawer: The drawer object used to visualize waypoints.

    Returns: full_spline: A numpy array representing the concatenated spline of the entire trajectory from the
    current position to the destination.
    """
    all_splines = []
    node_network_navigation = vehicle.navigation
    road_network = node_network_navigation.map.road_network
    checkpoints = node_network_navigation.checkpoints

    # Debugging: Print checkpoints
    print("Extracting splines from the following checkpoints:")
    for i, checkpoint in enumerate(checkpoints):
        print(f"Checkpoint {i}: {checkpoint}")

    for j in range(len(checkpoints) - 1):
        start_node = checkpoints[j]
        end_node = checkpoints[j + 1]
        lanes = road_network.graph[start_node][end_node]

        # Debugging: Print lane information
        print(f"Extracting lanes from node {start_node} to node {end_node}")

        for lane in lanes:
            # Debugging: Print lane index and length
            print(f"Extracting spline from lane index {lane.index} with length {lane.length}")
            spline = lane.get_polyline()
            all_splines.append(spline)

    # Concatenate all splines
    full_spline = np.concatenate(all_splines)

    # Debugging: Print final spline information
    print(f"Final spline has {len(full_spline)} points")

    # Display the spline in MetaDrive
    for waypoint in full_spline:
        draw_waypoint(drawer, waypoint)

    return full_spline

def min_lineseg_dist(p, a, b, d_ba=None):
    """Cartesian distance from point to line segment
    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # Convert inputs to CasADi DM if not already
    p = ca.DM(p) if not isinstance(p, (ca.DM, ca.MX)) else p
    a = ca.DM(a) if not isinstance(a, (ca.DM, ca.MX)) else a
    b = ca.DM(b) if not isinstance(b, (ca.DM, ca.MX)) else b    

    # normalized tangent vectors
    if d_ba is None:
        d_ba = b - a
    d_ba = ca.DM(d_ba)

    # Compute the norm (hypotenuse) of each row in d_ba
    norm = ca.sqrt(ca.sum1(d_ba**2))
    
    # Normalize the vectors
    norm = ca.reshape(norm, d_ba.size1(), 1)
    d = d_ba / norm

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = ca.sum1((a - p) * d)
    t = ca.sum1((p - b) * d)

    # clamped parallel distance
    zeros_vec = ca.DM.zeros(s.size())
    h = ca.fmax(ca.fmax(s, t), zeros_vec)

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    min_dists = ca.sqrt(h**2 + ca.transpose(c)**2)

    return min_dists

# Test cases
a = ca.DM([[1, 2], [3, 4]])
b = ca.DM([[5, 6], [7, 8]])
p1 = ca.DM([0, 1])
p2 = ca.DM([[0, 1], [1, 0]])

# Single point test
print("Single point test")
min_dists_single = min_lineseg_dist(p1, a, b)
print("Min distances (single point):\n", min_dists_single)

# Multiple points test
print("\nMultiple points test")
min_dists_multiple = min_lineseg_dist(p2, a, b)
print("Min distances (multiple points):\n", min_dists_multiple)


# numpy code to compare against:
def min_lineseg_dist_np(p, a, b, d_ba=None):
        """Cartesian distance from point to line segment
        Edited to support arguments as series, from:
        https://stackoverflow.com/a/54442561/11208892

        Args:
            - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
            - a: np.array of shape (x, 2)
            - b: np.array of shape (x, 2)
        """
        # normalized tangent vectors
        p = np.asarray(p)
        if d_ba is None:
            d_ba = b - a
        d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

        # signed parallel distance components
        # rowwise dot products of 2D vectors
        s = np.multiply(a - p, d).sum(axis=1)
        t = np.multiply(p - b, d).sum(axis=1)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(s))])

        # perpendicular distance component
        # rowwise cross products of 2D vectors
        d_pa = p - a
        c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
        min_dists = np.hypot(h, c)
        return min_dists

# Test cases
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
p1 = np.array([0, 1])
p2 = np.array([[0, 1], [1, 0]])

# Single point test
print("Single point test")
min_dists_single = min_lineseg_dist_np(p1, a, b)
print("Min distances (single point):\n", min_dists_single)

# Multiple points test
print("\nMultiple points test")
min_dists_multiple = min_lineseg_dist_np(p2, a, b)
print("Min distances (multiple points):\n", min_dists_multiple)

# numpy code to compare against:
def min_lineseg_dist_np2(p, a, b, d_ba=None):
        """Cartesian distance from point to line segment
        Edited to support arguments as series, from:
        https://stackoverflow.com/a/54442561/11208892

        Args:
            - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
            - a: np.array of shape (x, 2)
            - b: np.array of shape (x, 2)
        """
        # normalized tangent vectors
        p = np.asarray(p)
        if d_ba is None:
            d_ba = b - a
        d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

        # signed parallel distance components
        # rowwise dot products of 2D vectors
        s = np.multiply(a - p, d).sum(axis=1)
        t = np.multiply(p - b, d).sum(axis=1)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(s))])

        # perpendicular distance component
        # rowwise cross products of 2D vectors
        d_pa = p - a
        c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
        min_dists = np.hypot(h, c)

        # Convert final result to CasADi DM
        min_dists = ca.DM(min_dists)

        return min_dists

# Test cases
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
p1 = np.array([0, 1])
p2 = np.array([[0, 1], [1, 0]])

# Single point test
print("Single point test")
min_dists_single = min_lineseg_dist_np2(p1, a, b)
print("Min distances (single point):\n", min_dists_single)

# Multiple points test
print("\nMultiple points test")
min_dists_multiple = min_lineseg_dist_np2(p2, a, b)
print("Min distances (multiple points):\n", min_dists_multiple)