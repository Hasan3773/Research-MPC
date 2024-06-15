from scipy.interpolate import CubicHermiteSpline
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import boxconstraint as bx
import math
import spline_dist

class ClassMPC:

    def __init__(self, full_spline):
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
        self.opti = ca.Opti()

        # Declare variables
        self.X = self.opti.variable(state_dim, N + 1)  # state trajectory variables over prediction horizon
        self.U = self.opti.variable(control_dim, N)  # control trajectory variables over prediction horizon
        self.P = self.opti.parameter(state_dim)  # initial state parameter
        self.Q_base = ca.MX.eye(state_dim)  # Base state penalty matrix (emphasizes position states)
        weight_increase_factor = 1.00  # Increase factor for each step in the prediction horizon
        self.R = ca.MX.eye(control_dim)  # control penalty matrix for objective function
        self.W = self.opti.parameter(2, 1)  # Reference trajectory parameter
        self.A , self.B = get_a_and_b_from_full_spline(full_spline) # list of start and end points

        # Objective
        obj = 0
        for k in range(N):
            Q = self.Q_base * (weight_increase_factor ** k)  # Increase weight for each step in the prediction horizon

            x_k = self.X[:, k]  # Current state
            u_k = self.U[:, k]  # Current control input
            x_next = self.X[:, k + 1]  # Next state
            dx = spline_dist.min_lineseg_dist(x_k[:2], self.A, self.B)
            
            du = u_k  # Control input deviation (assuming a desired control input of zero)

            # Quadratic cost with reference state and control input
            obj += dx + ca.mtimes(
                [ca.mtimes(du.T, self.R), du])  # Minimize quadratic cost and deviation from reference state

        self.opti.minimize(obj)

        # actual values for metadrive car
        max_steering_angle_deg = 60
        max_steering_angle_rad = 60 * ca.pi / 180

        # Dynamics (Euler discretization using bicycle model)
        for k in range(N):
            steering_angle_rad = self.U[0, k] * max_steering_angle_rad  # Convert normalized steering angle to radians

            self.opti.subject_to(self.X[:, k + 1] == self.X[:, k] + dt * ca.vertcat(
                self.X[3, k] * ca.cos(self.X[2, k]),
                self.X[3, k] * ca.sin(self.X[2, k]),
                (self.X[3, k] / params['L']) * ca.tan(steering_angle_rad),
                self.U[1, k]
            ))

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.P)  # Initial state constraint

        # Input constraints
        steering_angle_bounds = [-1.0, 1.0]
        acceleration_bounds = [-1.0, 1.0]
        lb = np.array([steering_angle_bounds[0], acceleration_bounds[0]]).reshape(-1, 1)
        ub = np.array([steering_angle_bounds[1], acceleration_bounds[1]]).reshape(-1, 1)
        action_space = bx.BoxConstraint(lb=lb, ub=ub)

        # Apply constraints to optimization problem
        for i in range(N):
            # Input constraints
            self.opti.subject_to(action_space.H_np @ self.U[:, i] <= action_space.b_np)

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
                "ipopt.print_level": 0}
        self.opti.solver('ipopt', opts)

        # Array to store closed-loop trajectory states (X and Y coordinates)
        closed_loop_data = []
        open_loop_data = []
        residuals_data = []

        # Initialize warm-start parameters
        self.prev_sol_x = None
        self.prev_sol_u = None


    def generate_spline(vehicle):
        # Distance between waypoints
        DISTANCE = 2

        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
        long_pos = current_lane.local_coordinates(vehicle.position)[0]
        lane_theta = current_lane.heading_theta_at(long_pos)
        waypoint_x = vehicle.position[0] + np.cos(lane_theta) * DISTANCE *2
        waypoint_y = vehicle.position[1] + np.sin(lane_theta) * DISTANCE *2
        return ca.vertcat(waypoint_x, waypoint_y), [[waypoint_x, waypoint_y]]

    def make_line(x_offset, height, y_dir=1, color=(1,105/255,180/255)):
        points = [(x*y_dir,x_offset+x,height*x/10+height) for x in range(10)]
        colors = [np.clip(np.array([*color,1])*(i+1)/11, 0., 1.0) for i in range(10)]
        if y_dir<0:
            points = points[::-1]
            colors = colors[::-1]
        return points, colors


    # cubic_spline references W symbol & position references P symbol
    def find_action(self, vehicle, drawer=None):

        waypoints, waypoint_arr = ClassMPC.generate_spline(vehicle)
        if drawer is not None:
            draw_waypoint(drawer, waypoint_arr)

        ClassMPC.get_splines_to_destination(vehicle, drawer)

        #  Fetch initial state from MetaDrive
        x0 = vehicle.position[0]
        y0 = vehicle.position[1]
        theta0 = np.arctan2(vehicle.heading[1], vehicle.heading[0])
        v0 = ca.sqrt(vehicle.velocity[0] ** 2 + vehicle.velocity[1] ** 2)

        print("Current x: ", x0)
        print("Current y: ", y0)
        print("Current theta: ", theta0)
        print("Current velocity: ", v0)
        
        # Store current state in the closed-loop trajectory data
        # if self.i > 0:
        # self.closed_loop_data.append([x0, y0, theta0, v0])

        # Set initial state for optimization problem
        initial_state = ca.vertcat(x0, y0, theta0, v0)
        self.opti.set_value(self.P, initial_state)

        # Set the reference trajectory for the current iteration
        self.opti.set_value(self.W, waypoints)

        if self.prev_sol_x is not None and self.prev_sol_u is not None:
            # Warm-starting the solver with the previous solution
            self.opti.set_initial(self.X, self.prev_sol_x)
            self.opti.set_initial(self.U, self.prev_sol_u)

        # Solve the optimization problem
        sol = self.opti.solve()

        # If the solver is successful, apply the first control input to the vehicle
        if sol.stats()['success']:
            u = sol.value(self.U[:, 0])

            # Bound acceleration and steering angle to [-1, 1]
            u[0] = np.clip(u[0], -1, 1)
            u[1] = np.clip(u[1], -1, 1)

            print("Steering angle: ", u[0])
            print("Acceleration: ", u[1])

        # Update previous solution variables for warm-starting next iteration
        prev_sol_x = sol.value(self.X)
        prev_sol_u = sol.value(self.U)

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
        print("Number of checkpoints: ", len(checkpoints))
        for i, checkpoint in enumerate(checkpoints):
            print(f"Checkpoint {i}: {checkpoint}")

        for j in range(len(checkpoints) - 1):
            start_node = checkpoints[j]
            end_node = checkpoints[j + 1]
            lanes = road_network.graph[start_node][end_node]

            # Debugging: Print lane information
            print(f"Extracting lanes from node {start_node} to node {end_node}")

            if vehicle.lane in lanes:
                current_lane = vehicle.lane
            else:
                current_lane = lanes[0]

            spline = current_lane.get_polyline()
            all_splines.append(spline)

            # for lane in lanes:
            #     # Debugging: Print lane index and length
            #     print(f"Extracting spline from lane index {lane.index} with length {lane.length}")
            #     spline = lane.get_polyline()
            #     all_splines.append(spline)

        # Concatenate all splines
        full_spline = np.concatenate(all_splines)

        # Debugging: Print final spline information
        print(f"Final spline has {len(full_spline)} points")

        # Display the spline in MetaDrive
        draw_waypoint(drawer, full_spline)

        print("length of full_spline: ", len(full_spline))
        print("full_spline: ", full_spline)

        return full_spline

def get_a_and_b_from_full_spline(full_spline):
    A = full_spline[0:-2].T
    B = full_spline[1:-1].T
    return A, B

def draw_waypoint(drawer, waypoints, color=None):
    HEIGHT = 0.5
    if color is None:
        color = np.array([0.09090909, 0.03743316, 0.06417112, 0.09090909])
    points = [(int(waypoint[0]), int(waypoint[1]), HEIGHT) for waypoint in waypoints]
    colors = [color for _ in range(len(waypoints))]
    drawer.reset()
    drawer.draw_points(points, colors)
