
from scipy.interpolate import CubicHermiteSpline
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import boxconstraint as bx
import math

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
