# Will be trying to piece together Luc's code which pretty much has everything but object avoidance 

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import math 

from boxconstraint import BoxConstraint

class MPCController:
    def __init__(self, timestep=0.1, horizon=5):


