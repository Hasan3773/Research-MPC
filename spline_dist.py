import numpy as np
import casadi as ca
import metadrive.utils.interpolating_line as il 

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

if __name__ == "__main__":
     
    # Test cases
    a = ca.DM([[1, 2], [3, 4]])
    b = ca.DM([[5, 6], [7, 8]])
    p1 = ca.DM([0, 1])
    p2 = ca.DM([[0, 1], [1, 0]])

    print("\n--- Testing CasADI ---")
    # Single point test
    print("Single point test")
    min_dists_single = min_lineseg_dist(p1, a, b)
    print("Min distances (single point):\n", min_dists_single)

    # Multiple points test
    print("Multiple points test")
    min_dists_multiple = min_lineseg_dist(p2, a, b)
    print("Min distances (multiple points):\n", min_dists_multiple)

    # Test cases
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    p1 = np.array([0, 1])
    p2 = np.array([[0, 1], [1, 0]])
    
    print("\n--- Testing numpy1 ---")
    # Single point test
    print("Single point test")
    min_dists_single = min_lineseg_dist_np(p1, a, b)
    print("Min distances (single point):\n", min_dists_single)
    
    # Multiple points test
    print("Multiple points test")
    min_dists_multiple = min_lineseg_dist_np(p2, a, b)
    print("Min distances (multiple points):\n", min_dists_multiple)    

    # Test cases
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    p1 = np.array([0, 1])
    p2 = np.array([[0, 1], [1, 0]])
    print("\n--- Testing numpy2 ---")

    # Single point test
    print("Single point test")
    min_dists_single = min_lineseg_dist_np2(p1, a, b)
    print("Min distances (single point):\n", min_dists_single)

    # Multiple points test
    print("Multiple points test")
    min_dists_multiple = min_lineseg_dist_np2(p2, a, b)
    print("Min distances (multiple points):\n", min_dists_multiple)