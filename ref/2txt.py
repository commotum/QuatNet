import numpy as np
from itertools import product   # cartesian product, not permutations

def quaternion_encoding(P, d):
    """
    Map a 3D integer lattice point P to a unit quaternion.
    
    Args:
        P (tuple of ints): The (x, y, z) lattice coordinates.
        d (int): Even edge length of the cube.
        
    Returns:
        np.ndarray: Quaternion [w, x, y, z].
    """
    x, y, z = P
    h = d / 2
    R = h * np.sqrt(3)
    
    # Radial distance
    rho = np.linalg.norm(P)
    # Normalized radius
    eta = rho / R if R != 0 else 0.0
    # Angle mapping
    theta = eta * 2 * np.pi
    
    # Unit axis
    if rho > 0:
        V = np.array(P) / rho
    else:
        V = np.array([1.0, 0.0, 0.0])
    
    # Quaternion components
    w = np.cos(theta / 4)
    s = np.sin(theta / 4)
    x_q, y_q, z_q = s * V
    
    return np.array([w, x_q, y_q, z_q])

# Define a formatting function and demonstrate on sample points

def format_quaternion(q):
    """
    Format a quaternion [w, x, y, z] with leading sign and 4 decimal places.
    Returns a string like "[+w.wwww, -x.xxxx, +y.yyyy, -z.zzzz]".
    """
    return "[" + ", ".join(f"{v:+.4f}" for v in q) + "]"

# Define a formatting function for integer 3-vectors with leading sign
def format_vector(v):
    """
    Format a 3-element integer tuple with leading sign and no decimals.
    Returns a string like "[+x, -y, +z]".
    """
    return "[" + ", ".join(f"{int(val):+d}" for val in v) + "]"

# Generate quaternion encodings for all points in a d=4 lattice

def lattice(d):
    h = d//2
    return product(range(-h, h+1), repeat=3)

# parameters
d = 10
outfile = "/home/jake/Developer/QuatNet/ref/D10.txt"

with open(outfile, "w") as f:
    for P in lattice(d):
        q = quaternion_encoding(P, d)
        f.write(format_quaternion(q) + "\n")

len_written = sum(1 for _ in open(outfile))
len_written