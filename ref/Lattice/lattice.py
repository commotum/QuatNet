import numpy as np
from itertools import product   # cartesian product, not permutations

def quaternion_encoding(P, r):
    """
    Map a 3D integer lattice point P to a unit quaternion.
    
    Args:
        P (tuple of ints): The (x, y, z) lattice coordinates.
        d (int): Even edge length of the cube.
        
    Returns:
        np.ndarray: Quaternion [w, x, y, z].
    """
    x, y, z = P
    
    # Radial distance
    rho = np.linalg.norm(P)
    # Normalized radius
    eta = rho / r if r != 0 else 0.0
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

def lattice(r):
    """
    Generate all 3D integer lattice points (x, y, z) such that each coordinate
    is in the range [-r, r] inclusive. This forms a cubic lattice region centered
    at the origin.
    
    Args:
        r (int): Radius of lattice i.e. the lattice half-width in each axis (inclusive)
        
    Returns:
        itertools.product: Iterator over all (x,y,z) points in lattice cube
                          with coordinates in range [-r,r]
    """
    return product(range(-r, r+1), repeat=3)

def map_to_unit_interval(vec):
    """
    Maps each value in a 3-vector from [-1.0, 1.0] to [0.0, 1.0] using the rule:
    -1.0 → 0.0,  0.0 → 0.5,  +1.0 → 1.0 (linear mapping)

    Args:
        vec (tuple or list of float): 3-vector with values in [-1.0, 1.0]

    Returns:
        list of float: 3-vector with values in [0.0, 1.0]
    """
    return [(x + 1.0) / 2.0 for x in vec]

def to_rgb_hex(mapped_vec):
    """
    Convert a 3-vector with values in [0.0, 1.0] to an RGB hex color string.
    
    Args:
        mapped_vec (list of float): 3-vector with values in [0.0, 1.0]
        
    Returns:
        str: RGB hex color string in format "#RRGGBB"
    """
    # Convert [0.0, 1.0] values to [0, 255] integers
    rgb = [int(x * 255) for x in mapped_vec]
    
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def to_rgb_tuple(mapped_vec):
    """
    Convert a 3-vector with values in [0.0, 1.0] to an RGB tuple.
    
    Args:
        mapped_vec (list of float): 3-vector with values in [0.0, 1.0]
        
    Returns:
        tuple: RGB values as (r,g,b) integers in range [0,255]
    """
    return tuple(int(x * 255) for x in mapped_vec)



def vector_to_color(P, r):
    """
    Convert a 3D lattice point to both quaternion encoding and RGB color.
    
    Args:
        P (tuple of int): The (x,y,z) lattice coordinates
        r (int): Radius/scale parameter for quaternion encoding
        
    Returns:
        tuple: (quaternion array, RGB hex color string)
    """
    # Get quaternion encoding (already normalized internally)
    q = quaternion_encoding(P, r)
    
    # Map raw P coordinates to color space
    mapped = map_to_unit_interval(P)
    rgb = to_rgb_tuple(mapped) # to_rgb_hex(mapped)
    
    return q, rgb

def print_encoding(P, r):
    """
    Print formatted lattice point, quaternion encoding and RGB color.
    
    Args:
        P (tuple of int): The (x,y,z) lattice coordinates
        r (int): Radius/scale parameter for quaternion encoding
    """
    q, rgb = vector_to_color(P, r)
    print(f"{format_quaternion(q)} {rgb}") # {format_vector(P)} ↦ 


r = 1  # radius of lattice
for P in lattice(r):
    print_encoding(P, r)

print()



# Print encodings for the 8 octants
octants = [
    (+1, +1, +1),  # White   Octant I
    (+1, +1, -1),  # Yellow  Octant II  
    (+1, -1, +1),  # Magenta Octant III
    (+1, -1, -1),  # Red     Octant IV
    (-1, +1, +1),  # Cyan    Octant V
    (-1, +1, -1),  # Green   Octant VI
    (-1, -1, +1),  # Blue    Octant VII
    (-1, -1, -1),  # Black   Octant VIII
]

for P in octants:
    print_encoding(P, r)