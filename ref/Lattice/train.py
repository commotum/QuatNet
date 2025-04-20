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

    x,y,z = P
    rho = np.linalg.norm(P)
    η   = rho/r if r else 0.0
    θ   = 2*np.pi * η       # full angle
    axis = np.array(P)/rho if rho>0 else np.array([1.,0.,0.])
    w    = np.cos(θ/2)      # ← half‑angle
    s    = np.sin(θ/2)
    xyz  = s * axis
    return np.array([w, *xyz])

# Define a formatting function and demonstrate on sample points

def format_quaternion(q):
    """
    Format a quaternion [w, x, y, z] with leading sign and 4 decimal places.
    Returns a string like "[+w.wwww, -x.xxxx, +y.yyyy, -z.zzzz]".
    """
    return ", ".join(f"{v:+.4f}" for v in q)

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

def unit_lattice(r):
    """
    Generate 3D integer lattice points (x, y, z) where the absolute value of each coordinate
    is equal. For example, points like (±1,±1,±1), (±2,±2,±2), etc. up to radius r.
    
    Args:
        r (int): Maximum radius of lattice points to generate
        
    Returns:
        list: List of (x,y,z) tuples where abs(x)=abs(y)=abs(z) and each ≤ r
    """
    points = []
    for mag in range(1, r+1):
        # Generate all sign combinations for current magnitude
        for signs in product([-1, 1], repeat=3):
            point = tuple(sign * mag for sign in signs)
            points.append(point)
    return points


def map_to_unit_interval(vec, r):
    # now takes raw P in [-r,r]
    return [ (x/r + 1.0)/2.0 for x in vec ]


def to_rgb_hex(mapped_vec, r):
    """
    Convert a 3-vector with values in [-r, r] to an RGB hex color string.
    
    Args:
        mapped_vec (list of float): 3-vector with values in [-r, r]
        r (int): Radius/scale parameter to normalize values
        
    Returns:
        str: RGB hex color string in format "#RRGGBB"
    """
    # Convert [-r, r] values to [0, 255] integers
    rgb = [int((x/r + 1.0) * 127.5) for x in mapped_vec]
    
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def to_rgb_tuple(mapped):
    # mapped ∈ [0,1] → 0–255
    return tuple(int(v * 255) for v in mapped)


def vector_to_color(P, r):
    q = quaternion_encoding(P, r)
    m = map_to_unit_interval(P, r)
    rgb = to_rgb_tuple(m)
    return q, rgb

def print_encoding(P, r):
    """
    Print formatted lattice point, quaternion encoding and RGB color.
    
    Args:
        P (tuple of int): The (x,y,z) lattice coordinates
        r (int): Radius/scale parameter for quaternion encoding
    """
    q, rgb = vector_to_color(P, r)
    print(f"[{format_quaternion(q)}, {rgb[0]}, {rgb[1]}, {rgb[2]}]") #{format_vector(P)} ↦ 
    

r = 4  # radius of lattice
for P in lattice(r):
    print_encoding(P, r)

#print()

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

#for P in octants:
#    print_encoding(P, r)

#print()

#for P in unit_lattice(r):
#    print_encoding(P, r)