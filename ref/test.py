import numpy as np

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / norm

def quaternion_multiply(q1, q2):
    """Multiply two quaternions q1 = [w1, x1, y1, z1], q2 = [w2, x2, y2, z2]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """Return the conjugate of a quaternion q = [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def vector_to_quaternion(v):
    """Convert a 3D vector to a purely imaginary quaternion."""
    return np.array([0.0, v[0], v[1], v[2]])

def quaternion_to_vector(q):
    """Extract the vector part from a quaternion."""
    return q[1:4]

def rgb_to_quaternion(rgb, normalize=True):
    """
    Convert an RGB pixel to a purely imaginary quaternion.
    
    Parameters:
    rgb (np.array): Array of [R, G, B] values (integers 0-255 or floats 0-1).
    normalize (bool): If True, normalize RGB values to [0, 1].
    
    Returns:
    np.array: Quaternion [0, R, G, B] (real part = 0).
    """
    rgb = np.array(rgb, dtype=float)
    if normalize:
        rgb = rgb / 255.0
    return np.array([0.0, rgb[0], rgb[1], rgb[2]])

def quaternion_to_rgb(q, denormalize=True):
    """
    Convert a purely imaginary quaternion to RGB values.
    
    Parameters:
    q (np.array): Quaternion [0, R, G, B].
    denormalize (bool): If True, scale values to [0, 255].
    
    Returns:
    np.array: RGB values as [R, G, B].
    """
    rgb = quaternion_to_vector(q)
    if denormalize:
        rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255)  # Ensure values are in valid RGB range
    return rgb

def rotate_vector(v, u, alpha):
    """
    Rotate vector v by angle 2*alpha around axis u using quaternion rotation.
    
    Parameters:
    v (np.array): 3D vector to rotate (unit vector or normalized).
    u (np.array): 3D rotation axis (unit vector).
    alpha (float): Rotation angle in radians, |alpha| <= pi.
    
    Returns:
    np.array: Rotated vector g.
    """
    v = normalize(v)
    u = normalize(u)
    if abs(alpha) > np.pi:
        raise ValueError("Angle alpha must satisfy |alpha| <= pi.")
    
    a = np.array([np.cos(alpha), *(np.sin(alpha) * u)])
    v_quat = vector_to_quaternion(v)
    a_conj = quaternion_conjugate(a)
    temp = quaternion_multiply(a, v_quat)
    g_quat = quaternion_multiply(temp, a_conj)
    return quaternion_to_vector(g_quat)

def rotate_vector_orthogonal(v, u, alpha):
    """
    Rotate vector v by angle 2*alpha around axis u using Eq. 9 (orthogonal case).
    """
    v = normalize(v)
    u = normalize(u)
    u_cross_v = np.cross(u, v)
    g = np.cos(2 * alpha) * v + np.sin(2 * alpha) * u_cross_v
    return g

def rotate_vector_non_orthogonal(v, u, alpha):
    """
    Rotate vector v by angle 2*alpha around axis u using Eq. 10 (non-orthogonal case).
    """
    v = normalize(v)
    u = normalize(u)
    v1 = (np.dot(v, u)) * u
    v2 = v - v1
    u_cross_v2 = np.cross(u, v2)
    g = v1 + np.sin(2 * alpha) * u_cross_v2 + np.cos(2 * alpha) * v2
    return g

def are_orthogonal(u, v, tol=1e-6):
    """Check if vectors u and v are orthogonal."""
    return abs(np.dot(u, v)) < tol

def rotate_rgb(rgb, u, alpha, normalize_input=True, denormalize_output=True):
    """
    Rotate an RGB pixel in color space using quaternion rotation.
    
    Parameters:
    rgb (np.array): Input RGB values [R, G, B] (integers 0-255).
    u (np.array): 3D rotation axis (unit vector or normalized).
    alpha (float): Rotation angle in radians, |alpha| <= pi.
    normalize_input (bool): If True, normalize RGB to [0, 1] before rotation.
    denormalize_output (bool): If True, scale output to [0, 255].
    
    Returns:
    np.array: Rotated RGB values [R, G, B].
    """
    # Convert RGB to quaternion
    q = rgb_to_quaternion(rgb, normalize=normalize_input)
    
    # Extract vector part for rotation
    v = quaternion_to_vector(q)
    
    # Rotate the vector
    g = rotate_vector(v, u, alpha)
    
    # Convert back to quaternion
    g_quat = vector_to_quaternion(g)
    
    # Convert to RGB
    rgb_rotated = quaternion_to_rgb(g_quat, denormalize=denormalize_output)
    
    # Verify with orthogonal/non-orthogonal methods
    if are_orthogonal(u, v):
        g_ortho = rotate_vector_orthogonal(v, u, alpha)
        rgb_ortho = quaternion_to_rgb(vector_to_quaternion(g_ortho), denormalize=denormalize_output)
        print("Orthogonal method RGB:", rgb_ortho)
    else:
        g_non_ortho = rotate_vector_non_orthogonal(v, u, alpha)
        rgb_non_ortho = quaternion_to_rgb(vector_to_quaternion(g_non_ortho), denormalize=denormalize_output)
        print("Non-orthogonal method RGB:", rgb_non_ortho)
    
    return rgb_rotated

# Example usage
if __name__ == "__main__":
    # Example RGB pixel (0-255)
    rgb = np.array([255, 128, 0])
    
    # Rotation axis (unit vector)
    u = np.array([0.0, 1.0, 0.0])
    
    # Rotation angle (45 degrees in radians)
    alpha = np.pi / 4
    
    # Rotate RGB
    rgb_rotated = rotate_rgb(rgb, u, alpha, normalize_input=True, denormalize_output=True)
    print("Original RGB:", rgb)
    print("Rotated RGB (quaternion method):", rgb_rotated)