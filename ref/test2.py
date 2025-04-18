import numpy as np

def rgb_to_quaternion(rgb, normalize=True):
    """
    Convert an RGB pixel to a purely imaginary quaternion.
    
    Parameters:
    rgb (np.array): Array of [R, G, B] values (integers 0-255 or floats 0-1).
    normalize (bool): If True, normalize RGB values to [0, 1].
    
    Returns:
    np.array: Quaternion [0, R, G, B] (real part = 0).
    """
    # Ensure RGB is a numpy array
    rgb = np.array(rgb, dtype=float)
    
    # Normalize to [0, 1] if requested
    if normalize:
        rgb = rgb / 255.0
    
    # Form purely imaginary quaternion: [0, R, G, B]
    return np.array([0.0, rgb[0], rgb[1], rgb[2]])

# Example usage
rgb_pixel = np.array([255, 128, 0])
quat = rgb_to_quaternion(rgb_pixel, normalize=True)
print("Original RGB: [{:+04d}, {:+04d}, {:+04d}]".format(
    int(rgb_pixel[0]), int(rgb_pixel[1]), int(rgb_pixel[2])))
print("Quaternion RGB: [{:+.4f}, {:+.4f}, {:+.4f}, {:+.4f}]".format(
    quat[0], quat[1], quat[2], quat[3]))