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

