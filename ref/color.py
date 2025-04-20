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
