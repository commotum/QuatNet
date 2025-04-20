import numpy as np

def quaternion_encoding(P, d):
    x, y, z = P
    MAX = d // 2
    SPREAD = 4 * MAX
    total_abs = abs(x) + abs(y) + abs(z)
    w = SPREAD - total_abs
    q = np.array([w, x, y, z], dtype=float)
    norm = np.linalg.norm(q)
    return q / norm if norm != 0 else np.array([1.0, 0.0, 0.0, 0.0])

def format_quaternion(q):
    return "[" + ", ".join(f"{v:+.4f}" for v in q) + "]"

def format_vector(v):
    return "[" + ", ".join(f"{int(val):+d}" for val in v) + "]"

def format_rgb(rgb):
    return "[" + ", ".join(f"{val:3d}" for val in rgb) + "]"

def get_rgb_from_vector(P, MAX):
    rgb_float = [MAX + i if i < 0 else i for i in P]
    rgb_scaled = [int(round(np.clip(f / MAX * 255, 0, 255))) for f in rgb_float]
    return tuple(rgb_scaled)

# Parameters
d = 2
h = d // 2
coords = range(-h, h + 1)

# Output loop
for x in coords:
    for y in coords:
        for z in coords:
            P = (x, y, z)
            q = quaternion_encoding(P, d)
            rgb = get_rgb_from_vector(P, d // 2)
            print(f"{format_vector(P)} {format_quaternion(q)} {format_rgb(rgb)}")


