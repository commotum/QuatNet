import numpy as np

# ───────────────────────────────────────────────────────────
#  RoPE rotation matrix Θ_m for a given position m
# ───────────────────────────────────────────────────────────
def rope_matrix(dim: int = 10, m: int = 7, base: int = 10_000) -> np.ndarray:
    """
    Build the block‑diagonal rotation matrix Θ_m.

    dim  – total (real) dimension, must be even (10 → 5 blocks)
    m    – sequence index
    base – RoPE base (default 10 000, as used in GPT‑NeoX / RoFormer)
    """
    assert dim % 2 == 0, "dimension must be even"
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))  # ω_j
    Θ = np.zeros((dim, dim))

    for i, θ in enumerate(m * inv_freq):                      # θ_j = m·ω_j
        c, s = np.cos(θ), np.sin(θ)
        Θ[2*i:2*i+2, 2*i:2*i+2] = [[c, -s],
                                   [s,  c]]
    return Θ


# ───────────────────────────────────────────────────────────
#  Pretty‑print the matrix with the requested formatting
# ───────────────────────────────────────────────────────────
def format_rope_matrix(Θ: np.ndarray, width: int = 8, eps: float = 1e-5) -> str:
    """Return a string with each entry width‑padded as specified."""
    def fmt(v: float) -> str:
        if abs(v) < eps:                # treat as zero
            return "+0".ljust(width)
        return f"{v:+.5f}"              # sign + five decimals (exact length 8)
    return "\n".join(" ".join(fmt(v) for v in row) for row in Θ)


# ───────────────────────────────────────────────────────────
#  Example: position m = 7
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    Θ_7 = rope_matrix(m=7)             # 10×10, five blocks
    print(format_rope_matrix(Θ_7))
