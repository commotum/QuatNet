import jax.numpy as jnp
from einops import rearrange

def quat_mul(q, p):
    """
    Standard quaternion multiplication q ⊗ p.
    q, p: shape (..., 4) as [w, x, y, z].
    Returns shape (...,4).
    """
    qw, qx, qy, qz = jnp.split(q, 4, axis=-1)
    pw, px, py, pz = jnp.split(p, 4, axis=-1)
    return jnp.concatenate([
        qw*pw - qx*px - qy*py - qz*pz,
        qw*px + qx*pw + qy*pz - qz*py,
        qw*py - qx*pz + qy*pw + qz*px,
        qw*pz + qx*py - qy*px + qz*pw,
    ], axis=-1)

def biquat_mul(qr, qi, xr, xi):
    """
    (qr + qi·I) ⊗ (xr + xi·I) = (real + imag·I).
    All inputs shape (...,4). Returns (real, imag).
    """
    real = quat_mul(qr, xr) - quat_mul(qi, xi)
    imag = quat_mul(qr, xi) + quat_mul(qi, xr)
    return real, imag

def bique_rope(x, coords, phase, base: float = 10000.):
    """
    Biquaternion RoPE mixing space+time, with true independent hyperbolic axis.
    x      – real array (..., dim), dim % 8 == 0
    coords – (...,3)   spatial positions
    phase  – (...)     scalar phase (time)
    """
    *prefix, dim = x.shape
    assert dim % 8 == 0, "dim must be multiple of 8"
    blocks = dim // 8

    # 1) split into biquaternion blocks
    xb = rearrange(x, "... (b d) -> ... b d", b=blocks, d=8)
    xr, xi = xb[..., :4], xb[..., 4:]

    # 2) frequencies
    j        = jnp.arange(blocks)
    inv_freq = 1.0 / (base ** (j / (2*blocks)))  # (blocks,)

    # 3) circular rotor qr from coords
    θ_vec = coords[..., None, :] * inv_freq[None, :, None]       # (...,b,3)
    angle = jnp.linalg.norm(θ_vec, axis=-1, keepdims=True)       # (...,b,1)
    eps   = 1e-6
    axis_u = θ_vec / jnp.maximum(angle, eps)                     # (...,b,3)
    wr = jnp.cos(angle/2)
    sr = jnp.sin(angle/2)
    qr = jnp.concatenate([wr, axis_u * sr], axis=-1)            # (...,b,4)

    # 4) initial hyperbolic qi (we'll correct its axis next)
    φ  = phase[..., None] * inv_freq[None, :, None]              # (...,b,1)
    ch = jnp.cosh(φ/2)
    sh = jnp.sinh(φ/2)
    qi_init = jnp.concatenate([ch, axis_u * sh], axis=-1)        # (...,b,4)

    # 5) compute true hyperbolic axis via conj(qr) * qi_init
    conj_qr     = qr * jnp.array([1., -1., -1., -1.])            # quaternion conjugate
    h_full      = quat_mul(conj_qr, qi_init)                     # (...,b,4)
    h_axis_vec  = h_full[..., 1:]                                # (...,b,3)
    h_axis_norm = jnp.linalg.norm(h_axis_vec, axis=-1, keepdims=True)
    axis_h      = h_axis_vec / jnp.maximum(h_axis_norm, eps)     # (...,b,3)

    # 6) rebuild correct qi with independent axis
    qi = jnp.concatenate([ch, axis_h * sh], axis=-1)             # (...,b,4)

    # 7) apply bi‑quaternion multiply
    yr, yi = biquat_mul(qr, qi, xr, xi)

    # 8) flatten back
    yb = jnp.concatenate([yr, yi], axis=-1)                      # (...,b,8)
    return rearrange(yb, "... b d -> ... (b d)")

