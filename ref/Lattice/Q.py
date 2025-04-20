import jax.numpy as jnp
from einops import rearrange, repeat

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
    Biquaternion multiply (qr + qi·I) ⊗ (xr + xi·I)
    using (a+bI)(c+dI) = (ac−bd)+(ad+bc)I.
    All inputs shape (...,4).
    Returns two quaternions (real, imag).
    """
    real = quat_mul(qr, xr) - quat_mul(qi, xi)
    imag = quat_mul(qr, xi) + quat_mul(qi, xr)
    return real, imag

def bique_rope(x, coords, phase, base: float = 10000.):
    """
    Biquaternion rotary embedding coupling space & time.

    x      – real array (..., dim), dim % 8 == 0 (bi‑quaternion slots)
    coords – (...,3)   spatial positions
    phase  – (...)     scalar phase (e.g., time)
    """
    *prefix, dim = x.shape
    assert dim % 8 == 0, f"dim ({dim}) must be multiple of 8"
    blocks = dim // 8

    # 1) split into biquaternion blocks
    xb = rearrange(x, "... (b d) -> ... b d", b=blocks, d=8)
    assert xb.shape[-1] == 8
    xr, xi = xb[..., :4], xb[..., 4:]
    assert xr.shape[-1] == xi.shape[-1] == 4

    # 2) canonical RoPE inv_freq
    j        = jnp.arange(blocks)
    inv_freq = 1.0 / (base ** (j / (2 * blocks)))  # (blocks,)

    # 3) spatial rotor qr from coords
    #    coords[..., None, :] -> (..., 1, 3)
    #    inv_freq[None, :, None] -> (1, blocks, 1) broadcasts to (..., blocks, 1)
    θ_vec = coords[..., None, :] * inv_freq[None, :, None]  # (..., blocks, 3)
    angle = jnp.linalg.norm(θ_vec, axis=-1, keepdims=True)  # (..., blocks, 1)

    # avoid division by zero
    eps  = 1e-6
    axis = θ_vec / jnp.maximum(angle, eps)                  # (..., blocks, 3)

    wr = jnp.cos(angle / 2)
    sr = jnp.sin(angle / 2)
    qr = jnp.concatenate([wr, axis * sr], axis=-1)          # (..., blocks, 4)
    assert qr.shape[-1] == 4

    # 4) hyperbolic rotor qi reusing same axis
    φ  = phase[..., None] * inv_freq[None, :, None]         # (..., blocks, 1)
    ch = jnp.cosh(φ / 2)
    sh = jnp.sinh(φ / 2)
    qi = jnp.concatenate([ch, axis * sh], axis=-1)          # (..., blocks, 4)
    assert qi.shape[-1] == 4

    # 5) apply biquaternion multiply
    yr, yi = biquat_mul(qr, qi, xr, xi)
    assert yr.shape[-1] == yi.shape[-1] == 4

    # 6) flatten back
    yb = jnp.concatenate([yr, yi], axis=-1)                 # (..., blocks, 8)
    return rearrange(yb, "... b d -> ... (b d)")
