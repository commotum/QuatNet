"""

4-Dimension Attention

A compact, norm-preserving linear transform that elevates RoPE from ℝ² to ℝ⁴.
Rather than 1D→2D planar rotations, it biquaternionically rotates 3D spatial
coordinates alongside a temporal phase. Each 4-D block of Q/K undergoes
block-diagonal rotor spins—guided by position and time—preserving norms
without any learned positional parameters.

"""

import jax.numpy as jnp
from einops import rearrange

def quat_mul(q, p):
    """
    Multiply two quaternions q ⊗ p.
    
    Both q and p are arrays of shape (..., 4) representing [w, x, y, z].
    Returns the quaternion product, same shape.
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
    Multiply two biquaternions (qr + qi·I) ⊗ (xr + xi·I).
    
    qr, qi, xr, xi each are (...,4). Returns two quaternions:
      - real part = qr⊗xr - qi⊗xi
      - imag part = qr⊗xi + qi⊗xr
    """
    real = quat_mul(qr, xr) - quat_mul(qi, xi)
    imag = quat_mul(qr, xi) + quat_mul(qi, xr)
    return real, imag

def bique_rope(x, coords, phase, base: float = 10000.):
    """
    Biquaternion rotary embedding mixing spatial and temporal shifts.

    Args:
      x      : real embedding array of shape (..., dim), where dim % 8 == 0.
      coords : real spatial positions, shape (..., 3).
      phase  : scalar phase (e.g. time), shape (...).
      base   : frequency base for RoPE.

    Returns:
      Transformed embedding of same shape as x.
    """
    # ensure dim can be divided into 8-dim biquaternion slots
    *prefix, dim = x.shape
    assert dim % 8 == 0, "embedding dim must be multiple of 8"
    blocks = dim // 8

    # 1) break embedding into [real_quat, imag_quat] pairs
    xb = rearrange(x, "... (b d) -> ... b d", b=blocks, d=8)
    xr, xi = xb[..., :4], xb[..., 4:]  # xr=real-part inputs, xi=imag-part inputs

    # 2) compute inverse frequencies for each block
    freqs = jnp.arange(blocks)
    inv_freq = 1.0 / (base ** (freqs / (2 * blocks)))  # shape (blocks,)

    # 3) build the circular (space) rotation quaternion qr
    #    map coords to per-block phases
    theta_vec = coords[..., None, :] * inv_freq[None, :, None]   # (...,blocks,3)
    theta = jnp.linalg.norm(theta_vec, axis=-1, keepdims=True)   # (...,blocks,1)
    eps = 1e-6
    axis_u = theta_vec / jnp.maximum(theta, eps)                 # unit axes (...,blocks,3)
    cos_t = jnp.cos(theta/2)
    sin_t = jnp.sin(theta/2)
    qr = jnp.concatenate([cos_t, axis_u * sin_t], axis=-1)       # circular rotor (...,blocks,4)

    # 4) build an initial hyperbolic quaternion qi_init from the same axis
    phi = phase[..., None] * inv_freq[None, :, None]             # (...,blocks,1)
    cosh_p = jnp.cosh(phi/2)
    sinh_p = jnp.sinh(phi/2)
    qi_init = jnp.concatenate([cosh_p, axis_u * sinh_p], axis=-1)

    # 5) extract true hyperbolic rotation axis from conj(qr) ⊗ qi_init
    conj_mult = qr * jnp.array([1., -1., -1., -1.])              # quaternion conjugate of qr
    h_full   = quat_mul(conj_mult, qi_init)                      # full biquaternion (...,blocks,4)
    h_axis   = h_full[..., 1:]                                   # drop real component
    h_norm   = jnp.linalg.norm(h_axis, axis=-1, keepdims=True)
    axis_h   = h_axis / jnp.maximum(h_norm, eps)                 # normalized hyperbolic axis

    # 6) rebuild the hyperbolic quaternion qi with its proper axis
    qi = jnp.concatenate([cosh_p, axis_h * sinh_p], axis=-1)

    # 7) apply the two rotations to the input quaternions
    yr, yi = biquat_mul(qr, qi, xr, xi)

    # 8) recombine and flatten back to original embedding shape
    yb = jnp.concatenate([yr, yi], axis=-1)                      # (...,blocks,8)
    return rearrange(yb, "... b d -> ... (b d)")
