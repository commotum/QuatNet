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

def fixed_bique_rope(x, coords, phase, base: float = 10000.):
    """
    x     – real array (..., dim), dim % 8 == 0
             interpreted as biquaternion slots of size 8.
    coords– (...,3) real or int spatial positions
    phase – (...)     real scalar phase (eg time)
    """
    *prefix, dim = x.shape
    assert dim % 8 == 0, "dim must be multiple of 8"
    blocks = dim // 8

    # 1) split into biquaternion blocks
    xb = rearrange(x, "... (b d) -> ... b d", b=blocks, d=8)
    xr = xb[..., :4]   # real quaternion part
    xi = xb[..., 4:]   # imag quaternion part

    # 2) per‑block inv_freq
    inv_freq = 1.0 / (base ** (jnp.arange(blocks) / blocks))  # (b,)

    # 3) build q_r from (x,y,z)
    θ_vec = coords[..., None, :] * inv_freq               # (...,b,3)
    angle = jnp.linalg.norm(θ_vec, axis=-1, keepdims=True)  # (...,b,1)
    axis  = jnp.where(angle>0, θ_vec/angle, jnp.array([1.,0,0])[None,None,:])
    wr = jnp.cos(angle/2)
    sr = jnp.sin(angle/2)
    qr = jnp.concatenate([wr, axis*sr], axis=-1)           # (...,b,4)

    # 4) build q_i from phase t (hyperbolic rotor)
    φ = phase[..., None] * inv_freq                        # (...,b)
    ch = jnp.cosh(φ/2)[..., None]
    sh = jnp.sinh(φ/2)[..., None]
    # pick a fixed quaternion axis for the hyperbolic part (e.g. i‑axis)
    axis_t = jnp.array([1.,0,0])[None,None,:]
    qi = jnp.concatenate([ch, axis_t*sh], axis=-1)         # (...,b,4)

    # 5) do the biquaternion multiply on each slot
    yr, yi = biquat_mul(qr, qi, xr, xi)  # each (...,b,4)

    # 6) flatten back to (..., dim)
    yb = jnp.concatenate([yr, yi], axis=-1)                # (...,b,8)
    return rearrange(yb, "... b d -> ... (b d)")
