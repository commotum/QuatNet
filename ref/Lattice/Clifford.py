"""
Abstract:
We introduce Clifford RoPE, a 4-dimensional, norm-preserving positional embedding for transformer attention
built on the real Clifford algebra Cl(1,3) (signature (+,-,-,-)). Unlike traditional RoPE—which groups
dimensions into 2-D complex planes or 8-D biquaternion blocks to encode rotations and boosts—Clifford RoPE
works directly with 4-vectors (t,x,y,z). For each token position, we:

1. Compute per-block inverse frequencies for spatial and temporal modulation.
2. Build a 3×3 spatial rotation via Rodrigues’ formula and embed it in a 4×4 block.
3. Assemble a 4×4 Minkowski boost along the same axis using cosh/sinh.
4. Multiply boost × rotation to get a single Cl(1,3) rotor R∈Spin(1,3).
5. Apply R to each 4-vector sub-block, preserving the Minkowski norm and encoding both
   time-like “before vs after” and spatial orientation in one real-valued transform.

Clifford RoPE uses 4 floats per block vs. 8 for biquaternions, cuts per-block flops by ~4×, and amortizes
rotor build across all heads/blocks per position. No complex or quaternion code—pure real JAX ops.
"""

import jax.numpy as jnp
from einops import rearrange

def build_rotation_matrix(axis, theta):
    """
    Rodrigues' formula for 3×3 rotation about 'axis' by angle theta.
    axis: (...,3) unit vectors, theta: (...,1) angles.
    """
    cos_t = jnp.cos(theta)[...,0]
    sin_t = jnp.sin(theta)[...,0]
    ux, uy, uz = axis[...,0], axis[...,1], axis[...,2]
    uuT = jnp.stack([
        ux*ux, ux*uy, ux*uz,
        uy*ux, uy*uy, uy*uz,
        uz*ux, uz*uy, uz*uz
    ], -1).reshape((*axis.shape[:-1],3,3))
    zero = jnp.zeros_like(ux)
    u_cross = jnp.stack([
        zero,  -uz,   uy,
        uz,    zero, -ux,
       -uy,     ux,  zero
    ], -1).reshape((*axis.shape[:-1],3,3))
    I3 = jnp.eye(3)
    return (cos_t[...,None,None]*I3
          + (1-cos_t)[...,None,None]*uuT
          + sin_t[...,None,None]*u_cross)

def clifford_rope(x, coords, times, base: float = 10000.):
    """
    Clifford RoPE: real Cl(1,3) rotors on 4D blocks of Q/K.
    x     : (..., d) embeddings, d % 4 == 0
    coords: (...,3) spatial positions (x,y,z)
    times : (...,)  temporal positions (t)
    """
    *pref, d = x.shape
    assert d % 4 == 0, "embedding dim must be multiple of 4"
    B = d // 4

    xb = rearrange(x, "... (b k) -> ... b k", b=B, k=4)

    freqs = jnp.arange(B)
    invf = 1.0 / (base ** (freqs / (2 * B)))  # (B,)

    theta_vec = coords[..., None, :] * invf[None, :, None]  # (...,B,3)
    theta = jnp.linalg.norm(theta_vec, axis=-1, keepdims=True)  # (...,B,1)
    axis_u = theta_vec / jnp.maximum(theta, 1e-6)

    R3 = build_rotation_matrix(axis_u, theta)  # (...,B,3,3)
    M_rot = jnp.zeros((*R3.shape[:-2],4,4))
    M_rot = M_rot.at[...,0,0].set(1.0)
    M_rot = M_rot.at[...,1:,1:].set(R3)

    phi = times[..., None] * invf[None, :, None]  # (...,B,1)
    ch = jnp.cosh(phi)
    sh = jnp.sinh(phi)
    M_boost = jnp.zeros((*axis_u.shape[:-1],4,4))
    M_boost = M_boost.at[...,0,0].set(ch[...,0])
    M_boost = M_boost.at[...,0,1:].set(axis_u * sh[...,0][...,None])
    M_boost = M_boost.at[...,1:,0].set(axis_u * sh[...,0][...,None])
    eye3 = jnp.eye(3)
    M_boost = M_boost.at[...,1:,1:].set(
        eye3 + (ch[...,0]-1)[...,None,None] * (axis_u[..., :,None] * axis_u[...,None, :])
    )

    R4 = jnp.einsum("...ij,...jk->...ik", M_boost, M_rot)
    yb = jnp.einsum("...ij,...bj->...bi", R4, xb)
    return rearrange(yb, "... b k -> ... (b k)")
