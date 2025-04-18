import numpy as np
np.random.seed(0)               # reproducible toy example

# ── build one Θₘ rotation block for RoPE ───────────────────
def rope_matrix(dim, m, base=10_000):
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    Θ = np.zeros((dim, dim))
    for i, θ in enumerate(m * inv_freq):
        c, s = np.cos(θ), np.sin(θ)
        Θ[2*i:2*i+2, 2*i:2*i+2] = [[c, -s],
                                   [s,  c]]
    return Θ

# ── toy transformer inputs ────────────────────────────────
seq_len, dim = 4, 10           # 4 tokens, same 10‑D as the RoPE blocks
scale        = dim ** -0.5

X   = np.random.randn(seq_len, dim)   # random token embeddings
W_q = np.random.randn(dim, dim)       # random projections
W_k = np.random.randn(dim, dim)

Q = X @ W_q
K = X @ W_k

# ── logits WITHOUT RoPE ───────────────────────────────────
logits_plain = Q @ K.T * scale

# ── apply RoPE token‑wise ─────────────────────────────────
Q_rot = np.empty_like(Q)
K_rot = np.empty_like(K)
for m in range(seq_len):
    Θ = rope_matrix(dim, m)           # Θₘ for position m
    Q_rot[m] = Θ @ Q[m]
    K_rot[m] = Θ @ K[m]

logits_rope = Q_rot @ K_rot.T * scale

# ── show the effect ───────────────────────────────────────
np.set_printoptions(precision=3, suppress=True)
print("Scaled‑dot‑product logits WITHOUT RoPE:\n", logits_plain, "\n")
print("Scaled‑dot‑product logits WITH    RoPE:\n", logits_rope,  "\n")
print("Difference (RoPE − plain):\n", logits_rope - logits_plain)
