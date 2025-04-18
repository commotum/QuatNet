## Discrete Cubic Lattice and Quaternion Encoding of Spatial Positions

### 1. Discrete Integer Cubic Lattice

Consider an even integer \( d \in 2\mathbb{N} \), representing the **edge length** of a discrete cubic lattice. Define the symmetric integer cubic lattice as follows:

\[
\Lambda_d = \left\{ (x,y,z) \in \mathbb{Z}^3 : -\frac{d}{2} \le x,y,z \le \frac{d}{2} \right\}.
\]

The **half-extent** (or semiedge length) of the cube is given by:

\[
h = \frac{d}{2},
\]

resulting in vertices located at \((\pm h, \pm h, \pm h)\).

### 2. Circumradius as Reference Length

The **circumradius** \(R\), or outer radius, is the Euclidean distance from the origin to any corner vertex of the lattice:

\[
R = \| (h,h,h) \| = h\sqrt{3} = \frac{d\sqrt{3}}{2}.
\]

### 3. Normalized Radial Coordinate

For any lattice point \( \mathbf{P} = (x,y,z) \in \Lambda_d \), define the radial distance \( \rho \) from the origin as:

\[
\rho = \| \mathbf{P} \|,
\]

and the normalized radial coordinate \( \eta \) as:

\[
\eta = \frac{\rho}{R} \in [0,1].
\]

### 4. Angle Mapping

The normalized radial coordinate \( \eta \) is mapped linearly to an angle \( \theta \in [0, \pi] \) as:

\[
\theta = \eta\pi.
\]

### 5. Axis and Quaternion Encoding

Define the unit vector \( \mathbf{V} \) corresponding to lattice point \( \mathbf{P} \):

\[
\mathbf{V} = \begin{cases}
\frac{\mathbf{P}}{\rho}, & \rho \neq 0, \\
(1,0,0), & \rho = 0.
\end{cases}
\]

The quaternion encoding \( Q(\mathbf{P}) \) for spatial positions within the lattice is thus defined as:

\[
Q(\mathbf{P}) = \cos\left(\frac{\theta}{2}\right) + \sin\left(\frac{\theta}{2}\right)(V_x\mathbf{i} + V_y\mathbf{j} + V_z\mathbf{k}) \in \mathbb{H},
\]

where \( \mathbb{H} \) denotes the quaternion algebra.

### 6. Illustrative Example

Consider \( d = 512 \), hence \( h = 256 \) and \( R = 256\sqrt{3} \). Given the lattice point \( \mathbf{P} = (128, -20, 5) \):

Compute:

\[
\begin{aligned}
\rho &= \sqrt{128^2 + (-20)^2 + 5^2} \approx 129.71, \\
\eta &= \frac{129.71}{256\sqrt{3}} \approx 0.293, \\
\theta &= \eta \pi \approx 0.921\;\text{rad}.
\end{aligned}
\]

The corresponding unit vector is:

\[
\mathbf{V} = \frac{(128, -20, 5)}{129.71} \approx (0.987, -0.154, 0.039).
\]

Thus, the quaternion encoding is:

\[
Q(\mathbf{P}) = \cos(0.4605) + \sin(0.4605)(0.987\mathbf{i} - 0.154\mathbf{j} + 0.039\mathbf{k}).
\]

### Terminology Summary

| Quantity | Symbol | Customary Name |
|----------|--------|----------------|
| \( h = d/2 \) | Half-extent (semiedge) |
| \( R \) | Circumradius (outer radius) |
| \( \eta \) | Normalized radial coordinate |
| \( Q(\mathbf{P}) \) | Quaternion encoding of lattice point |

This structured approach provides an embedding from integer lattice points within an even-sized cubic lattice into the unit quaternion 3-sphere, preserving radial ordering through the linear angle mapping \( \eta \mapsto \theta \).



Step | RoPE (1D) | 3D → Quaternion
Input | token index i∈{0…L}i\in\{0…L\}i∈{0…L} | lattice point P=(x,y,z)\mathbf P=(x,y,z)P=(x,y,z)
Normalize | α=iL∈[0,1]\displaystyle\alpha=\tfrac{i}{L}\in[0,1]α=Li​∈[0,1] | η=ρR∈[0,1],  ρ=∥P∥\displaystyle\eta=\tfrac{\rho}{R}\in[0,1],\;\rho=\|\mathbf P\|η=Rρ​∈[0,1],ρ=∥P∥
Angle | θ=2π α\displaystyle\theta=2\pi\,\alphaθ=2πα (or θ=i ω\theta=i\,\omegaθ=iω) | θ=π η\displaystyle\theta=\pi\,\etaθ=πη
Rotation element | Complex plane: eiθ=(cos⁡θ,sin⁡θ)e^{i\theta}=(\cos\theta,\sin\theta)eiθ=(cosθ,sinθ) | Quaternion: e(θ/2)V=(cos⁡θ2,  sin⁡θ2 V)e^{(\theta/2) V}=\bigl(\cos\tfrac\theta2,\;\sin\tfrac\theta2\,V\bigr)e(θ/2)V=(cos2θ​,sin2θ​V)
Axis | implicit 2D basis (1,i)(1,i)(1,i) | axis‑angle basis V=P/ρV=\mathbf P/\rhoV=P/ρ
Embed | interleave cos⁡θ\cos\thetacosθ & sin⁡θ\sin\thetasinθ across hidden dims | pack (w,x′,y′,z′)=(cos⁡θ2,sin⁡θ2 V)(w,x',y',z')=(\cos\tfrac\theta2,\sin\tfrac\theta2\,V)(w,x′,y′,z′)=(cos2θ​,sin2θ​V) into 4‑dim slots