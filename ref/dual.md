This paper combines the composition of Chinese characters with KG to build the CRKG, an innovative way of splitting Chinese characters into quadruples (h, r, t, a) according to their radicals. Here, h is the primary component of a Chinese character, t is the Chinese character to be studied, a is the minor component of the character, and r is how the primary component is connected to the minor component. This approach explores the relationship between radicals and the sound, form, and meaning of Chinese characters.

The concept of the dual number is similar to the complex number, which has the mathematical form $\hat{q}=q_1+q_2 \xi$ and satisfies $\xi^2=0$, where $q_1$ and $q_2$ represent the real and dual parts, respectively, and $\xi$ represents the dual operator. A dual quaternion is an dual pair of quaternions whose real and dual parts are both quaternions, also known as octonions. It has the ability to the rotary of quaternions and the translation ability of dual numbers and a conventional quaternion can only represent rotation in space, while a dual quaternion can represent any combination of rotation and translation in space. Some dual quaternion properties are as follows:

Conjugate: The conjugate of a dual quaternion $\hat{q}=q_1+q_2 \xi$ is defined as $\hat{q}^*=q_1^*+q_2^* \xi$.

Norm: The norm of a dual quaternion $\hat{q}=q_1+q_2 \xi$ is defined as $\|\hat{q}\|=\sqrt{\hat{q} \hat{q}^*}=\sqrt{\hat{q}^*} \hat{q}$, and when $q_1 \neq 0$, the norm can be written as

$\|\hat{q}\|=\left\|q_1\right\|+\xi \frac{\left\langle q_1, q_2\right\rangle}{\left\|q_1\right\|}$.

Unit dual quaternions are those satisfying $\|\hat{q}\|=1$.

Inverse: The inverse of a dual quaternion is defined only when $q_1 \neq$ 0. In this case, we have
$\hat{q}^{-1}=\frac{\hat{q}^*}{\|\hat{q}\|^2}$
Note that the inverse of a unit dual quaternion is just conjugation.

Rigid Transformation: Convert the unit dual quaternion to translation and rotation form as follows:
$\hat{q}=r+\frac{1}{2} t r \xi$
where $r$ stands for rotation unit quaternion, and $r=\left[\cos \frac{\theta}{2}, \sin \frac{\theta}{2} \mathbf{v}\right]$ be a quaternion that represents a rotation about the unit vector $\mathbf{v}$ through $\theta$, where $\mathbf{v} \in \mathbb{R} \mathbf{i}+\mathbb{R} \mathbf{j}+\mathbb{R} \mathbf{k}$ is a unit vector. $t$ stands for translation quaternion and $t=1+\frac{\xi}{2}\left(t_0 \mathbf{i}+t_1 \mathbf{j}+t_2 \mathbf{k}\right)$. Corresponds to translation by vector $\left(t_0, t_1, t_2\right)$. If we have a 3 D vector $\left(v_0, v_1, v_2\right)$, we define the associated unit dual $\hat{v}=1+\xi\left(v_0 \mathbf{i}+v_1 \mathbf{j}+v_2 \mathbf{k}\right)$, the rotation of vector $\left(v_0, v_1, v_2\right)$ by a dual quaternion $\hat{q}$ can then be written as $\hat{q} \hat{v} \overline{\hat{q}^*}$.