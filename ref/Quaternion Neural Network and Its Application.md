# Quaternion Neural Network and Its Application

#### Teijiro Isokawa ${}^1$, Tomoaki Kusakabe ${}^1$, Nobuyuki Matsui ${}^1$, and Ferdinand Peper ${}^2$
${}^1$ Division of Computer Engineering, Himeji Institute of Technology, Japan \{isokawa, tomoaki, matsui\}@comp.eng.himeji-tech.ac.jp  
${}^2$ Communications Research Laboratory, Nanotechnology Group, Japan peper@crl.go.jp


### Abstract  
Quaternion neural networks are models of which computations in the neurons is based on quaternions, the four-dimensional equivalents of imaginary numbers. This paper shows by experiments that the quaternion-version of the Back Propagation (BP) algorithm achieves correct geometrical transformations in color space for an image compression problem, whereas real-valued BP algorithms fail.

## 1 Introduction
Though most real-valued neural network models are able to learn arbitrary nonlinear functions, they perform less well when it comes to geometrical transformations, like affine transformations in 2 or 3 dimensional space. Some researchers[1] have found that the use of complex-valued neural networks results in improved performance on such transformation problems. These results inspired the Quaternion Neural Network Model in [2], which is trained by a BP learning algorithm. Such models employ neurons in which all computations are based on quaternions, a four-dimensional extension of imaginary numbers discovered by Sir W.R. Hamilton, which have found extensive use in modern mathematics and physics[3]. It turns out that such a quaternion neural model learns 3D affine transformations very well[2]. Using this as the starting point in this paper, we apply a quaternion BP learning algorithm on a color image compression problem, which is a problem in which the fidelity of colors is only preserved when the affine transformations in color space are correct. Experiments show improved performance of our quaternion scheme as compared to real-valued BP.

## 2 Quaternion
Quaternions form a class of hypercomplex numbers that consist of a real number and three kinds of imaginary number, $\boldsymbol{i}, \boldsymbol{j}, \boldsymbol{k}$. Formally, a quaternion is defined as a vector $\boldsymbol{x}$ in a 4 -dimensional vector space, i.e.,

$$
\begin{equation}
\boldsymbol{x}=x^{(e)}+x^{(i)} \boldsymbol{i}+x^{(j)} \boldsymbol{j}+x^{(k)} \boldsymbol{k} \qquad \qquad (1)
\end{equation}
$$

where $x^{(e)}$ and $x^{(i)}, x^{(j)}, x^{(k)}$ are real numbers. $K^4$, the division ring of quaternions, thus constitutes the four-dimensional vector space over the real numbers with the bases $1, \boldsymbol{i}, \boldsymbol{j}, \boldsymbol{k}$.

Quaternions satisfy the following identities, known as the Hamilton rules:

$$
\begin{aligned}
\boldsymbol{i}^2 = \boldsymbol{j}^2 = \boldsymbol{k}^2 = \boldsymbol{i}\boldsymbol{j}\boldsymbol{k} = -1, &\qquad \qquad (2)\\
\boldsymbol{i}\boldsymbol{j} = -\boldsymbol{j}\boldsymbol{i} = \boldsymbol{k}, 
\boldsymbol{j}\boldsymbol{k} = -\boldsymbol{k}\boldsymbol{j} = \boldsymbol{i}, 
\boldsymbol{k}\boldsymbol{i} = -\boldsymbol{i}\boldsymbol{k} = \boldsymbol{j}. &\qquad \qquad (3)
\end{aligned}
$$

From these rules it follows immediately that multiplication of quaternions is not commutative.

$$
\begin{aligned}
\overline{\boldsymbol{x}}=x^{(e)} -x^{(i)} \boldsymbol{i} -x^{(j)} \boldsymbol{j} -x^{(k)} \boldsymbol{k} &\qquad \qquad (4)
\end{aligned}
$$

where $x^{(e)}$ is regarded as the real part and $x^{(i)} i+x^{(j)} j+x^{(k)} k$ as the imaginary part of $\boldsymbol{x}$.

The quaternion norm of $\boldsymbol{x}, \mathrm{n}(\boldsymbol{x})$, is defined by

$$
\begin{aligned}
n(\boldsymbol{x}) & =\sqrt{\boldsymbol{x} \overline{\boldsymbol{x}}}=\sqrt{\overline{\boldsymbol{x}} \boldsymbol{x}} \\ & =\sqrt{x^{(e)^2}+x^{(j)^2}+x^{(j)^2}+x^{(k)^2}} &\qquad \qquad (5)
\end{aligned}
$$

For convenience in the following explanation, we define the purely imaginary quaternion as a quaternion with zero real part. A purely imaginary quaternion $x$ can thus be expressed as

$$
\begin{aligned}
\boldsymbol{x}=x^{(i)} i+x^{(j)} j+x^{(k)} \boldsymbol{k} \hfill (6)
\end{aligned}
$$