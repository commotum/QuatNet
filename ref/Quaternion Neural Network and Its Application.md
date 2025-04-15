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
\boldsymbol{x}=x^{(i)} i+x^{(j)} j+x^{(k)} \boldsymbol{k} \qquad \qquad (6)
\end{aligned}
$$

## 3 Geometric Description by Quaternion

A rotation $\boldsymbol{g}$ in 3 D vector space $\boldsymbol{I} \in \boldsymbol{K}^3$ can be computed as

$$
\begin{aligned}
\boldsymbol{g}=\boldsymbol{a} v \overline{\boldsymbol{a}} \qquad \qquad (7)
\end{aligned}
$$

where $\boldsymbol{a}$ is a quaternion with $n(\boldsymbol{a})=1$ and $\boldsymbol{v}$ is a purely imaginary quaternion with $n(\boldsymbol{v})=1$. This quaternion $\boldsymbol{a}$ is denoted by

$$
\begin{aligned}
\boldsymbol{a}=\cos \alpha+(\sin \alpha) \cdot \boldsymbol{u} \qquad \qquad (8)
\end{aligned}
$$

where $\alpha$ is an angle satisfying $|\alpha| \leq \pi$ and $\boldsymbol{u}$ is a purely imaginary quaternion with $n(\boldsymbol{u})=1$. Eq.(7) can always be applied whether $\boldsymbol{u}$ and $\boldsymbol{v}$ are orthogonal or not. In the case that $\boldsymbol{u}$ and $\boldsymbol{v}$ are orthogonal, rotation $\boldsymbol{g}$ is expressed from Eq.(8) as

$$
\begin{aligned} 
\boldsymbol{g}=(\cos 2 \alpha) \boldsymbol{v}+(\sin 2 \alpha) \boldsymbol{u} \times \boldsymbol{v} \qquad \qquad (9)
\end{aligned}
$$

This shows the vector $\boldsymbol{v}$ rotated by the angle $2 \alpha$ around $\boldsymbol{u}$ (see Fig.1). In the case that $\boldsymbol{u}$ and $\boldsymbol{v}$ are not orthogonal, Eq.(7) is expressed as

$$
\begin{aligned} 
\boldsymbol{g} & =\boldsymbol{a}\left(\boldsymbol{v}_1+\boldsymbol{v}_2\right) \overline{\boldsymbol{a}}=\boldsymbol{a} \boldsymbol{v}_1 \overline{\boldsymbol{a}}+\boldsymbol{a} \boldsymbol{v}_2 \overline{\boldsymbol{a}} \\
&=\boldsymbol{v}_1+(\sin 2 \alpha)\left(\boldsymbol{u} \times \boldsymbol{v}_2\right)+(\cos 2 \alpha) \boldsymbol{v}_2 \qquad \qquad (10)
\end{aligned}
$$

where $\boldsymbol{v}_1$ and $\boldsymbol{v}_2$ are the components of vector $\boldsymbol{v}$ and satisfy $\boldsymbol{v}_1 \parallel \boldsymbol{u}$ and $\boldsymbol{v}_2 \perp \boldsymbol{u}$. Thus Eq.(10) also shows the vector $\boldsymbol{v}$ rotated by the angle $2 \alpha$ around $\boldsymbol{u}$.

## 4 Quaternion Neural Network

In this section we propose a layered quaternion neural network model and a quaternion BP algorithm to train it. Our quaternion neuron model adopts purely imaginary quaternions as input and output signals. The output $y_j$ of quaternion neuron $j$ is expressed as

$$
\begin{aligned}
& \boldsymbol{s}_{\boldsymbol{j}}=\sum_i \frac{\boldsymbol{w}_{j i} \boldsymbol{x}_i \overline{\boldsymbol{w}}_{j i}}{\left|\boldsymbol{w}_{\boldsymbol{j} \boldsymbol{i}}\right|}-\boldsymbol{\theta}_{\boldsymbol{j}} &\qquad \qquad (11) \\
& \boldsymbol{y}_{\boldsymbol{j}}=f\left(\boldsymbol{s}_{\boldsymbol{j}}\right) &\qquad \qquad (12)
\end{aligned}
$$

where $i$ denotes the indices of neurons in the previous layer, and $\boldsymbol{x}, \boldsymbol{y}, \theta, \boldsymbol{s} \in I$, $\boldsymbol{w} \in \boldsymbol{K}^4$ respectively are the vector of inputs to the neurons, the vector of outputs from the neurons, the threshold, the internal potential, and the weights of the connections to the neurons in layer $i$. The activation function $f$ is defined by

$$
\begin{aligned}
& f(s)=h\left(s^{(i)}\right) \boldsymbol{i}+h\left(s^{(j)}\right) \boldsymbol{j}+h\left(s^{(k)}\right) \boldsymbol{k} &\qquad \qquad (13) \\
& h(x)=\frac{1}{1+e^{-x}} &\qquad \qquad (14)
\end{aligned}
$$

The quaternion neurons as defined above form the basis of a layered quaternion neural network. We adopt the quaternion equivalent of the BP algorithm for training the network, and define the error $E$ between the output values of the network and the target training data as

$$
\begin{aligned}
E & =\frac{1}{2}\left(e^{(i)^2}+e^{(j)^2}+e^{(k)^2}\right) \\
e^{(\mu)} & =y_k^{(\mu)}-d^{(\mu)}, \mu=\{i, j, k\}
\end{aligned}
$$

where $y_k^{(\mu)}$ is the output values of the neuron in the output layer and $d^{(\mu)}$ is the target value. The connection weights $\boldsymbol{w}$ are updated by the gradient descent method:

$$
\begin{aligned}
\boldsymbol{w}^{\text {new }} & =\boldsymbol{w}^{o l d}+\Delta \boldsymbol{w} \\
\Delta w^{(\nu)} & =-\eta \cdot \frac{\partial E}{\partial w^{(\nu)}}, \quad \nu=\{e, i, j, k\}
\end{aligned}
$$

where $\eta$ is a constant denoting the learning coefficient.

## 5 Experimental Results

### 5.1 Image Compression by Neural Networks
To show the improved performance of quaternion BP as compared to real-valued (conventional) BP, we conduct image compression using BP on the neural networks, a task first proposed in [4]. To conduct image compression, a neural network with three layers is used, such that the number of neurons in the input layer is the same as in the output layer, and the number of neurons in the hidden layer is less than the number of neurons in the input layer. The hidden layer thus acts as a bottleneck, through which the information from the input layer to the output layer must pass with as little information loss as possible. An image to be compressed is prepared and uniformly divided into many samples of small subimages. The network is then trained by inputting the samples to both the input as well as the output neurons of the network. As a result of the training the network will try to find values of its weights such that there is a strong association between the input and the output. Such types of networks are also called autoassociators. Figure 2 shows an example of a network for the image compression problem.

The following conditions are used for the computer simulations. The images used consist of $256 \times 256(=65536)$ pixels, each of which has its color values represented by 24 bits. The images are divided into 4096 samples of $4 \times 4(=16)$ pixels in size. The neural networks have three layers, and the neurons of the networks