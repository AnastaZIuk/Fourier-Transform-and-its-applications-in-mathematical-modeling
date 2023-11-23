# Fourier-Transform-and-its-applications-in-mathematical-modeling

## Discrete Fourier Transform

### DFT definition

Let 

$$
k, n \in \\{0, 1, \ldots, N-1\\}, \quad N \in \mathbb{N}, \quad j = \sqrt{-1}
$$

DFT is defined as

$$
X_{}^{f}(k) = \sum_{n=0}^{N-1} x(n)W_{N}^{kn}
$$

where **_N_** is amount of data samples as well as **_DFT coefficients_**

$$
\left(a_n\right)_{n \in \\{0, 1, \ldots, N-1\\}}
$$

is a uniformly sampled sequence as to exactly the appropriate interval **_T_** called later `samplingInterval`, **_N_** is called later `samplingRate`

$$
\begin{align*}
W_{N}^{} &= \exp\left(\frac{-j2\pi}{N}\right) \\
\end{align*}
$$

is **_N_**-th root of unity

$$
\begin{align*}
W_{N}^{kn} &= \exp\left(\frac{-j2\pi}{N}kn\right) \\
\end{align*}
$$

and

$$
X_{}^{f}(k)
$$

is the **_k_**-th **_DFT coefficient_**

