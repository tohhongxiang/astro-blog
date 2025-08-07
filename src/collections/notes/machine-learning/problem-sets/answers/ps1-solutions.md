---
title: "Problem Set 1"
date: 2025-08-06
---

## 1. Linear Classifiers (Logistic Regression and GDA)

(a) In lecture we saw the average empirical loss for logistic regression:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))
$$

where $y^{(i)} \in {0, 1}$, $h_\theta(x) = g(\theta^T x)$ and $g(z) = 1 / (1 + e^{-z})$.

Find the Hessian $H$ of this function, and show that for any vector $z$, it holds true that

$$
z^T H z \geq 0.
$$

**Hint:** You may want to start by showing that $\sum_i \sum_j z_i x_i x_j z_j = (x^T z)^2 \geq 0$. Recall also that $g'(z) = g(z)(1 - g(z))$.

**Remark:** This is one of the standard ways of showing that the matrix $H$ is positive semi-definite, written "$H \succeq 0$". This implies that $J$ is convex, and has no local minima other than the global one. If you have some other way of showing $H \succeq 0$, you're also welcome to use your method instead of the one above.

We start with:

$$
g(z) = \frac{1}{1 + e^{-z}} \implies g'(z) = \frac{-e^{-z}}{(1 + e^{-z})^2} = g(z)(1 - g(z))
$$

First, we find $\frac{\partial J(\theta)}{\partial \theta_j}$:

$$
\begin{align*}
\frac{\partial J(\theta)}{\partial \theta_j} &= - \frac{1}{m} \sum_{i=1}^{m} \frac{\partial}{\partial \theta_j} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] \\
&= - \frac{1}{m} \sum_{i=1}^{m} \frac{\partial}{\partial \theta_j} \left[ y^{(i)} \log(h_\theta(x^{(i)})) \right] + \frac{\partial}{\partial \theta_j} \left[(1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
\end{align*}
$$

Let $z = \theta^T x$. Then:

$$
\frac{\partial h_\theta(x)}{\partial \theta_j} = \frac{d g(z)}{d z} \cdot \frac{\partial z}{\partial \theta_j} = g(z)(1 - g(z)) \cdot x_j = h_\theta(x)(1 - h_\theta(x)) x_j
$$

And:

$$
\nabla_\theta h_\theta(x) = h_\theta(x)(1 - h_\theta(x)) x
$$

So for the first derivative term:
$$
\begin{align*}
\frac{\partial}{\partial \theta_j} \left[ y^{(i)} \log(h_\theta(x^{(i)})) \right] &= y^{(i)} \frac{1}{h_\theta(x^{(i)})} \frac{\partial}{\partial \theta_j} h_\theta(x^{(i)}) \\
&= y^{(i)} \frac{x^{(i)}_j h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)}))}{h_\theta(x^{(i)})} \\
&= y^{(i)} x_j^{(i)} (1 - h_\theta(x^{(i)}))
\end{align*}
$$

For the second derivative term:

$$
\begin{align*}
\frac{\partial}{\partial \theta_j} \left[(1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] &= (1 - y^{(i)}) \frac{- x^{(i)}_j h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)}))}{1 - h_\theta(x^{(i)})} \\
&= -(1 - y^{(i)}) x_j^{(i)} h_\theta(x^{(i)})
\end{align*}
$$

Substituting back:

$$
\begin{align*}
\frac{\partial J(\theta)}{\partial \theta_j} &=  - \frac{1}{m} \sum_{i=1}^{m} \frac{\partial}{\partial \theta_j} \left[ y^{(i)} \log(h_\theta(x^{(i)})) \right] + \frac{\partial}{\partial \theta_j} \left[(1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] \\
&= - \frac{1}{m} \sum_{i=1}^{m} y^{(i)} x_j^{(i)} (1 - h_\theta(x^{(i)})) - (1 - y^{(i)}) x_j^{(i)} h_\theta(x^{(i)}) \\
&= -\frac{1}{m} \sum_{i=1}^{m} x_j^{(i)} (y^{(i)} (1 - h_\theta(x^{(i)})) - (1 - y^{(i)})h_\theta(x^{(i)})) \\
&= -\frac{1}{m} \sum_{i=1}^{m} x_j^{(i)} (y^{(i)} - h_\theta(x^{(i)}))
\end{align*}
$$

Then:

$$
\nabla_\theta J(\theta) = \begin{bmatrix}
    \frac{\partial}{\partial \theta_1} \\
    \vdots \\
    \frac{\partial}{\partial \theta_d}
\end{bmatrix} = \begin{bmatrix}
    -\frac{1}{m} \sum_{i=1}^{m} x_1^{(i)} (y^{(i)} - h_\theta(x^{(i)})) \\
    \vdots \\
    -\frac{1}{m} \sum_{i=1}^{m} x_d^{(i)} (y^{(i)} - h_\theta(x^{(i)}))
\end{bmatrix} = -\frac{1}{m} \sum_{i = 1}^{m} (y^{(i)} - h_\theta(x^{(i)})) x^{(i)}
$$

Now to find the hessian $\nabla_\theta^2 J(\theta)$:

$$
\begin{align*}
    \nabla_\theta^2 J(\theta) &= - \nabla_\theta \left( \frac{1}{m} \sum_{i = 1}^{m} y^{(i)} x^{(i)} - h_\theta(x^{(i)}) x^{(i)} \right) \\
    &= \frac{1}{m} \sum_{i = 1}^{m} x^{(i)} \nabla_\theta \left( h_\theta(x^{(i)})  \right) \\
    &= \frac{1}{m} \sum_{i = 1}^{m} h_\theta(x^{(i)})(1 - h_\theta(x^{(i)})) x^{(i)} (x^{(i)})^T
\end{align*}
$$
- Recall that the Hessian would be a $\mathbb{R}^{d \times d}$ matrix, so we use $x^{(i)} (x^{(i)})^T$ to fit the shape.

We can actually write this in matrix form. Let's define a few more terms:
1. $X = \begin{bmatrix} (x^{(1)})^T \\ \vdots \\ (x^{(m)})^T \end{bmatrix} \in \mathbb{R}^{m \times d}$
2. $h = \begin{bmatrix} h^{(1)} \\ \vdots \\ h^{(m)} \end{bmatrix}$
3. $D = \operatorname{diag}(h^{(1)}(1 - h^{(1)}), \dots, h^{(m)}(1 - h^{(m)})) \in \mathbb{R}^{m \times m}$

Now we can write the above in a matrix form:

$$
\nabla_\theta^2 J(\theta) = \frac{1}{m} X^T D X
$$

Now, to prove that $\nabla_\theta^2 J(\theta)$ is PSD. Let's first note that for $H = \nabla_\theta^2 J(\theta)$, the $i, j$-th entry is:

$$
H_{ij} = \frac{1}{m} \sum_{k = 1}^{m} h_\theta(x^{(k)})(1 - h_\theta(x^{(k)})) x^{(k)}_i (x^{(k)})_j
$$

$$
\begin{align*}
z^T H z &= \sum_{i} \sum_{j} z_i H_{ij} z_j \\
&= \sum_{i} \sum_{j} z_i \left( \frac{1}{m} \sum_{k = 1}^{m} h_\theta(x^{(k)})(1 - h_\theta(x^{(k)})) x^{(k)}_i (x^{(k)})_j \right) z_j \\
&= \frac{1}{m} \sum_{k = 1}^{m} h_\theta(x^{(k)})(1 - h_\theta(x^{(k)})) \sum_{i} \sum_{j} (z_i x_i^{(k)} x_j^{(k)} z_j)
\end{align*}
$$

Note that this inner sum:

$$
\begin{align*}
\sum_i \sum_j z_i x_i^{(k)} z_j x_j^{(k)} &= \sum_i z_i x_i^{(k)} \sum_j z_j x_j^{(k)} \\
&= \left( \sum_i z_i x_i^{(k)} \right) \left( \sum_j z_j x_j^{(k)} \right) \\
&= (z^T x^{(k)})^2 \geq 0
\end{align*}
$$

We also note: for each $k$:
- $h_\theta(x^{(k)}) \in (0, 1)$ due to the definition of the sigmoid function
- Therefore $h_\theta(x^{(k)})(1 - h_\theta(x^{(k)})) > 0$

Since $\sum_{i, j} (z_i x_i^{(k)} x_j^{(k)} z_j) \geq 0$ and $h_\theta(x^{(k)})(1 - h_\theta(x^{(k)})) > 0$, we get

$$
z^T H z \geq 0 \quad \forall z \in \mathbb{R}^d
$$