---
title: "Problem Set 1"
date: 2025-08-06
---

## 1. Linear Classifiers (Logistic Regression and GDA)

### Question 1(a)
In lecture we saw the average empirical loss for logistic regression:

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

#### Solution

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

---

### Question 1(c)

Recall that in GDA we model the joint distribution of $(x, y)$ by the following equations:

$$
P(y) =
\begin{cases}
\phi & \text{if } y = 1 \\
1 - \phi & \text{if } y = 0
\end{cases}
$$

$$
P(x|y = 0) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}}
\exp\left( -\frac{1}{2} (x - \mu_0)^T \Sigma^{-1} (x - \mu_0) \right)
$$

$$
P(x|y = 1) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}}
\exp\left( -\frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1) \right)
$$

where $\phi, \mu_0, \mu_1,$ and $\Sigma$ are the parameters of our model.  
Suppose we have already fit $\phi, \mu_0, \mu_1,$ and $\Sigma$, and now want to predict $y$ given a new point $x$.  

To show that GDA results in a classifier that has a linear decision boundary, show the posterior distribution can be written as

$$
P(y = 1 \mid x; \phi, \mu_0, \mu_1, \Sigma)
= \frac{1}{1 + \exp\left(-(\theta^T x + \theta_0)\right)},
$$

where $\theta \in \mathbb{R}^n$ and $\theta_0 \in \mathbb{R}$ are appropriate functions of $\phi, \Sigma, \mu_0,$ and $\mu_1$.

#### Solution

To simplify notation, let $\theta = (\phi, \mu_0, \mu_1, \Sigma)$. By Bayes' Theorem:

$$
P(y = 1 \mid x; \theta) = \frac{P(x \mid y = 1; \theta) P(y = 1)}{P(x \mid y = 0; \theta) P(y = 0) + P(x \mid y = 1; \theta) P(y = 1)}
$$

Solving each term individually:

$$
\begin{align*}
P(x \mid y = 1; \theta) P(y = 1) &= \frac{\phi}{(2\pi)^{n/2} |\Sigma|^{1/2}}
\exp\left( -\frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1) \right) \\
P(x \mid y = 0; \theta) P(y = 0) &= \frac{1 - \phi}{(2\pi)^{n/2} |\Sigma|^{1/2}}
\exp\left( -\frac{1}{2} (x - \mu_0)^T \Sigma^{-1} (x - \mu_0) \right)
\end{align*}
$$

Putting them all together:

$$
\begin{align*}
P(y = 1 \mid x; \theta) &= \frac{\phi \exp\left( -\frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1) \right)}{(1 - \phi) \exp\left( -\frac{1}{2} (x - \mu_0)^T \Sigma^{-1} (x - \mu_0) \right) + \phi \exp\left( -\frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1) \right)} \\
&= \frac{1}{1 + \frac{1-\phi}{\phi} \exp \left( -\frac{1}{2}((x - \mu_0)^T \Sigma^{-1} (x - \mu_0) - (x - \mu_1)^T \Sigma^{-1} (x - \mu_1)) \right)} \\
&= \frac{1}{1 + \frac{1 - \phi}{\phi} \exp \left( - (\mu_1 - \mu_0)^T \Sigma^{-1} x  -\frac{1}{2} \mu_0^T \Sigma^{-1} \mu_0 + \frac{1}{2} \mu_1^T \Sigma^{-1} \mu_1 \right)} \\
&= \frac{1}{1 + \exp \left( - (\mu_1 - \mu_0)^T \Sigma^{-1} x - \frac{1}{2} \mu_0^T \Sigma^{-1} \mu_0 + \frac{1}{2} \mu_1^T \Sigma^{-1} \mu_1 + \log \frac{1 - \phi}{\phi} \right)} \\
&= \frac{1}{1 + \exp \left( -(\theta^T x + \theta_0) \right)}
\end{align*}
$$
- From the first line to the second line, we divide all by the numerator.
- From the second line to the third line, we expand the quadratic difference.
- From the third line to the fourth line, we put $\frac{1 - \phi}{\phi}$ into the $\exp$.

We can see that:

$$
\begin{align*}
\theta &= \Sigma^{-1} (\mu_1 - \mu_0) \\
\theta_0 &= \frac{1}{2} \mu_0^T \Sigma^{-1} \mu_0 - \frac{1}{2} \mu_1^T \Sigma^{-1} \mu_1 + \log \frac{\phi}{1 - \phi}
\end{align*}
$$
- Note: $(\Sigma^{-1})^T = \Sigma^{-1}$ due to it being symmetric

We can see, our decision boundary $P(y = 1 \mid x) = \frac{1}{2}$ occurs when 

$$
\theta^T x + \theta_0 = 0
$$

which is a hyperplane, i.e., linear in $x$.

---

### Question 1(d)

For this part of the problem only, you may assume $n$ (the dimension of $x$) is 1, so that $\Sigma = [\sigma^2]$ is just a real number, and likewise the determinant of $\Sigma$ is given by $|\Sigma| = \sigma^2$.  

Given the dataset, we claim that the maximum likelihood estimates of the parameters are given by

$$
\phi = \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}
$$

$$
\mu_0 = \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 0\} x^{(i)}}
{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 0\}}
$$

$$
\mu_1 = \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\} x^{(i)}}
{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}}
$$

$$
\Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
$$

The log-likelihood of the data is

$$
\ell(\phi, \mu_0, \mu_1, \Sigma)
= \log \prod_{i=1}^m p(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma)
$$

$$
= \log \prod_{i=1}^m p(x^{(i)} \mid y^{(i)}; \mu_0, \mu_1, \Sigma) p(y^{(i)}; \phi).
$$

By maximizing $\ell$ with respect to the four parameters, prove that the maximum likelihood estimates of $\phi, \mu_0, \mu_1,$ and $\Sigma$ are indeed as given in the formulas above. (You may assume that there is at least one positive and one negative example, so that the denominators in the definitions of $\mu_0$ and $\mu_1$ above are non-zero.)

#### Solution

To simplify notation, let $\theta = (\phi, \mu_0, \mu_1, \Sigma)$. Then:

$$
\begin{align*}
\ell(\theta) &= \log \prod_{i=1}^m p(x^{(i)} \mid y^{(i)}; \theta) p(y^{(i)}; \theta) \\
&= \sum_{i=1}^m \log p(x^{(i)} \mid y^{(i)}; \theta) + \log p(y^{(i)}; \theta)
\end{align*}
$$

Looking at the first term:

$$
\begin{align*}
    p(x^{(i)} \mid y^{(i)}; \theta) &= \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}}
\exp\left( -\frac{1}{2} (x - \mu_{y^{(i)}})^T \Sigma^{-1} (x - \mu_{y^{(i)}}) \right) \\
    \log p(x^{(i)} \mid y^{(i)}; \theta) &= -\frac{n}{2} \log (2\pi) - \frac{1}{2} \log |\Sigma| -\frac{1}{2} (x - \mu_{y^{(i)}})^T \Sigma^{-1} (x - \mu_{y^{(i)}})
\end{align*}
$$

Looking at the second term:

$$
\begin{align*}
    p(y^{(i)}; \theta) = \phi^{y^{(i)}} (1 - \phi)^{1 - y^{(i)}} \\
    \log p(y^{(i)}; \theta) = y^{(i)} \log \phi + (1 - y^{(i)}) \log (1 - \phi)
\end{align*}
$$

So the overall log likelihood becomes:

$$
\sum_{i=1}^m \left( -\frac{n}{2} \log (2\pi) - \frac{1}{2} \log |\Sigma| -\frac{1}{2} (x^{(i)} - \mu_{y^{(i)}})^T \Sigma^{-1} (x^{(i)} - \mu_{y^{(i)}}) + y^{(i)} \log \phi + (1 - y^{(i)}) \log (1 - \phi) \right)
$$

We can move the first two terms out of the sum:

$$
-\frac{mn}{2} \log (2\pi) - \frac{m}{2} \log |\Sigma| + \sum_{i=1}^m \left( -\frac{1}{2} (x^{(i)} - \mu_{y^{(i)}})^T \Sigma^{-1} (x^{(i)} - \mu_{y^{(i)}}) + y^{(i)} \log \phi + (1 - y^{(i)}) \log (1 - \phi) \right)
$$

We want differentiate w.r.t. $\phi$. First, we extract out all terms containing $\phi$:

$$
\ell_\phi(\theta) = \sum_{i=1}^m \left( y^{(i)} \log \phi + (1 - y^{(i)}) \log (1 - \phi) \right)
$$

Then differentiate w.r.t. $\phi$:

$$
\nabla_\phi \ell_\phi(\theta) = \sum_{i=1}^m \frac{y^{(i)}}{\phi} - \frac{1 - y^{(i)}}{1 - \phi} = \frac{1}{\phi} \sum_{i=1}^m y^{(i)} - \frac{m}{1- \phi} + \frac{1}{1 - \phi} \sum_{i=1}^m y^{(i)}
$$

Set to 0 and solve:

$$
0 = \frac{1}{\phi} \sum_{i=1}^m y^{(i)} - \frac{m}{1- \phi} + \frac{1}{1 - \phi} \sum_{i=1}^m y^{(i)}
$$

Multiply by $\phi(1-\phi)$:

$$
0 = (1 - \phi) \sum_{i=1}^m y^{(i)} - m \phi + \phi \sum_{i=1}^m y^{(i)} \implies \phi = \frac{1}{m} \sum_{i=1}^m y^{(i)} = \frac{1}{m} \sum_{i=1}^m 1\{y^{(i)} = 1\}
$$

Now we extract out only terms containing $\mu$:

$$
\ell_\mu(\theta) = -\frac{1}{2} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})^T \Sigma^{-1} (x^{(i)} - \mu_{y^{(i)}})
$$

Consider the following:

$$
I_0 = \{i: y^{(i)} = 0\}, I_1 = \{i: y^{(i)} = 1\}
$$
- $|I_0| = n_0$, $|I_1| = n_1$, $m = n_0 + n_1$

Now we can split $\ell_\mu(\theta)$ into 2 groups:

$$
\ell_\mu(\theta) = -\frac{1}{2} \left( \sum_{i \in I_0} (x - \mu_{0})^T \Sigma^{-1} (x - \mu_{0})  + \sum_{i \in I_1} (x - \mu_{1})^T \Sigma^{-1} (x - \mu_{1})\right)
$$

Differentiate w.r.t. $\mu_0$:

$$
\nabla_{\mu_0} \ell_\mu(\theta) = \Sigma^{-1} \sum_{i \in I_0} x^{(i)} - \mu_0 = \Sigma^{-1} \sum_{i \in I_0} x^{(i)} - \Sigma^{-1} n_0 \mu_0
$$

Set $\nabla_{\mu_0} \ell_\mu(\theta) = 0$ and solving for $\mu_0$:

$$
\Sigma^{-1} \sum_{i \in I_0} x^{(i)} - \Sigma^{-1} n_0 \mu_0 = 0 \implies \mu_0 = \frac{1}{n_0} \sum_{i \in I_0} x^{(i)} = \frac{\sum_{i=1}^m 1\{y^{(i)} = 0\}x^{(i)}}{\sum_{i=1}^m 1\{y^{(i)} = 0\}}
$$
- $1\{y^{(i)} = 0\}$ acts as a filter to only include those $x^{(i)}$ which correspond to the $0$ class.

Similarly for $\mu_1$:

$$
\mu_1 = \frac{\sum_{i=1}^m 1\{y^{(i)} = 1\}x^{(i)}}{\sum_{i=1}^m 1\{y^{(i)} = 1\}}
$$

Now we isolate terms only containing $\Sigma$:

$$
\ell_\Sigma(\theta) = - \frac{m}{2} \log |\Sigma| + \sum_{i=1}^m \left( -\frac{1}{2} (x^{(i)} - \mu_{y^{(i)}})^T \Sigma^{-1} (x^{(i)} - \mu_{y^{(i)}}) \right)
$$

Since $\Sigma \in \mathbb{R}$, and $|\Sigma| = \sigma^2$:

$$
\nabla_\Sigma \ell_\Sigma(\theta) = -\frac{m}{2} \Sigma^{-1} + \frac{1}{2} \Sigma^{-1} \left( \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}}) (x^{(i)} - \mu_{y^{(i)}})^T \right ) \Sigma^{-1}
$$

Setting to 0 and solving:

$$
\Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}}) (x^{(i)} - \mu_{y^{(i)}})^T
$$

So

$$
\begin{cases}
    \phi &= \frac{1}{m} \sum_{i=1}^m 1\{y^{(i)} = 1\} \\
    \mu_0 &= \frac{\sum_{i=1}^m 1\{y^{(i)} = 0\}x^{(i)}}{\sum_{i=1}^m 1\{y^{(i)} = 0\}} \\
    \mu_1 &= \frac{\sum_{i=1}^m 1\{y^{(i)} = 1\}x^{(i)}}{\sum_{i=1}^m 1\{y^{(i)} = 1\}} \\
    \Sigma &= \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}}) (x^{(i)} - \mu_{y^{(i)}})^T
\end{cases}
$$

Refer to my notes on [GDA](../../chapter-4/gda) for a more detailed proof