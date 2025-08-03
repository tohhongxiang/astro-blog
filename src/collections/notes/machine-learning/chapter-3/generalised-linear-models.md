---
title: "Chapter 3: Locally Weighted & Logistic Regression"
date: 2025-08-02
---

This is [lecture 4](https://www.youtube.com/watch?v=iZTeva0WSTQ&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=4) of the Stanford CS229 course.

We go through
- Perceptron
- Exponential Family
- Generalised Linear Models
- Softmax Regression (Multiclass classification)

# Perceptron

The perceptron uses the function:

$$
h_\theta(\mathbf{x}) = g(\theta^T \mathbf{x}), \quad g(z) = \begin{cases}
    1 & z \geq 0 \\
    0 & z < 0
\end{cases}
$$

And uses the same update rule:

$$
\theta_j \coloneqq \theta_j + \alpha \left( y^{(i)} - h_\theta(\mathbf{x}^{(i)})\right) \mathbf{x}_j^{(i)}
$$

# Exponential Family

A class of probability distributions:

$$
P(y; \eta) = b(y) \exp \left[ \eta^T T(y) - a(\eta) \right]
$$
- $y$: Data
- $\eta$: Natural parameter
- $T(y)$: Sufficient statistic. For our discussions, $T(y) = y$.
- $b(y)$: Base measure
- $a(\eta)$: Log partition function. Plays the role of a normalisation constant, to make sure that $P(y; \eta)$ sums/integrates over $y$ to 1.

We show that the Bernoulli distribution (used for binary data) belongs to the exponential family. Let $\phi$ be the probability of an event.

$$
\begin{align*}
P(y; \phi) &= \phi^y (1 - \phi)^{1 - y} \\
&= \exp \left( \log \left( \phi^y (1 - \phi)^{1 - y} \right) \right) \\
&= \exp \left( \log \left( \frac{\phi}{1 - \phi} \right) y + \log (1 - \phi) \right)
\end{align*}
$$

We can see that:
1. $b(y) = 1$
2. $\eta = \log \left( \frac{\phi}{1 - \phi} \right) \implies \phi = \frac{1}{1 - e^{-\eta}}$
3. $T(y) = y$
4. $a(\eta) = -\log(1 - \phi) = -\log \left( 1 - \frac{1}{1 - e^{-\eta}} \right) = \log(1 + e^\eta)$

Hence, the Bernoulli distribution belongs to the exponential family.

We now show that the Gaussian distribution belongs to the exponential family. 
- In this case, we assume that $\sigma$ is constant. We assume that $\sigma^2 = 1$.
- Recall the value of $\sigma^2$ had no effect on our final choice of $\theta$ and $h_\theta(x)$. Hence, for simplicity, set $\sigma^2 = 1$.
- If we leave $\sigma^2$ as a variable, the Gaussian distribution can also be shown to be in the exponential family, where $\eta \in \mathbb{R}^2$ and depends on both $\mu$ and $\sigma$. 

$$
\begin{align*}
P(y; \mu) &= \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{(y - \mu)^2}{2}\right) \\
&= \frac{1}{\sqrt{2\pi}} \exp \left(-\frac{y^2}{2}\right) \exp \left( \mu y - \frac{1}{2} \mu^2 \right)
\end{align*}
$$

We can see that:
1. $b(y) = \frac{1}{\sqrt{2\pi}} \exp \left(-\frac{y^2}{2}\right)$
2. $T(y) = y$
3. $\eta = \mu$
4. $a(\eta) = \frac{\mu^2}{2} = \frac{\eta^2}{2}$

## Properties of Exponential Family

1. Maximum Likelihood Estimation (MLE) w.r.t. $\eta$ is concave. Therefore the negative log-likelihood (NLL) is convex.
2. $\text{E}[y; \eta] = \frac{\partial}{\partial \eta} a(\eta)$
3. $\text{Var}[y; \eta] = \frac{\partial^2}{\partial \eta^2} a(\eta)$

# Generalised Linear Models (GLMs)

