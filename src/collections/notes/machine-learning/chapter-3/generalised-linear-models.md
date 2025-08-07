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

The [exponential family](https://en.wikipedia.org/wiki/Exponential_family) is a class of probability distributions of the form:

$$
P(y; \eta) = b(y) \exp \left( \eta^T T(y) - a(\eta) \right)
$$

- $P(y; \eta)$ is the probability density function of the distribution.
  - $P(y; \eta) \in \mathbb{R}$ is a scalar.
  - It is the probability of observing the data $y$ given by the parameter $\eta$.
  - Note that $\eta$ is not the variable we are observing. We are observing $y$.
- $\eta$ is the **natural parameter** or **canonical parameter** of the distribution.
  - It is a reparameterisation of the original parameters of the distribution to fit the exponential family form.
  - $\eta \in \mathbb{R}^d$, where $d$ is the number of parameters required to define the distribution.
    - If the distribution only needs one number to define it (e.g. for logistic regression, only $\phi$), then $d = 1$
    - If the distribution needs multiple numbers (e.g. for gaussian distribution, we have $\mu, \sigma^2$), then $d > 1$
  - E.g. for the Bernoulli distribution, if $\phi$ is the probability of success, then:

  $$
    \eta = \log \left(\frac{\phi}{1 - \phi}\right)
  $$
  - For a Gaussian with **fixed** variance $\sigma^2$, if $\mu$ is the mean, then:

  $$
    \eta = \mu
  $$
  - When connecting $\eta$ to the parameters of the distribution ($\phi$ for Bernoulli distribution, $\mu$ for Gaussian distribution), this function $g^{-1}$ is called the **canonical link function**:

  $$
    \eta = g^{-1}(\cdot)
  $$
  - The function $g$ that gives the distribution's mean as a function of the natural parameter is called the **canonical response function**

  $$
    \text{E}[T(y); \eta] = g(\eta)
  $$
  - In the context of GLMs, we set:
  $$
    \eta = \theta^T \mathbf{x}
  $$
  - Bringing this together, $\eta$ connects the inputs of a distribution $\mathbf{x}$ to the parameters for the distribution. For example, in a GLM for logistic regression, we model:

  $$
    \eta = \theta^T \mathbf{x} \implies p = g^{-1}(\theta^T \mathbf{x}) = \frac{1}{1 + e^{-\theta^T\mathbf{x}}}
  $$
- $T(y)$ is the sufficient statistic (for the distributions we consider, it will often be the case where $T(y) = y$)
  - $T(y) \in \mathbb{R}^d$, the same dimensions as $\eta$.
  - $T(y)$ captures all information about the parameter $\eta$ that is available in the data $y$ to compute the likelihood.
  - "Sufficient" means you do not need the full data - just $T(y)$ is sufficient to estimate the parameters.
- $a(\eta)$ is the **log-partition function**.
  - $a(\eta) \in \mathbb{R}$ is a scalar
  - $e^{-a(\eta)}$ is a normalisation constant, to make sure that the distribution $P(y; \eta)$ sums/integrates over $y$ to 1

  $$
    \int P(y; \eta) dy = 1
  $$

  We can solve for $a(\eta)$:

  $$
    \begin{align*}
        \int P(y; \eta) dy &= 1 \\
        \int b(y) \exp \left( \eta^T T(y) - a(\eta) \right) dy &= 1 \\
        \exp(-a(\eta)) \int b(y) \exp (\eta^T T(y)) dy &= 1 \\
        \exp(a(\eta)) &= \int b(y) \exp (\eta^T T(y)) dy \\
        a(\eta) &= \log \int b(y) \exp (\eta^T T(y)) dy
    \end{align*}
  $$

  - The derivatives of $a(\eta)$ gives:

  $$
    \nabla a(\eta) = \text{E}[T(y)], \quad \nabla^2 a(\eta) = \text{Cov}(T(y))
  $$

- $b(y)$ is the **base measure**
  - $b(y) \in \mathbb{R}$ is a scalar.
  - It handles parts of the distribution that do not depend on $\eta$.
  - For example: In a poisson distribution:
  $$
    b(y) = \frac{1}{y!}
  $$
  - In a Gaussian distribution:
    $$
        b(y) = \frac{1}{\sqrt{2\pi}}
    $$

## Properties of Exponential Family Distributions

1. $\text{E}[y; \eta] = \frac{\partial}{\partial \eta} a(\eta)$
2. $\text{Var}[y; \eta] = \frac{\partial^2}{\partial \eta^2} a(\eta)$
3. Maximum Likelihood Estimation (MLE) w.r.t. $\eta$ is concave. Therefore the negative log-likelihood (NLL) is convex.

### 1. Why is $\nabla a(\eta) = \text{E}(T(y))$?

We now prove the following:

$$
\nabla a(\eta) = \text{E}_{P(y; \eta)} [T(y)]
$$

Recall that the exponential family is of the form:

$$
P(y; \eta) = b(y) \exp \left( \eta^T T(y) - a(\eta) \right)
$$
- $\eta \in \mathbb{R}^d$: Natural parameter
- $T(y) \in \mathbb{R}^d$: Sufficient statistic
- $a(\eta)$: Log-partition function to ensure normalisation

and $a(\eta)$ can be written as:

$$
a(\eta) = \log \int b(y) \exp (\eta^T T(y)) dy = \log Z(\eta)
$$
- We define $Z(\eta) = \int b(y) \exp (\eta^T T(y)) dy$

Differentiating $a(\eta)$:

$$
\nabla a(\eta) = \nabla \log Z(\eta) = \frac{1}{Z(\eta)} \nabla Z(\eta)
$$

Computing $\nabla Z(\eta)$:

$$
\begin{align*}
\nabla Z(\eta) &= \nabla \int b(y) \exp (\eta^T T(y)) dy \\
&= \int b(y) \nabla \exp \left( \eta^T T(y) \right) dy \\
&= \int b(y) \exp \left( \eta^T T(y) \right) T(y) dy
\end{align*}
$$
- Since we are differentiating with respect to $\eta$, we can move the $\nabla$ inside the integral (If the integrand is smooth in $\eta$, and the integral of its derivative exists and is continuous, by Leibniz's rule, you can interchange the differentiation and integration).
- Using chain rule: $\nabla \exp \left( \eta^T T(y) \right) = \exp \left( \eta^T T(y) \right) T(y)$

Substituting back into $\nabla a(\eta)$:

$$
\begin{align*}
\nabla a(\eta) &= \frac{1}{Z(\eta)} \nabla Z(\eta) \\
&= \frac{1}{Z(\eta)} \int b(y) \exp \left( \eta^T T(y) \right) T(y) dy \\
&= \int \frac{b(y) \exp \left( \eta^T T(y) \right)}{Z(\eta)} T(y) dy \\
&= \int P(y; \eta) T(y) dy \\
&= \text{E}_{P(y; \eta)} [T(y)]
\end{align*}
$$
- Recall that $P(y; \eta) = b(y) \exp \left( \eta^T T(y) - a(\eta) \right) = \frac{b(y) \exp \left( \eta^T T(y) \right)}{e^{a(\eta)}} = \frac{b(y) \exp \left( \eta^T T(y) \right)}{Z(\eta)}$
- Recall the definition of expectation for a continuous random variable: $\text{E}[f(x)] = \int P(x) f(x) dx$

Therefore: 

$$
\nabla a(\eta) = \text{E}_{P(y; \eta)} [T(y)]
$$

### 2. Why is $\nabla^2 a(\eta) = \text{Cov}_{P(y; \eta)}[T(y)]$?

We now prove the identity that:

$$
\nabla^2 a(\eta) = \text{Cov}_{P(y; \eta)}[T(y)]
$$

We start with what we had from the previous section:

$$
\nabla a(\eta) = \text{E}_{P(y; \eta)} [T(y)]
$$

Differentiating both sides with respect to $\eta$:

$$
\nabla^2 a(\eta) = \nabla \left( \int P(y; \eta) T(y) dy \right) = \int \nabla P(y; \eta) T(y)^T dy
$$
- Using Leibniz's rule, we can move the gradient inside the integral.
- We transpose $T(y)$ to ensure that the result is a matrix, because $\nabla^2 a(\eta) \in \mathbb{R}^{d \times d}$
  - $\nabla P(y; \eta) \in \mathbb{R}^d$
  - $T(y)^T \in \mathbb{R}^{1 \times d}$
  - $\nabla P(y; \eta) T(y)^T \in \mathbb{R}^{d \times d}$

Recall from earlier:

$$
P(y; \eta) = \frac{b(y) \exp (\eta^T T(y))}{Z(\eta)} = \exp(\eta^T T(y) - a(\eta)) b(y)
$$

So:

$$
\begin{align*}
\nabla P(y; \eta) &= \nabla \exp(\eta^T T(y) - a(\eta)) b(y) \\
&= \exp(\eta^T T(y) - a(\eta)) b(y) (T(y) - \nabla a(\eta)) \\
&= P(y; \eta) (T(y) - \nabla a(\eta)) \\
&= P(y; \eta) (T(y) - \text{E}[T(y)])
\end{align*}
$$

Therefore:

$$
\begin{align*}
\nabla^2 a(\eta) &= \int \nabla P(y; \eta) T(y)^T dy \\
&= \int P(y; \eta) (T(y) - \text{E}[T(y)])T(y)^T dy
\end{align*}
$$

Now, we want to symmetrise the integrand, so that we can get the covariance matrix out:

$$
\text{Cov}(Z) = \text{E}[(Z - \text{E}[Z])(Z - \text{E}[Z])^T] = \int P(z)(z - \text{E}[z])(z - \text{E}[z])^T dz
$$
- Note: This is similar to the scalar version of variance:
$$
\text{Var}[X] = \int P(x) (x - \text{E}[x])^2 dx
$$

Let's define $\mu = \text{E}[T(y)]$. Now:

$$
\begin{align*}
(T(y) - \mu)T(y)^T &= (T(y) - \mu)(T(y) - \mu + \mu)^T \\
&= (T(y) - \mu)(T(y) - \mu)^T + (T(y) - \mu)\mu^T
\end{align*}
$$

So now:

$$
\begin{align*}
\nabla^2 a(\eta) &= \int P(y; \eta) (T(y) - \text{E}[T(y)])T(y)^T dy \\
&= \int P(y; \eta)\left[(T(y) - \text{E}[T(y)])(T(y) - \text{E}[T(y)])^T + (T(y) - \text{E}[T(y)])\mu^T \right] dy \\
&= \int P(y; \eta)(T(y) - \text{E}[T(y)])(T(y) - \text{E}[T(y)])^T dy + \int P(y; \eta) (T(y) - \text{E}[T(y)])\mu^T dy
\end{align*}
$$

Looking at the second term of the integral:

$$
\begin{align*}
\int P(y; \eta) (T(y) - \text{E}[T(y)])\mu^T dy &= \mu^T\int P(y; \eta) (T(y) - \text{E}[T(y)]) dy \\
&= \mu^T \text{E}[T(y) - \mu] \\
&= 0
\end{align*}
$$

Therefore we can rewrite $\nabla^2 a(\eta)$ as:

$$
\begin{align*}
\nabla^2 a(\eta) &= \int P(y; \eta)(T(y) - \text{E}[T(y)])(T(y) - \text{E}[T(y)])^T dy + \int P(y; \eta) (T(y) - \text{E}[T(y)])\mu^T dy \\
&= \int P(y; \eta)(T(y) - \text{E}[T(y)])(T(y) - \text{E}[T(y)])^T dy \\
&= \text{Cov}(T(y))
\end{align*}
$$

### 3. NLL is Convex

The likelihood is given by:

$$
\mathcal{L}(\theta) = \prod_{i = 1}^{n} P(y^{(i)} | \mathbf{x}^{(i)}; \theta)
$$

And the log-likelihood is given by:

$$
\mathcal{l}(\theta) = \sum_{i = 1}^{n} \log P(y^{(i)} | \mathbf{x}^{(i)}; \theta)
$$

Recall that in the GLM, the conditional distribution of $y$ over $\mathbf{x}$ lies in the exponential family:

$$
P(y ; \eta) = b(y) \exp (\eta T(y) - a(\eta))
$$
- $\eta = \theta^T \mathbf{x}$: Natural parameter
- $T(y)$: Sufficient statistic
- $a(\eta)$: Log-partition function

For a dataset $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$, the negative log-likelihood is:

$$
\begin{align*}
-\mathcal{l}(\theta) &= -\sum_{i = 1}^{n} \log P(y^{(i)} | \mathbf{x}^{(i)}; \theta) \\
&= -\sum_{i = 1}^{n} \log \left( b(y^{(i)}) \exp (\eta^{(i)} T(y^{(i)}) - a(\eta^{(i)})) \right)\\
&= -\sum_{i = 1}^{n} \log \left( b(y^{(i)}) \exp (\theta^T \mathbf{x}^{(i)} T(y^{(i)}) - a(\theta^T \mathbf{x}^{(i)})) \right)\\
&= - \sum_{i = 1}^{n} \left[ \theta^T \mathbf{x}^{(i)} T(y^{(i)}) - a(\theta^T \mathbf{x}^{(i)}) + \log b(y^{(i)})\right] \\
&= \sum_{i = 1}^{n} \left[ a(\theta^T \mathbf{x}^{(i)}) - \theta^T \mathbf{x}^{(i)} T(y^{(i)}) - \log b(y^{(i)}) \right]
\end{align*}
$$

Let's look at each of the terms:
1. $a(\theta^T \mathbf{x})$ is convex in $\theta$

    Recall that the first derivative:

    $$
    \nabla a(\eta) = \text{E}_{P(y; \eta)}[T(y)]
    $$

    and the second derivative:

    $$
    \nabla^2 a(\eta) = \text{Cov}_{P(y; \eta)}[T(y)] \succeq 0
    $$

    The covariance matrix is always positive semi-definite. We now prove that the covariance matrix is always positive semi-definite. 
    
    The definition of the covariance matrix:

    $$
    \Sigma = \text{Cov}[\mathbf{X}] = \text{E}[(\mathbf{X} - \text{E}[\mathbf{X}])(\mathbf{X} - \text{E}[\mathbf{X}])^T] \in \mathbb{R}^{n \times n}
    $$

    To show that $\Sigma$ is positive semidefinite, we must show:
    $$
    \forall \mathbf{z} \in \mathbb{R}^n, \quad \mathbf{z}^T \Sigma \mathbf{z} \geq 0
    $$

    Then:

    $$
    \begin{align*}
    \mathbf{z}^T \Sigma \mathbf{z} &= \mathbf{z}^T \text{E}[(\mathbf{X} - \text{E}[\mathbf{X}])(\mathbf{X} - \text{E}[\mathbf{X}])^T] \mathbf{z} \\
    &= \text{E}[\mathbf{z}^T (\mathbf{X} - \text{E}[\mathbf{X}])(\mathbf{X} - \text{E}[\mathbf{X}])^T \mathbf{z}] \\
    &= \text{E}[\mathbf{z}^T (\mathbf{X} - \text{E}[\mathbf{X}])(\mathbf{z}^T(\mathbf{X} - \text{E}[\mathbf{X}]))^T] \\
    &= \text{E}[(\mathbf{z}^T(\mathbf{X} - \text{E}[\mathbf{X}]))^2]
    \end{align*}
    $$

    This final expectation is the expectation of a square, which is always non-negative:

    $$
     \mathbf{z}^T \Sigma \mathbf{z} = \text{E}[(\mathbf{z}^T(\mathbf{X} - \text{E}[\mathbf{X}]))^2] \geq 0
    $$

    Therefore, the covariance matrix $\Sigma$ is positive semidefinite.

    $$
      \Sigma \succeq 0
    $$

    Since the covariance matrix is always positive semi-definite, $a(\eta)$ is convex.

    Since $\eta^{(i)} = \theta^T \mathbf{x}^{(i)}$ is a linear function, it is both convex and concave. Then, $a(\theta^T \mathbf{x}^{(i)})$ is also convex (When combining an affine (linear) transformation with a convex function, the convexity of the function is maintained.).

2. $-\theta^T \mathbf{x}^{(i)} T(y^{(i)})$ is a linear function of $\theta$. It is both convex and concave, hence it does not break convexity.
3. $\log b(y^{(i)})$ is constant in $\theta$, hence it does not affect the convexity of $-\mathcal{l}(\theta)$.

Hence, the NLL is convex. This means:
1. Convexity ensures gradient descent converges to a global minimum.
2. The are no local minima.
3. This makes training GLMs efficient and theoretically sound.

# Members of the Exponential Family

We will prove that the Gaussian distribution (used in linear regression) and the Bernoulli distribution (used in logistic regression) are part of the exponential family.

## Gaussian Distribution
We now show that the Gaussian distribution belongs to the exponential family. 
- In this case, we assume that $\sigma$ is constant. We assume that $\sigma^2 = 1$.
- Recall the value of $\sigma^2$ had no effect on our final choice of $\theta$ and $h_\theta(x)$. Hence, for simplicity, set $\sigma^2 = 1$.
- If we leave $\sigma^2$ as a variable, the Gaussian distribution can also be shown to be in the exponential family, where $\eta \in \mathbb{R}^2$ and depends on both $\mu$ and $\sigma$. 

$$
\begin{align*}
P(y; \mu) &= \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{(y - \mu)^2}{2}\right) \\
&= \frac{1}{\sqrt{2\pi}} \exp \left(-\frac{y^2}{2}\right) \exp \left( \mu y - \frac{1}{2} \mu^2 \right) \\
&= b(y) \exp \left( \eta^T T(y) - a(\eta) \right)
\end{align*}
$$

We can see that:
1. $b(y) = \frac{1}{\sqrt{2\pi}} \exp \left(-\frac{y^2}{2}\right)$
2. $T(y) = y$
3. $\eta = \mu$
4. $a(\eta) = \frac{\mu^2}{2} = \frac{\eta^2}{2}$

Hence, the Gaussian distribution is a member of the exponential family.

## Bernoulli Distribution

We show that the Bernoulli distribution (used for binary data) belongs to the exponential family. Let $\phi$ be the probability of an event.

$$
\begin{align*}
P(y; \phi) &= \phi^y (1 - \phi)^{1 - y} \\
&= \exp \left( \log \left( \phi^y (1 - \phi)^{1 - y} \right) \right) \\
&= \exp \left( \log \left( \frac{\phi}{1 - \phi} \right) y + \log (1 - \phi) \right) \\
&= b(y) \exp \left( \eta^T T(y) - a(\eta) \right)
\end{align*}
$$

We can see that:
1. $b(y) = 1$
2. $\eta = \log \left( \frac{\phi}{1 - \phi} \right) \implies \phi = \frac{1}{1 + e^{-\eta}}$
3. $T(y) = y$
4. $a(\eta) = -\log(1 - \phi) = -\log \left( 1 - \frac{1}{1 + e^{-\eta}} \right) = \log(1 + e^\eta)$

Hence, the Bernoulli distribution belongs to the exponential family.

# Generalised Linear Models (GLMs)

A generalised linear model (GLM) is a generalisation of the ordinary linear regression which allows for:
1. Response variables that have error distribution models other than a normal distribution (e.g. binomal, poisson, etc.).
2. A non-linear relationship between the mean of the response and the predictors via a **link function**.

## Constructing GLMs

A GLM consists of 3 components:
1. Random component: The distribution of the response variable $y$ should be from the exponential family. I.e.:
    $$
    y | \mathbf{x}; \theta \sim \text{ExponentialFamily}(\eta)
    $$
2. Systematic component: The linear predictor: 
    $$
        \eta = \theta^T \mathbf{x}
    $$
    - $\mathbf{x} \in \mathbb{R}^n$ are our inputs.
    - $\theta \in \mathbb{R}^n$ is what the model tries to learn.
3. Link function: A function $g^{-1}(\cdot)$, which connects the mean of the response $\mu = \text{E}[y | \mathbf{x}; \theta]$ to the linear predictor:
    $$
        g^{-1}(\mu) = \eta \quad \text{or} \quad \mu = g(\eta)
    $$
    - $g$ is the **canonical response function**, where $\text{E}[y; \eta] = \mu = g(\eta)$
    - $g^{-1}$ is the **canonical link function**.

So the entire connection looks like:

$$
\theta \xrightarrow[\theta^T \mathbf{x}]{} \eta \xrightarrow[g]{} \phi / \mu, \sigma^2 / \lambda
$$

| Model parameters | Natural Parameters | Canonical Parameters                                               |
| ---------------- | ------------------ | ------------------------------------------------------------------ |
| $\theta$         | $\eta$             | Bernoulli: $\phi$, Gaussian: $(\mu, \sigma^2)$, Poisson: $\lambda$ |

1. Model parameter $\theta$
   - During the training phase of a GLM, we are learning the parameters $\theta$ of the linear model
2. Natural parameter $\eta$:
   - The connection between the model parameter $\theta$ and the natural parameter $\eta$ is linear. I.e. $\eta = \theta^T \mathbf{x}$. Note that this is a design choice. We choose to reparameterise $\eta$ by a linear model.
3. Canonical parameters $\phi, (\mu, \sigma^2), \lambda$
   - The function that connects the natural parameter $\eta$ to the canonical parameters $\phi, (\mu, \sigma^2), \lambda$ is the canonical response function $g(\cdot)$.
   - Conversely, the function that connects the canonical parameters $\phi, (\mu, \sigma^2), \lambda$ to the natural parameter $\eta$ is the link response function $g^{-1}(\cdot)$.

When training our GLM: We want to perform MLE to learn $\hat{\theta}$:

$$
\hat{\theta} = \argmax_\theta \log P(y^{(i)} | \mathbf{x}^{(i)}; \theta)
$$

For GLMs, the update rule is defined as:

$$
\theta_j \coloneqq \theta_j + \alpha \left(y^{(i)} - h_\theta(\mathbf{x}^{(i)}) \right) \mathbf{x}_j
$$

During testing: Run our hypothesis function $h$

$$
h_\theta(\mathbf{x}) = \text{E}[T(y) | \mathbf{x}; \theta] = \text{E}[y | \mathbf{x}; \theta]
$$

## Deriving the Hypothesis Function for GLMs

Using GLMs, we can derive the hypothesis functions used.

In general, for any GLM, the hypothesis function is defined as:

$$
h_\theta(\mathbf{x}) = \text{E}[T(y) | \mathbf{x}; \theta] = \mu = g(\eta) = g(\theta^T \mathbf{x})
$$
- GLMs are designed to predict the mean of the response variable. Hence $h_\theta(\mathbf{x}) = \text{E}[T(y) | \mathbf{x}; \theta]$.
- $\text{E}[T(y) | \mathbf{x}; \theta] = \mu$ by the definition of what a GLM is.

### Linear Regression

We consider that the target variable $y$ is continuous, and we model the conditional distribution of $y$ given $\mathbf{x}$ as a Gaussian $\mathcal{N}(\mu, \sigma^2)$

As we previously saw, the Gaussian is part of the exponential family distribution, where $\mu = \eta$. So:

$$
h_\theta(\mathbf{x}) = \text{E}[y | \mathbf{x}; \theta] = \mu = \eta = \theta^T \mathbf{x}
$$
- $h_\theta(\mathbf{x}) = \text{E}[y | \mathbf{x}; \theta]$ from our assumption that our hypothesis $h(\mathbf{x}) = \text{E}[y | \mathbf{x}]$
- The expected value of a Gaussian distribution is $\text{E}[y | \mathbf{x}] = \mu$
- We saw previously that $\mu = \eta$
- From our third assumption, $\eta = \theta^T \mathbf{x}$

Hence, this gives us our hypothesis function for linear regression:

$$
h_\theta(\mathbf{x}) = \theta^T \mathbf{x}
$$

### Logistic Regression

For logistic classification, we know that $y \in \{0, 1\}$. Given that $y$ is binary-valued, we will use the Bernoulli family of distributions to model the conditional distribution of $y$ give $x$.

In our previous formulation, we saw that the Bernoulli distribution had $\phi = 1 / (1 + e^{-\eta})$. Furthermore, for the Bernoulli distribution, $\text{E}[y | \mathbf{x}; \theta] = \phi$. So, following a similar derivation:

$$
h_\theta(\mathbf{x}) = \text{E}[y | \mathbf{x}; \theta] = \phi = \frac{1}{1 + e^{-\eta}} = \frac{1}{1 + e^{-\theta^T \mathbf{x}}}
$$

Hence, for logistic regression, this gives us our hypothesis function:

$$
h_\theta(\mathbf{x}) = \frac{1}{1 + e^{-\theta^T \mathbf{x}}}
$$


## Deriving the Update Rule for GLMs

Recall that GLMs are based on distributions from the exponential family, which can be written as:

$$
P(y; \eta) = b(y) \exp \left( \eta^T T(y) - a(\eta) \right)
$$

We then define the GLM, assuming:
1. The response $y$ is from the exponential family
2. The natural parameter $\eta$ is linked to the inputs via a linear function:
    $$
    \eta = \theta^T \mathbf{x}
    $$
3. The expected value of $y$ is related to $\eta$ via:

    $$
        \text{E}[y] = \nabla a(\eta) = a'(\eta)
    $$

In our discussions, $\eta$ is a scalar (e.g. Bernoulli, Gaussian, Poisson). For simplicity, we drop the transpose:

$$
P(y; \eta) = b(y) \exp \left[ \eta T(y) - a(\eta) \right]
$$

Now we want to perform maximum likelihood estimation (MLE). So we maximise the log-likelihood of the data point $(\mathbf{x}, y)$:

$$
\begin{align*}
P(y; \eta) &= b(y) \exp \left[ \eta T(y) - a(\eta) \right] \\
\log P(y; \eta) &= \log \left( b(y) \exp \left[ \eta T(y) - a(\eta) \right] \right) \\
&= \log b(y) + \eta T(y) - a(\eta)
\end{align*}
$$
- Recall that our data is generated from a probability distribution $P(y | x; \theta)$, and we want to choose $\theta$ such that it maximises the probability of $\mathbf{x}$ appearing. In MLE terms, we want to find $\hat{\theta}$ such that:

    $$
    \hat{\theta} = \argmax_\theta \prod_{i = 1}^{n} P(y^{(i)} | \mathbf{x}^{(i)}; \theta)
    $$
- However, products are messy to optimise, hence we take the log:
    $$
    \hat{\theta} = \argmax_\theta \sum_{i = 1}^{n} \log P(y^{(i)} | \mathbf{x}^{(i)}; \theta)
    $$
- Also recall that log is monotonic: Maximising the log-likelihood is equivalent to maximising the likelihood.

In a GLM, we assume:
1. $\eta = \theta^T \mathbf{x}$ (Linear in $\theta$)
2. Often $T(y) = y$

So the log-likelihood is now:

$$
\log P(y; \theta) = \log b(y) + \theta^T \mathbf{x} y - a(\theta^T \mathbf{x})
$$
- We replaced $\eta = \theta^T \mathbf{x}$ to reparameterise $P(y; \eta) \to P(y; \theta)$
- We replaced $T(y) = y$

Now we want to compute the gradient of the log-likelihood:

$$
\begin{align*}
\frac{\partial}{\partial \theta_j} \log P(y; \theta) &= \frac{\partial}{\partial \theta_j} \left( \log b(y) + \theta^T \mathbf{x} y - a(\theta^T \mathbf{x}) \right) \\
&= \frac{\partial}{\partial \theta_j} \log b(y) + \frac{\partial}{\partial \theta_j} \theta^T \mathbf{x} y - \frac{\partial}{\partial \theta_j} a(\theta^T \mathbf{x}) \\
&= 0 + \mathbf{x}_j y - a'(\theta^T \mathbf{x}) \mathbf{x}_j \\
&= \mathbf{x}_j y - h_\theta(\mathbf{x}) \mathbf{x}_j \\
&= (y - h_\theta(\mathbf{x})) \mathbf{x}_j
\end{align*} 
$$
- $\frac{\partial}{\partial \theta_j} \log b(y) = 0$, because $\log b(y)$ does not depend on $\theta$
- $\frac{\partial}{\partial \theta_j} \theta^T \mathbf{x} y = \frac{\partial}{\partial \theta_j} \left( y \sum_k \theta_k \mathbf{x}_k \right) = y \mathbf{x}_j$
- $\frac{\partial}{\partial \theta_j} a(\theta^T \mathbf{x}) = a'(\theta^T \mathbf{x}) \mathbf{x}_j$ by chain rule. 
- Recall that $a'(\theta^T \mathbf{x}) = \text{E}[T(y) | \mathbf{x}] = h_\theta(\mathbf{x})$

Now we have the rule:

$$
\frac{\partial}{\partial \theta_j} \log P(y; \theta) = (y - h_\theta(\mathbf{x})) \mathbf{x}_j
$$

Or, for a specific $(\mathbf{x}^{(i)}, y^{(i)})$:

$$
\frac{\partial}{\partial \theta_j} \log P(y^{(i)}; \theta) = (y^{(i)} - h_\theta(\mathbf{x}^{(i)})) \mathbf{x}^{(i)}_j
$$

We can now state our update rule for gradient ascent to maximise our log-likelihood:

$$
\begin{align*}
\theta_j &\coloneqq \theta_j + \alpha \frac{\partial}{\partial \theta_j} \log P(y^{(i)}; \theta) \\
&= \theta_j + \alpha \left( y^{(i)} - h_\theta(\mathbf{x}^{(i)}) \right) \mathbf{x}_j^{(i)}
\end{align*}
$$

# Resources

- https://www.youtube.com/watch?v=xwM9XcnQ4Us
- https://aman.ai/cs229/glm/