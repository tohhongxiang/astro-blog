---
title: "Chapter 3: Softmax for Multi-Class Classification"
date: 2025-08-06
---

Multi-class classification. We have some input $\mathbf{x}^{(i)} \in \mathbb{R}^d$, and we want to predict its corresponding class $y \in \{1, ..., k\}$, where $k$ is the number of classes to predict.

We model the distribution of $y$ according to a multinomial distribution. In this case: $p(y | \mathbf{x}; \theta)$ is a distribution over $k$ possible discrete outcomes, and is thus a multinomial distribution.

Note: A multinomial distriubtion involves $k$ numbers $\phi_1, ..., \phi_k$, which specifies the probability of each of the outcomes. $\sum_{i = 1}^k \phi_i = 1$. 

We now design a parameterised model that outputs $\phi_1, ..., \phi_k$ which satisfies this constraint, given the input $\mathbf{x}$.

We introduce $k$ groups of parameters $\theta_1, ..., \theta_k \in \mathbb{R}^d$. Intuitively, we want $\theta_1^T \mathbf{x}, ..., \theta_k^T \mathbf{x}$ to represent $\phi_1, ..., \phi_k$, which are the probabilities $P(y = 1 | \mathbf{x}; \theta), ..., P(y = k | \mathbf{x}; \theta)$.

However, there are 2 issues:
1. $\theta_j^T \mathbf{x}$ is not necessarily within $[0, 1]$.
2. $\sum_{i = 1}^k \theta_i^T \mathbf{x}$ is not necessarily 1.

Hence, we use the **softmax** function to turn $(\theta_1^T \mathbf{x}, ..., \theta_k^T \mathbf{x})$ into a probability vector with nonnegative entries that sum up to 1.

We define the softmax function $\text{softmax}: \mathbb{R}^k \mapsto \mathbb{R}^k$:

$$
\text{softmax}(t_1, ..., t_k) = \begin{bmatrix}
    \frac{\exp (t_1)}{\sum_{j = 1}^k \exp (t_j)} \\
    \vdots \\
    \frac{\exp (t_k)}{\sum_{j = 1}^k \exp (t_j)} \\
\end{bmatrix}
$$

The inputs to the softmax function ($t_i$) are all logits.
- By definition, the output of the softmax function is a probability vector, whose entries are nonnegative, and sum up to 1.
- We use $\exp(t)$ to make all the values nonnegative.
- We divide by the $\sum_{j = 1}^k \exp (t_j)$ to normalize and ensure that the entries sum up to 1.

Let $(t_1, ..., t_k) = (\theta_1^T \mathbf{x}, ..., \theta_k^T \mathbf{x})$. We apply softmax to $(t_1, ..., t_k)$ and use the outputs as probabilities:

$$
\begin{bmatrix}
    P(y = 1 | \mathbf{x}; \theta) \\ \vdots \\ P(y = k | \mathbf{x}; \theta)
\end{bmatrix} = \text{softmax}(t_1, ..., t_k) = \begin{bmatrix}
    \frac{\exp (t_1)}{\sum_{j = 1}^k \exp (t_j)} \\
    \vdots \\
    \frac{\exp (t_k)}{\sum_{j = 1}^k \exp (t_j)} \\
\end{bmatrix} = \begin{bmatrix} \phi_1 \\ \vdots \\ \phi_k \end{bmatrix}
$$
- For notational convenience, let $\phi_i = \frac{\exp (t_i)}{\sum_{j = 1}^k \exp (t_j)}$

More succinctly:

$$
P(y = i | \mathbf{x}; \theta) = \phi_i =  \frac{\exp (t_i)}{\sum_{j = 1}^k \exp (t_j)} =  \frac{\exp (\theta_i^T \mathbf{x})}{\sum_{j = 1}^k \exp (\theta_j^T \mathbf{x})}
$$

We now compute the negative log-likelihood of a single example $(x, y)$:

$$
- \log p(y | \mathbf{x}; \theta) = -\log \left( \frac{\exp (\theta_y^T \mathbf{x})}{\sum_{j = 1}^k \exp (\theta_j^T \mathbf{x})} \right)
$$

Therefore, the loss function, the negative log-likelihood of the training data, is given as:

$$
\mathcal{l}(\theta) = \sum_{i = 1}^{n} -\log \left( \frac{\exp (\theta_{y^{(i)}}^T \mathbf{x}^{(i)})}{\sum_{j = 1}^k \exp (\theta_j^T \mathbf{x}^{(i)})} \right)
$$

We define the cross-entropy loss $\mathcal{l}_{\text{ce}}: \mathbb{R}^k \times \{1, ..., k\} \mapsto \mathbb{R}_{\geq 0}$, which modularises in the complex equation above:

$$
\mathcal{l}_{\text{ce}}((t_1, ..., t_k), y) = - \log \left( \frac{\exp (t_y)}{\sum_{j = 1}^k \exp(t_j)} \right)
$$

With this notation, we rewrite the loss function:

$$
\mathcal{l}(\theta) = \sum_{i = 1}^{n} \mathcal{l}_{\text{ce}}((\theta_1^T \mathbf{x}^{(i)}, ..., \theta_k^T \mathbf{x}^{(i)}), y)
$$

Conveniently, the cross-entropy loss has a simple gradient. Let $t = (t_1, ..., t_k)$ and recall $\phi_i =  \frac{\exp (t_i)}{\sum_{j = 1}^k \exp (t_j)}$. We can see that:

$$
\begin{align*}
    \frac{\partial \mathcal{l}_{\text{ce}}(t, y)}{\partial t_i} &= - \frac{\partial}{\partial t_i} \log \left( \frac{\exp (t_y)}{\sum_{j = 1}^k \exp(t_j)} \right) \\
    &= - \frac{\partial}{\partial t_i} \log \exp(t_y) + \frac{\partial}{\partial t_i} \log \left( \sum_{j = 1}^k \exp(t_j) \right) \\
    &= - \frac{\partial}{\partial t_i} t_y + \frac{\partial}{\partial t_i} \log \left( \sum_{j = 1}^k \exp(t_j) \right) \\
    &= -1\{y = i\} + \frac{\exp(t_i)}{\sum_{j = 1}^k \exp(t_j)} \\
    &= -1\{y = i\} + \phi_i \\
    &= \phi_i - 1\{y = i\}
\end{align*}
$$
- First, we use the definition of $\mathcal{l}_{\text{ce}}(t, y)$
- We separate the logarithm with $\log \frac{a}{b} = \log a - \log b$
- $\log \exp x = x$
- For the fourth line, If $y = i$, then $\frac{\partial}{\partial t_i} t_y = \frac{\partial}{\partial t_i} t_i = 1$. If $y \neq i$, then $\frac{\partial}{\partial t_i} t_y = 0$. This is denoted by $1{y = i}$, where $1\{ \cdot \}$ is the indicator function.
- For the fourth line's second term: We know that $\frac{\partial}{\partial x} \log f(x) = \frac{f'(x)}{f(x)}$. Here, $f(x) = \sum_{j = 1}^k \exp(t_j)$, and $f'(x) = \exp(t_i)$
- From our previous definition, $\frac{\exp (t_i)}{\sum_{j = 1}^k \exp (t_j)} = \phi_i$

This equation can be written in vectorised notation in the form:

$$
\begin{align*}
\frac{\partial \mathcal{l}_{\text{ce}}(t, y)}{\partial t} &= \begin{bmatrix}
    \frac{\partial \mathcal{l}_{\text{ce}}(t, y)}{\partial t_1} \\
    \vdots \\
    \frac{\partial \mathcal{l}_{\text{ce}}(t, y)}{\partial t_k}
\end{bmatrix} \\
&= \begin{bmatrix} \phi_1 - 1\{y = 1\} \\ \vdots \\ \phi_k - 1\{y = k\} \end{bmatrix} \\
&= \begin{bmatrix} \phi_1 \\ \vdots \\ \phi_k \end{bmatrix} - \begin{bmatrix} 1\{y = 1\} \\ \vdots \\ 1\{y = k\} \end{bmatrix} \\
&= \phi - e_y
\end{align*}
$$
- $e_s \in \mathbb{R}^k$ is the $s$-th natural basis vector, where the $s$-th entry is 1, and all other entries are 0.
- In this case, the entry which corresponds to the $y$-th class is 1, and the rest are 0.

Now, using chain rule:

$$
\frac{\partial \mathcal{l}_{\text{ce}}((\theta_1^T \mathbf{x}, ..., \theta_k^T \mathbf{x}), y)}{\partial \theta_i} = \frac{\partial \mathcal{l}_{\text{ce}}(t, y)}{\partial t_i} \frac{\partial t_i}{\partial \theta_i} = (\phi_i - 1\{y = i\})(\mathbf{x})
$$

Hence, the gradient of the loss with respect to the part of parameter $\theta_i$ is:

$$
\frac{\partial \mathcal{l}(\theta)}{\partial \theta_i} = \sum_{j = 1}^{n} \left( \phi_i^{(j)} - 1 \{ y^{(j)} = i \} \right) \mathbf{x}^{(j)}
$$
- where $\phi_i^{(j)} = \frac{\exp (\theta_i^T \mathbf{x}^{(j)})}{\sum_{s = 1}^k \exp (\theta_s^T \mathbf{x}^{(j)})}$ is the probability that the model predicts item $i$ for example $\mathbf{x}^{(j)}$.

Now we can implement gradient descent to minimise the loss function $\mathcal{l}(\theta)$.

$$
\theta_j \coloneqq \theta_j - \alpha \frac{\partial}{\partial \theta_j} \mathcal{l}(\theta) = \theta_j - \alpha \sum_{i = 1}^{n} \left( \phi_j^{(i)} - 1 \{ y^{(i)} = j \} \right) \mathbf{x}^{(i)}
$$