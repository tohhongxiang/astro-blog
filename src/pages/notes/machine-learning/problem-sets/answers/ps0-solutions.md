---
title: "Problem Set 0"
date: 2025-07-29
---

# Some Additional Notes

## Differentiating Vectors

If we have a **scalar function** $f: \mathbb{R}^n \mapsto \mathbb{R}$, the gradient of $f$ with respect to $x \in \mathbb{R}^n$ is:

$$
\nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n
$$

If we have a **vector function** $f: \mathbb{R}^n \mapsto \mathbb{R}^m$, i.e.

$$
f(x) = \begin{bmatrix} f_1(x) \\ \vdots \\ f_m(x) \end{bmatrix}
$$

then the **Jacobian** $J_f(x) \in \mathbb{R}^{m \times n}$, where the $(i, j)$ entry is:

$$
    [J_f(x)]_{ij} = \frac{\partial f_i(x)}{\partial x_j}
$$

So,

$$
    J_f(x) = \frac{\partial f}{\partial x} 
    = \begin{bmatrix}
        \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
        \vdots & \ddots & \vdots \\
        \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
    \end{bmatrix} \in \mathbb{R}^{m \times n}
$$

There are a few common derivatives:

| Function                          | Derivative                     |
| --------------------------------- | ------------------------------ |
| $a^T x$ (scalar)                  | $\frac{d}{dx}(a^T x) = a$      |
| $x^T A x$ (scalar, $A$ symmetric) | $\frac{d}{dx}(x^T A x) = 2A x$ |
| $A x$ (vector)                    | $\frac{d}{dx}(A x) = A$        |
| $x^T A$ (row vector)              | $\frac{d}{dx}(x^T A) = A^T$    |

- Differentiating a scalar w.r.t. a vector -> Gradient -> Vector
- Differentiating a vector w.r.t. a vector -> Jacobian -> Matrix
- Differentiating a scalar w.r.t. a matrix -> Matrix of partials

# Questions

## 1. Gradients and Hessians

Recall that a matrix $A \in \mathbb{R}^{n \times n}$ is **symmetric** if $A^T = A$, that is, $A_{ij} = A_{ji}$ for all $i, j$. Also recall the gradient $\nabla f(x)$ of a function $f : \mathbb{R}^n \to \mathbb{R}$, which is the $n$-vector of partial derivatives:

$$
\nabla f(x) =
\begin{bmatrix}
\frac{\partial}{\partial x_1} f(x) \\
\vdots \\
\frac{\partial}{\partial x_n} f(x)
\end{bmatrix}
\quad \text{where} \quad
x =
\begin{bmatrix}
x_1 \\
\vdots \\
x_n
\end{bmatrix}.
$$

The Hessian $\nabla^2 f(x)$ of a function $f : \mathbb{R}^n \to \mathbb{R}$ is the $n \times n$ symmetric matrix of twice partial derivatives,

$$
\nabla^2 f(x) =
\begin{bmatrix}
\frac{\partial^2}{\partial x_1^2} f(x) & \frac{\partial^2}{\partial x_1 \partial x_2} f(x) & \cdots & \frac{\partial^2}{\partial x_1 \partial x_n} f(x) \\
\frac{\partial^2}{\partial x_2 \partial x_1} f(x) & \frac{\partial^2}{\partial x_2^2} f(x) & \cdots & \frac{\partial^2}{\partial x_2 \partial x_n} f(x) \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2}{\partial x_n \partial x_1} f(x) & \frac{\partial^2}{\partial x_n \partial x_2} f(x) & \cdots & \frac{\partial^2}{\partial x_n^2} f(x)
\end{bmatrix}.
$$


1. Let $f(x) = \frac{1}{2} x^T A x + b^T x$, where $A$ is a symmetric matrix and $b \in \mathbb{R}^n$ is a vector. What is $\nabla f(x)$?

    $$
    \begin{align*}
        f(x) &= \frac{1}{2} x^T A x + b^T x \\
        \nabla f(x) &= Ax + b
    \end{align*}
    $$

2. Let $f(x) = g(h(x))$, where $g : \mathbb{R} \to \mathbb{R}$ is differentiable and $h : \mathbb{R}^n \to \mathbb{R}$ is differentiable. What is $\nabla f(x)$?

    $$
    \begin{align*}
        f(x) &= g(h(x)) \\
        \nabla f(x) &= g'(h(x)) \nabla h(x)
    \end{align*}
    $$

3. Let $f(x) = \frac{1}{2} x^T A x + b^T x$, where $A$ is symmetric and $b \in \mathbb{R}^n$ is a vector. What is $\nabla^2 f(x)$?

    $$
    \begin{align*}
        f(x) &= \frac{1}{2} x^T A x + b^T x \\
        \nabla f(x) &= Ax + b \\
        \nabla^2 f(x) &= A
    \end{align*}
    $$

    For $Ax + b$, $Ax, b \in \mathbb{R}^n$. Since $b$ does not contain any terms of $x$, 
    
    $$
    \frac{\partial}{\partial x} b = 0
    $$

    For $Ax$, recall how to differentiate a vector function: If we have some vector function $f: \mathbb{R}^n \mapsto \mathbb{R}^m$, then the **Jacobian** $J_f(x) \in \mathbb{R}^{m \times n}$, where the $(i, j)$ entry is:

    $$
        [J_f(x)]_{ij} = \frac{\partial f_i(x)}{\partial x_j}
    $$

    So,

    $$
        J_f(x) = \frac{\partial f}{\partial x} 
        = \begin{bmatrix}
            \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
            \vdots & \ddots & \vdots \\
            \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
        \end{bmatrix} \in \mathbb{R}^{m \times n}
    $$

    Writing out $Ax$ explicitly:

    $$
        A = \begin{bmatrix}
            a_{11} & a_{12} & \dots & a_{1n} \\
            a_{21} & a_{22} & \dots & a_{2n} \\
            \vdots & \vdots & \ddots & \vdots \\
            a_{m1} & a_{m2} & \dots & a_{mn}
        \end{bmatrix},
        \quad
        x = \begin{bmatrix}
            x_1 \\
            x_2 \\
            \vdots \\
            x_n
        \end{bmatrix},
    $$

    Then:

    $$
    Ax =
    \begin{bmatrix}
    \sum_{j=1}^n a_{1j} x_j \\
    \sum_{j=1}^n a_{2j} x_j \\
    \vdots \\
    \sum_{j=1}^n a_{mj} x_j
    \end{bmatrix}
    $$

    So the $i$th entry of $Ax$ is:

    $$
    (Ax)_i = \sum_{j=1}^n a_{ij} x_j
    $$

    Then:

    $$
    \frac{\partial (Ax)_i}{\partial x_k} = a_{ik}
    $$

    This means:

    $$
        J_{Ax}(x) = \frac{\partial Ax}{\partial x} 
        = \begin{bmatrix}
            \frac{\partial (Ax)_1}{\partial x_1} & \cdots & \frac{\partial (Ax)_1}{\partial x_n} \\
            \vdots & \ddots & \vdots \\
            \frac{\partial (Ax)_n}{\partial x_1} & \cdots & \frac{\partial (Ax)_n}{\partial x_n}
        \end{bmatrix} = \begin{bmatrix}
            a_{11} & \cdots & a_{1n} \\
            \vdots & \ddots & \vdots \\
            a_{n1} & \cdots & a_{nn}
        \end{bmatrix} = A
    $$

    This means that the Jacobian (i.e., the matrix of these derivatives) is just $A$ itself.

4. Let $f(x) = g(a^T x)$, where $g : \mathbb{R} \to \mathbb{R}$ is continuously differentiable and $a \in \mathbb{R}^n$ is a vector. What are $\nabla f(x)$ and $\nabla^2 f(x)$?  

    $$
    \begin{align*}
        f(x) &= g(a^T x) \\
        \nabla f(x) &= g'(a^T x) \cdot a
    \end{align*}
    $$

    To find the Hessian:

    $$
    \begin{align*}
        \nabla^2 f(x) = \nabla \left( g'(a^T x) \cdot a \right)
    \end{align*}
    $$

    Note that:
    1. $g'(a^T x)$ is a scalar function of $x$
    2. $a$ is a constant vector

    Lets look at an easier example: Suppose we have:
    - A scalar function $u(x) = g'(a^T x)$, so $u: \mathbb{R}^n \mapsto \mathbb{R}$
    - A constant vector $v \in \mathbb{R}^n$

    So this expression is:

    $$
        h(x) = u(x) \cdot v
    $$

    We want to compute the Jacobian of $h(x)$. Let's write out $h$ explicitly:

    $$
        h(x) = u(x) \cdot v = \begin{bmatrix}
            u(x) \cdot v_1 \\
            \vdots \\
            u(x) \cdot v_n
        \end{bmatrix}
    $$

    So, each component $h_i(x) = u(x) \cdot v_i$.

    We know this is a **vector function**, so we want to find the Jacobian. For each row $h_i$:

    $$
        \frac{\partial h_i}{\partial x_j} = \frac{\partial}{\partial x_j} (u(x) \cdot v_i) = v_i \cdot \frac{\partial u}{\partial x_j}
    $$

    - $v_i$ is a constant, so it can come outside of the derivative, similar to how $\frac{d}{dx} c f(x) = c \frac{d}{dx} f(x)$

    So the full Jacobian becomes:

    $$
        \frac{\partial h}{\partial x} = \begin{bmatrix}
            v_1 \cdot \nabla u(x)^T \\
            v_2 \cdot \nabla u(x)^T \\
            \vdots \\
            v_n \cdot \nabla u(x)^T
        \end{bmatrix} = \nabla u(x) \cdot v^T
    $$

    - Recall that we want an $n \times n$ matrix
    - Recall that $u: \mathbb{R}^n \mapsto \mathbb{R}$
    - Then, $\nabla u(x) \in \mathbb{R}^n$ is a column vector
    - The Jacobian is an $n \times n$ matrix: Each row is the transpose of $\nabla u(x)$ scaled by $v_i$. We want to coerce it into a cleaner formulat with matrix multiplication, that's why we use $\nabla u(x) \cdot v^T$
    
    Therefore:

    $$
        \nabla (u(x) \cdot v) = \nabla u(x) \cdot v^T
    $$

    So, subsituting things back in:

    $$
    \begin{align*}
    \nabla (g'(a^T x) \cdot a) &= \nabla (u(x) \cdot a) \\ 
    &= \nabla u(x) \cdot a^T \\
    &= g''(a^T x) \cdot a a^T
    \end{align*}
    $$

    - By chain rule, $\nabla g'(a^T x) = g''(a^T x) \cdot a$

Sure! Here's the content from the image converted into Markdown with proper LaTeX delimiters:

---

### 2. \[0 points] Positive definite matrices

A matrix $A \in \mathbb{R}^{n \times n}$ is **positive semi-definite** (PSD), denoted $A \succeq 0$, if $A = A^T$ and $x^T A x \geq 0$ for all $x \in \mathbb{R}^n$. A matrix $A$ is **positive definite**, denoted $A \succ 0$, if $A = A^T$ and $x^T A x > 0$ for all $x \neq 0$, that is, all non-zero vectors $x$. The simplest example of a positive definite matrix is the identity $I$ (the diagonal matrix with 1s on the diagonal and 0s elsewhere), which satisfies

$$
x^T I x = \|x\|_2^2 = \sum_{i=1}^n x_i^2.
$$

## 2. Positive Definite Matrices

1. Let $z \in \mathbb{R}^n$ be an $n$-vector. Show that $A = zz^T$ is positive semidefinite.

    For the first condition:

    $$
    (zz^T)^T = (z^T)^T z^T = zz^T
    $$

    Hence, $A = A^T$ is satisfied

    For the second condition: we can write

    $$
    x^T zz^T x = (z^T x)(z^T x) = (z^T x)^2
    $$
    - We can do this because $z^T x, x^T z \in \mathbb{R}$

    For any number $k \in \mathbb{R}$, $k^2 \geq 0$. Hence, the second condition is proven, and we can conclude that $zz^T \succeq 0$


2. Let $z \in \mathbb{R}^n$ be a non-zero $n$-vector. Let $A = zz^T$. What is the null-space of $A$? What is the rank of $A$?

    The null-space of a matrix $A \in \mathbb{R}^{n \times n}$ is the set of all vectors $x \in \mathbb{R}^n$ such that:

    $$
    Ax = 0
    $$

    So,

    $$
    zz^Tx = 0
    $$

    Since $z$ is a non-zero $n$-vector, we can see that for this to equal 0:, $z^T x = 0$. This occurs if $x$ is orthogonal to $z$.
    - Recall: the dot product (or inner product) of two vectors:
        $$
            x^T z = \lVert x \rVert \lVert z \rVert \cos(\theta)
        $$
    - This is equal to 0 if $\cos(\theta) = 0$, therefore $\theta = 90 \degree$

    Hence, the null-space of $zz^T$ is the set of vectors orthogonal to $z$.

    The rank of a matrix $A$ is the dimension of its column space, i.e., the number of linearly independent columns. Since $A = zz^T$, every column is a scalar multiple of $z$. Hence, there is only 1 linearly independent direction. Therefore, the rank of $zz^T = 1$
    - A set of vectors $v_1, \cdots, v_k$ is linearly independent if none of the vectors can be written as a linear combination of others. 
    - This means, the only solution to the equation:

        $$
            \alpha_1 v_1 + \alpha_2 v_2 + \cdots + \alpha_k v_k = 0
        $$

        is

        $$
            \alpha_1 = \alpha_2 = \cdots = \alpha_k = 0
        $$
    - If even one of the columns can be built using the other columns, then the set is linearly dependent.
    - Since every column of $A$ is just a scalar multiple of $z$ (the $j$-th column is just $z_j z$):

        $$
            zz^T = \begin{bmatrix} z_1 z & z_2 z & \cdots & z_n z \end{bmatrix}
        $$

        All columns lie in the 1-dimensional subspace spanned by $z$
    - Suppose we take a combination:

        $$
            \alpha_1 v_1 + \alpha_2 v_2 + \cdots + \alpha_n v_n = 0
        $$

        Since $v_i = z_i z$:

        $$
            \alpha_1 z_1 z + \alpha_2 z_2 z + \cdots + \alpha_k z_n z = \left( \sum_{i = 1}^n \alpha_i z_i \right) z = 0
        $$

        Since $z > 0$, we can just satisfy

        $$
        \sum_{i = 1}^n \alpha_i z_i = 0
        $$

        and this has infinite solutions.
3. Let $A \in \mathbb{R}^{n \times n}$ be positive semidefinite and $B \in \mathbb{R}^{m \times n}$ be arbitrary, where $m, n \in \mathbb{N}$. Is $BAB^T$ PSD? If so, prove it. If not, give a counterexample with explicit $A, B$.

    We can see that $(BAB^T)^T = BAB^T$, so the first condition is satisfied.

    For the second condition:

    $$
        x^T B A B^T x = (B^T x)^T A (B^T x) = y^T A y
    $$

    $A \succeq 0 \implies y^T A y \geq 0 \implies x^T B A B^T x \geq 0 \implies BAB^T \succeq 0$

    Therefore $BAB^T$ is positive semidefinite.

Here is the content from the image converted into Markdown with LaTeX formatting using `$` and `$$` delimiters:

---

## 3. Eigenvectors, eigenvalues, and the spectral theorem

The eigenvalues of an $n \times n$ matrix $A \in \mathbb{R}^{n \times n}$ are the roots of the characteristic polynomial
$p_A(\lambda) = \det(\lambda I - A)$, which may (in general) be complex. They are also defined as the the values $\lambda \in \mathbb{C}$ for which there exists a vector $x \in \mathbb{C}^n$ such that $Ax = \lambda x$. We call such a pair $(x, \lambda)$ an **eigenvector, eigenvalue pair**. In this question, we use the notation $\text{diag}(\lambda_1, \dots, \lambda_n)$ to denote the diagonal matrix with diagonal entries $\lambda_1, \dots, \lambda_n$, that is,

$$
\text{diag}(\lambda_1, \dots, \lambda_n) =
\begin{bmatrix}
\lambda_1 & 0 & 0 & \cdots & 0 \\
0 & \lambda_2 & 0 & \cdots & 0 \\
0 & 0 & \lambda_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \lambda_n
\end{bmatrix}
$$

1. Suppose that the matrix $A \in \mathbb{R}^{n \times n}$ is diagonalizable, that is, $A = T \Lambda T^{-1}$ for an invertible matrix $T \in \mathbb{R}^{n \times n}$, where $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$ is diagonal. Use the notation $t^{(i)}$ for the $i$-th column of $T$, so that $T = [t^{(1)} \ \dots \ t^{(n)}]$, where $t^{(i)} \in \mathbb{R}^n$. Show that $A t^{(i)} = \lambda_i t^{(i)}$, so that the eigenvalues/eigenvector pairs of $A$ are $(t^{(i)}, \lambda_i)$.

    $$
        At^{(i)} = T \Lambda T^{-1} t^{(i)}
    $$

    Let's look at $T^{-1} t^{(i)}$. Consider the standard basis vector $e_i \in \mathbb{R}^n$, which is all zeros except for a 1 in the $i$-th position.

    Then:

    $$
        Te_i = t^{(i)}
    $$

    So:

    $$
        e_i = T^{-1} t^{(i)}
    $$

    Then:

    $$
        \Lambda e_i = \lambda_i e_i
    $$

    Therefore:
    $$
        At^{(i)} = T \Lambda (T^{-1} t^{(i)}) = T \Lambda e_i = \lambda_i t^{(i)}
    $$

A matrix $U \in \mathbb{R}^{n \times n}$ is **orthogonal** if $U^T U = I$. The spectral theorem, perhaps one of the most important theorems in linear algebra, states that if $A \in \mathbb{R}^{n \times n}$ is symmetric, that is, $A = A^T$, then $A$ is **diagonalizable by a real orthogonal matrix**. That is, there is a diagonal matrix $\Lambda \in \mathbb{R}^{n \times n}$ and orthogonal matrix $U \in \mathbb{R}^{n \times n}$ such that $U^T A U = \Lambda$, or, equivalently,

$$
A = U \Lambda U^T.
$$

Let $\lambda_i = \lambda_i(A)$ denote the $i$th eigenvalue of $A$.

1. Let $A$ be symmetric. Show that if $U = [u^{(1)} \ \dots \ u^{(n)}]$ is orthogonal, where $u^{(i)} \in \mathbb{R}^n$ and $A = U \Lambda U^T$, then $u^{(i)}$ is an eigenvector of $A$ and $A u^{(i)} = \lambda_i u^{(i)}$, where $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$.

    $U$ is orthogonal, which means:

    $$
        U^T U = I \implies U = (U^T)^{-1} \implies U^T = U^{-1}
    $$

    Then:

    $$
        Au^{(i)} = U \Lambda U^T u^{(i)} = U \Lambda U^{-1} u^{(i)}
    $$

    Since $U = \begin{bmatrix} u^{(1)} & u^{(2)} & \cdots & u^{(n)} \end{bmatrix}$:

    $$
        U e_i = u^{(i)} \implies e_i = U^{-1} u^{(i)}
    $$

    Therefore

    $$
        U \Lambda (U^{-1} u^{(i)}) = U \Lambda e_i = U \lambda_i e_i = \lambda_i u^{(i)}
    $$

    Therefore:

    $$
        Au^{(i)} = \lambda_i u^{(i)}
    $$

2. Show that if $A$ is PSD, then $\lambda_i(A) \geq 0$ for each $i$.

    Recall the 2 characteristics for PSD:
    1. $A = A^T$
    2. $\forall x \in \mathbb{R}^n, x^T A x \geq 0$

    From the spectral theorem, $A$ has an orthonormal basis of eigenvectors $u^{(1)}, ..., u^{(n)}$, with corresponding eigenvalues $\lambda_1, ..., \lambda_n$, and

    $$
    Au^{(i)} = \lambda_i u^{(i)}
    $$

    Consider the quadratic form for an eigenvector $u^{(i)}$:

    $$
        (u^{(i)})^T A u^{(i)} = (u^{(i)})^T \lambda_i u^{(i)} = \lambda_i (u^{(i)})^T u^{(i)}
    $$
    - We can reorder because $\lambda_i \in \mathbb{R}$

    Since $u^{(i)}$ is an eigenvector of $A$, $u^{(i)}$ is actually non-zero by definition, hence $(u^{(i)})^T u^{(i)} = \sum_{j=1}^n (u^{(i)}_j)^2 > 0$. Since $A$ is PSD, $(u^{(i)})^T A u^{(i)} \geq 0$. Therefore:

    $$
    \lambda_i = \frac{(u^{(i)})^T A u^{(i)}}{(u^{(i)})^T u^{(i)}} \geq 0
    $$

    Therefore, all the eigenvalues $\lambda_i$ of $A$ are non-negative.

    $$
        \lambda_i(A) \geq 0
    $$



