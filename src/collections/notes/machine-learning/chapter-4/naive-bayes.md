---
title: "Chapter 4: Naive Bayes"
date: 2025-08-08
---

We go through Naive Bayes covered in [Lecture 5](https://www.youtube.com/watch?v=nt63k3bfXS0&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=5).

# Naive Bayes

Naive Bayes is a probabilistic classification algorithm, based on Bayes' theorem. It is called *naive* because it assumes that all features (input variables) are conditionally independent given the class label, which is often a simplification, but works surprisingly well in practice.

# Procedure

Given a data point with features $\mathbf{x} = (x_1, ..., x_d)$, Naive Bayes predicts the class $y$ that maximises the posterior probability:

$$
\hat{y} = \argmax_y P(y | \mathbf{x})
$$

Using Bayes' theorem:

$$
P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) P(y)}{P(\mathbf{x})}
$$

Since $P(\mathbf{x})$ is the same for all classes, we focus on the numerator:

$$
P(y \mid \mathbf{x}) \propto P(\mathbf{x} \mid y) P(y)
$$

# Naive Assumption

Naive Bayes assumes that the features are independent given the class label:

$$
P(\mathbf{x} \mid y) = P(\mathbf{x}_1, ..., \mathbf{x}_d \mid y) =  \prod_{i=1}^d P(\mathbf{x}_i \mid y)
$$

which simplifes the calculation, turning the potentially complicated joint probability into a product of simpler conditional probabilities.

# Types of Naive Bayes Models

1. Gaussian Naive Bayes: We assume features follow a Gaussian distribution.
2. Multinomial Naive Bayes: Commonly used for discrete counts like word frequencies in text classification.
3. Bernoulli Naive Bayes: Assumes binary features (e.g. word present or not)

# Usefulness of Naive Bayes

1. Simple to train.
2. Works well with high-dimensional data (like text).
3. Surprisingly effective despite the strong independence assumption.
4. Good baseline model for classification problems.

# Example Problem: Spam Classification

We want to classify an email as:
1. Spam ($y = 1$)
2. Not spam ($y = 0$)

based on the presence of certain words in the email.

## Features

We have a dictionary of $d$ words, which is every word that could possibly appear in an email. So our feature vector $\mathbf{x} \in \mathbb{R}^d$. Each word $w$ has a specific entry in $\mathbf{x}$. 

Visually:

$$
\mathbf{x} = \begin{bmatrix}
    1 \\
    0 \\
    0 \\
    \vdots \\
    1 \\
    \vdots \\
    0
\end{bmatrix}
$$
- The first entry $\mathbf{x}_1$ corresponds to the word "a". If "a" appears in the email, we set $\mathbf{x}_1 = 1$.
- If the word does not appear in the email, its corresponding entry is 0.

The training data may look something like this:

| Email | $\mathbf{x}_1$ | $\mathbf{x}_2$ | ... | $\mathbf{x}_d$ | Label (Spam?) |
| ----- | -------------- | -------------- | --- | -------------- | ------------- |
| 1     | 1              | 0              | ... | 0              | 1 (spam)      |
| 2     | 1              | 0              | ... | 0              | 0 (not spam)  |
| 3     | 0              | 1              | ... | 1              | 1 (spam)      |
| 4     | 0              | 0              | ... | 0              | 0 (not spam)  |


## Model Parameters

We want to model $P(\mathbf{x} | y)$ and $P(y)$. There are $2^d$ possible values of $\mathbf{x}$, which means that if we were to model $\mathbf{x}$ explicitly with a multinomial distribution, we need $2^d - 1$ parameters, which is infeasible.

Hence, we assume that $\mathbf{x}_i$'s are conditionally independent given $y$ (Conditional independence assumption):

$$
    P(\mathbf{x}_1, ..., \mathbf{x}_d | y) = P(\mathbf{x}_1 | y) P(\mathbf{x}_2 | y) \cdots P(\mathbf{x}_d | y) = \prod_{i = 1}^d P(\mathbf{x}_i | y)
$$

Now, the parameters of the model are:

The parameters of the model:

$$
\begin{align*}
\phi_{j | y = 1} &= P(\mathbf{x}_{j = 1} | y = 1) \\
\phi_{j | y = 0} &= P(\mathbf{x}_{j = 1} | y = 0) \\
\phi_y &= P(y = 1)
\end{align*}
$$
- $\phi_{j | y = 1}$: Given that the email is spam $y = 1$, what is the probability of word $j$ appearing? E.g. given that the email is spam, what is the probability of the word "win" appearing?
- $\phi_{j | y = 0}$: Given that the email is not spam $y = 0$, what is the probability of word $j$ appearing? E.g. given that the email is not spam, what is the probability of the word "win" appearing?
- $\phi_y$: What is the probability that an email is spam?

We find the joint likelihood:

$$
\mathcal{L}(\phi_y, \phi_{j | y = 0}, \phi_{j | y = 1}) = \prod_{i = 1}^n P(\mathbf{x}^{(i)}, y^{(i)})
$$

By MLE, we get:

$$
\begin{align*}
    \phi_y &= \frac{1}{n} \sum_{i = 1}^n 1\{ y^{(i)} = 1 \} \\
    \phi_{j | y = 0} &= \frac{\sum_{i = 1}^n 1\{\mathbf{x}^{(i)}_j = 1, y^{(i)} = 0\}}{\sum_{i = 1}^n 1\{y^{(i)} = 0\}} \\
    \phi_{j | y = 1} &= \frac{\sum_{i = 1}^n 1\{\mathbf{x}^{(i)}_j = 1, y^{(i)} = 1\}}{\sum_{i = 1}^n 1\{y^{(i)} = 1\}}
\end{align*}
$$
- $\phi_y$: Probability that email is spam = (number of spam emails) / (total number of emails)
- $\phi_{j | y = 0}$: Probability that word $j$ appears in a non-spam email: (number of non-spam emails with word $j$) / (total number of non-spam emails)
- $\phi_{j | y = 1}$: Probability that word $j$ appears in a spam email: (number of spam emails with word $j$) / (total number of spam emails)

## Prediction

Given some new input $\mathbf{x}$, we can get the probability of whether this email is a spam email:

$$
\begin{align*}
P(y = 1 | \mathbf{x}) &= \frac{P(\mathbf{x} | y = 1) P(y = 1)}{P(\mathbf{x})} \\
&= \frac{\left( \prod_{j = 1}^n P(\mathbf{x}_j | y = 1) \right) P(y = 1)}{\left( \prod_{j = 1}^n P(\mathbf{x}_j | y = 0) \right) P(y = 0) + \left( \prod_{j = 1}^n P(\mathbf{x}_j | y = 1) \right) P(y = 1)}
\end{align*}
$$

and then pick the class with the higher posterior probability.

## Problem with Prediction: Zero Probability

We could possibly have a case where for some $\mathbf{x}_j$, $P(\mathbf{x}_j | y) = 0$. This could occur if the word corresponding to $\mathbf{x}_j$ never appears in our data. 

For example, the word "fr33" may never appear in our dataset, but was part of our dictionary. Then, the probability of this email being a spam email is:

$$
\begin{align*}
P(y = 1 | \mathbf{x}) &=  \frac{\left( \prod_{j = 1}^n P(\mathbf{x}_j | y = 1) \right) P(y = 1)}{\left( \prod_{j = 1}^n P(\mathbf{x}_j | y = 0) \right) P(y = 0) + \left( \prod_{j = 1}^n P(\mathbf{x}_j | y = 1) \right) P(y = 1)} \\
&= \frac{0}{0}
\end{align*}
$$

Similarly, the probability that this email is not a spam email is:

$$
\begin{align*}
P(y = 0 | \mathbf{x}) &=  \frac{\left( \prod_{j = 1}^n P(\mathbf{x}_j | y = 1) \right) P(y = 0)}{\left( \prod_{j = 1}^n P(\mathbf{x}_j | y = 0) \right) P(y = 0) + \left( \prod_{j = 1}^n P(\mathbf{x}_j | y = 1) \right) P(y = 1)} \\
&= \frac{0}{0}
\end{align*}
$$

This is the **zero-frequency** problem. To solve this, we can introduce Laplace Smoothing

### Laplace Smoothing

To fix zero probabilities, we add 1 to each count.

Previously, we defined the probability of a word $j$ appearing $\phi_j$:

$$
\begin{align*}
    \phi_{j | y = 0} &= \frac{\sum_{i = 1}^n 1\{\mathbf{x}^{(i)}_j = 1, y^{(i)} = 0\}}{\sum_{i = 1}^n 1\{y^{(i)} = 0\}} \\
    \phi_{j | y = 1} &= \frac{\sum_{i = 1}^n 1\{\mathbf{x}^{(i)}_j = 1, y^{(i)} = 1\}}{\sum_{i = 1}^n 1\{y^{(i)} = 1\}}
\end{align*}
$$

To perform Laplace smoothing, we add 1 to the numerator, and $k$ (the number of classes) in the denominator:

$$
\begin{align*}
    \phi_{j | y = 0} &= \frac{1 + \sum_{i = 1}^n 1\{\mathbf{x}^{(i)}_j = 1, y^{(i)} = 0\}}{2 + \sum_{i = 1}^n 1\{y^{(i)} = 0\}} \\
    \phi_{j | y = 1} &= \frac{1 + \sum_{i = 1}^n 1\{\mathbf{x}^{(i)}_j = 1, y^{(i)} = 1\}}{2 + \sum_{i = 1}^n 1\{y^{(i)} = 1\}}
\end{align*}
$$

In general, take the problem of estimating the mean of a multinomial random variable $z$ taking values in $\{1, ..., k\}$. We parameterise our multinomial with $\phi_j = P(z = j)$. Given a set of $n$ independent observations $\{z^{(1)}, ..., z^{(n)}\}$, the maximum likelihood estimates are given by:

$$
\phi_j = \frac{1}{n} \sum_{i=1}^n 1\{z^{(i)} = j\}
$$

As we saw previously, if we were to use these maximum likelihood estimates, then some of the $\phi_j$'smight end up as 0, which is a problem. To avoid this, we use **Laplace smoothing**:

$$
\phi_j = \frac{1 + \sum_{i=1}^n 1\{z^{(i)} = j\}}{k + n} 
$$
- Add 1 to the numerator, add $k$ to the denominator, where $k$ is the number of classes.

Note: $\sum_{j=1}^k = 1$ still holds:

$$
\begin{align*}
\sum_{j=1}^k \phi_j &= \sum_{j=1}^k \frac{1 + \sum_{i=1}^n 1\{z^{(i)} = j\}}{k + n} \\
&= \frac{1}{k + n} \left( \sum_{j=1}^k \left( 1 + \sum_{i=1}^n 1\{z^{(i)} = j\} \right) \right) \\
&= \frac{1}{k + n} \left( k + \sum_{j=1}^k \sum_{i=1}^n 1\{z^{(i)} = j\} \right) \\
&= \frac{1}{k + n}(k + n) \\
&= 1
\end{align*}
$$
- $\sum_{i=1}^n 1\{z^{(i)} = j\}$ is the count of all data that belong in class $j$, and we sum up over all classes, which is just the overall count of data points that we have in the dataset ($n$).