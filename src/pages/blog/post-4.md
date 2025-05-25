---
title: KNN Classifiers
description: A small description about K-Nearest-Neighbors
date: 2025-04-04
---

# Instance Based Classifiers

- K-Nearest Neighbors (KNN) Classifier
  - Uses $K$ closest points (nearest neighbors) for performing classification

# KNN Classifiers

KNN is a **lazy learning model**
- Lazy learning: A learning method where generalisation of training data is delayed until a query is made to a system
  - Lazy model training is efficient
  - Classifying unknown test instances is relatively expensive
- Opposed to eager learning, where the system tries to generalise training data before receiving queries
  - Eager models take a long time to train
  - However classifying unknown test instances is efficient, since we already learnt the underlying patterns of the training data

Requires 3 things
1. The set of stored labelled instances
2. Distance metric to compute distance between instances
3. Value of $K$, number of nearest neighbors to retrieve

To classify an unknown instance
- Compute distance to all training instances
- Identify $K$ nearest neighbors
- Use class labels of nearest neighbors to determine class label of unknown instance (e.g. by taking majority vote)

# Distance Metric

Compute distance between 2 data points in a $d$-dimensional space. We use euclidean distance (or $L_2$ norm distance)

$$
d(\bold{x}_i, \bold{x}_j) = \sqrt{\sum_{k=1}^{d} (x_{ik} - x_{jk})^2} = \lVert \bold{x}_i - \bold{x}_j \rVert_2
$$

# Choosing the Value of $K$

- If $K$ is too small, sensitive to small points
- If $K$ is too large, neighborhood may include points from other classes

# Determine Class Label

Determine the class from nearest neighbor list
- Take majority vote of class labels among the $K$-nearest neighbors

Given test data $\bold{x}^*$, majority voting:

$$
y^* = \argmax_c \sum_{(\bold{x}_i, y_i) \in \mathcal{N}_{\bold{x}^*}} I(c = y_i)
$$

- $I$ is the indicator function that returns 1 if its input is true, else 0
- $\mathcal{N}_{\bold{x}^*}$ is the set of nearest neighbors of the test instance $\bold{x}^*$

Every neighbor has the same impact on the classification, which makes the algorithm sensitive to the choice of $K$

## Revised Voting Scheme

Solution: Distance-weighted voting
- The weight of influence of each nearest neighbor $\bold{x}_i$ is according to its distance from the test instance $\bold{x}^*$
  - The smaller the distance, the higher the weight

$$
\begin{aligned}
    w_i &= \frac{1}{d(\bold{x}^*, \bold{x}_i)^2} \\
    y^* &= \argmax_c \sum_{(\bold{x}_i, y_i) \in \mathcal{N}_{\bold{x}^*}} w_i I(c = y_i)
\end{aligned}
$$

## Other Issues

- Scaling issues
  - Features may need to be scaled to prevent distance from being dominated by some features
  - E.g.
    - Height of person may vary from 1.5m to 1.8m
    - Weight may vary from 40kg - 200kg
    - Income of person may vary from 10K to 1M
  - Solution: Normalisation on features of different scales

# Normalisation

## Min-max Normalisation

Min-max normalisation: to $[min_{new}, max_{new}]$

$$
v_{new} = \frac{v_{old} - min_{old}}{max_{old} - min_{old}} (max_{new} - min_{new}) + min_{new}
$$

E.g., incomes range from 12000 to 98000. What is the value for 73600 after normalisation?
- (73600 - 12000)/(98000 - 12000)(1 - 0) + 0 = 0.716

## Standardisation (Z-score normalisation)

$$
v_{new} = \frac{v_{old} - \mu_{old}}{\sigma_{old}}
$$

This makes $\mu_{new} = 0, \sigma_{new} = 1$

E.g. $\mu = 54000, \sigma = 16000$. The value of 73600 after standardisation is
- (73600 - 54000) / 16000 = 1.225