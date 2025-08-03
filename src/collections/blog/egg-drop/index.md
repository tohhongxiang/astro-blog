---
title: The Egg Drop Problem
date: 2025-07-03
---

In an interview for a Software Engineering Position at Tiktok, I was asked this question. I found this question really interesting, and would like to write down my thoughts regarding the question as well.

## Problem Statement

Suppose you have two eggs and you want to determine from which floors in a one hundred floor building you can drop an egg such that is doesn't break. You are to determine the **minimum number of attempts** you need in order to find the critical floor in the worst case while using the best strategy.
- If the egg doesn't break at a certain floor, it will not break at any floor below.
- If the eggs break at a certain floor, it will break at any floor above.
- The egg may break at the first floor.
- The egg may not break at the last floor.

## Thought 1: Binary Search

Initially, I thought: "Maybe binary search would give the fastest solution, right? Maybe something like $\log_2(100)$ tries?". So I expanded on the solution a bit:
1. Let's drop the first egg on the 50th floor.
2. If the first egg does not break on the 50th floor, then we know the critical floor would be $ > 50$.
3. If the first egg breaks on the 50th floor, then we know the critical floor would be $< 50$.
4. Assuming the first egg broke on the 50th floor, then I would drop the second egg on the 25th floor.
5. However, if the second egg breaks on the 25th floor, then I would be unable to find that critical floor.

I came to the first realisation: If I broke the first egg at some floor $t$, then I would have to try every floor from $1$ to $t - 1$ so that I would be able to figure out the critical floor.

## Thought 2: Partitioning the floors

So instead of using binary search, let's go through $k$ (where $k \leq 100$) floors everytime with the first egg:
1. Drop the first egg at floor $k$.
2. If the first egg did not break, drop it at floor $2k$, $3k$, etc.
3. When the first egg breaks at some floor $mk$, then I would search from floor $(m - 1)k + 1$ to $mk - 1$ with the second egg, so this second egg would take at most $(mk - 1) - [(m - 1)k + 1] + 1 = k - 1$ times.
4. I would just need to find the value of $k$ that minimises the number of attempts in the worst case, right?

So in the worst case, how many times would I drop the egg? The number of attempts required $n$ in the worst case would be:

$$
n = \left\lceil\frac{100}{k}\right\rceil + k - 1
$$

- The first egg would require $\left\lceil\frac{100}{k}\right\rceil - 1$ drops. If all $\left\lceil\frac{100}{k}\right\rceil$ drops of the first egg did not break the egg, then I would know the critical floor is beyond 100, and I could stop there. So the egg must have broke on the last drop, and every drop before then would have been fine.
- Then the remaining $k - 1$ tries would be for the second egg.

My first instinct was that, this formula was not accurate. If $k \nmid 100$, does this work? Lets try $k = 15$:
- The first egg would be dropped at the following levels: 15, 30, 45, 60, 75, 90, 100
- In the case where the critical floor is 99, the number of attempts would be 7 for the first egg + 9 for the second egg = 16 in total.
- In the case where the critical floor was 89, the number of attempts would be 6 for the first egg + 14 for the second egg = 20.
- $\left\lceil\frac{100}{15}\right\rceil + 15 - 1 = 21$, which was wrong.

I needed to figure out the second term, where $k$ does not nicely divide $100$. I thought maybe this was too much effort, and tried another method instead.

## Thought 3: Putting it Together with Math

Let's put this into more concrete math notation. 

We have a building of $k$ floors where $k \geq 1$. We have $n$ eggs. There exists a floor $k^*$, where $1 \leq k^* \leq k$, such that if we dropped the egg at any level $\leq k^*$, it does not break. If we dropped the egg at any level $\gt k^*$, it breaks. We want to find $f(n, k)$, which gives the minimum number of egg drops in the worst case scenario.

## Thought 4: Starting Simple

Let's start with $n = 1$. We can see that, if we only have 1 egg, we would have to try every single floor from the first one up to the last floor. If we tried to skip any floors, and the egg breaks, we would have failed. Hence, we can conclude:

$$
f(1, k) = k
$$

We can also say, that if $k = 1$, we would only have to drop the egg once. Hence,

$$
f(n, 1) = 1
$$

So, what happens when we add a second egg? Let's drop the first egg at some floor $i$, where $1 \leq i \leq k$. In this situation, there are 2 possible cases:
1. The first egg breaks at floor $i$. Then this would mean our second egg has to use $f(1, i - 1)$
2. The first egg does not break at floor $i$. What would be the number of attempts then?

We seem to be stuck at point (2). Let's reframe the question instead.

## Thought 5: Reframing the Problem

Now, I have $t \geq 0$ tries to find the critical floor, and $n$ eggs. What's the largest amount of floors I can explore with $t$ tries and $n$ eggs?

With 1 egg, I can try at most $t$ times. So:

$$
f(1, t) = t
$$

And, if we had $t = 0$ tries, we can't explore any floors. So:

$$
f(n, 0) = 0
$$

How about 2 eggs? If we dropped the first egg at some floor $i$, then 
1. The first egg does not break. We would have $t - 1$ tries remaining. This means we can explore $f(2, t - 1)$ more floors above.
2. The first egg breaks. The critical floor would be below us, and we would have 1 remaining egg, and $t - 1$ remaining tries. From the previous observations, we know that we can explore at most $f(1, t - 1) = t - 1$ floors with that one egg. Hence, if we had at most $t$ attempts, we should start on the $t$th floor. If the first egg breaks, then explore the first to the $t - 1$th floor until we found our critical floor.

Therefore, the total number of floors we can explore with 2 eggs and $t$ tries would be:
$$
\begin{align*}
f(2, t) &= f(1, t - 1) + 1 + f(2, t - 1) \\
&= t + f(2, t - 1) \\
&= t + (t - 1) + \cdots + 1 + f(2, 0) \\
&= t + (t - 1) + \cdots + 1 \\
&= \frac{t(t + 1)}{2}
\end{align*}
$$

In our particular case of 2 eggs and 100 floors:
- $f(2, 13) = 91$
- $f(2, 14) = 105$

So, we need a minimum of 14 tries for 2 eggs and 100 floors.

Now we know, to find this value $t$, we must solve for the smallest $t$ such that

$$
\frac{t(t + 1)}{2} \geq k
$$

Therefore, for 2 eggs and $k$ floors, the minimum number of attempts in the worst case would be:

$$
t = \left\lceil\frac{-1 + \sqrt{1 + 8k}}{2}\right\rceil
$$

## The General Solution for $n$ Eggs and $t$ tries

Let's go back to the earlier equation which we had for 2 eggs and $t$ floors:

$$
f(2, t) = f(1, t - 1) + 1 + f(2, t - 1)
$$

We want to generalise this. With $n$ eggs and $t$ tries:
1. If the egg breaks, we will be able to explore $f(n - 1, t - 1)$ floors.
2. If the egg does not break, we will be able to explore $f(n, t - 1)$ floors.

Therefore, the total number of floors we can cover would be:
$$
f(n, t) = f(n - 1, t - 1) + 1 + f(n, t - 1)
$$

To find $f$, let's define an auxiliary function $g$, where

$$
g(n, t) = f(n + 1, t) - f(n, t)
$$

We are hoping that some telescopic series comes out, and we can cancel terms out. Expanding $g(n, t)$:

$$
\begin{align*}
g(n, t) &= f(n + 1, t) - f(n, t) \\
&= (f(n, t - 1) + f(n + 1, t - 1) + 1) - (f(n - 1, t - 1) + f(n, t - 1) + 1) \\
&= (f(n + 1, t - 1) - f(n, t - 1)) + (f(n, t - 1) - f(n - 1, t - 1)) \\
g(n, t) &= g(n, t - 1) + g(n - 1, t - 1)
\end{align*}
$$

Actually, this is a recursive identity of binomials that we can use. For $C(n, k) = \binom{n}{k} = \frac{n!}{k!(n - k)!}$:

$$
\begin{align*}
C(n, k) &= C(n - 1, k) + C(n - 1, k - 1) \\
C(n, 0) &= 1 \\
C(n, n) &= 1
\end{align*}
$$

Which is similar to our function $g$:
$$
\begin{align*}
g(n, t) &= g(n, t - 1) + g(n - 1, t - 1)
\end{align*}
$$

So, could we just plug in:

$$
g(n, t) = \binom{t}{n}
$$

We can't fully plug it in, because of a few cases. $\forall n, f(n, 0) = 0$, therefore $\forall n, g(n, 0) = 0$. However, $\binom{0}{0} = 1$. We can simply rewrite $g$:

$$
g(n, t) = \binom{t}{n + 1}
$$

Now we can use our telescopic sum for $f(n, t)$:

$$
\begin{align*}
f(n, t) &= [f(n, t) - f(n - 1, t)] + [f(n - 1, t) - f(n - 2, t)] + \cdots + [f(1, t) - f(0, t)] + f(0, t) \\
&= g(n - 1, t) + g(n - 2, t) + \cdots + g(0, t) + 0 \\
&= \binom{t}{n} + \binom{t}{n - 1} + \cdots + \binom{t}{1} \\
&= \sum_{i = 1}^{n} \binom{t}{i}
\end{align*}
$$

So for $n$ eggs and $t$ tries, we can explore $\sum_{i = 1}^{n} \binom{t}{i}$ floors. To find the minimum number of attempts for $n$ eggs and $k$ floors, we need to get the smallest $t$ such that

$$
\sum_{i = 1}^{n} \binom{t}{i} \geq k
$$

Unfortunately, I don't think there is a closed-form solution for the general case.

# Resources

- https://brilliant.org/wiki/egg-dropping/
- https://spencermortensen.com/articles/egg-problem/