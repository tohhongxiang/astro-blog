---
title: Binary Search
description: An explanation of the Binary Search Algorithm
date: 2025-06-30
---

## Introduction

Binary search is a fast and efficient algorithm for finding an item in a **sorted** list or array. It works by repeatedly dividing the search space in half until the item is found or the search space is empty.

## The Inspiration of Binary Search: "Guess the Number"

Imagine this game:

> I’m thinking of a number between 1 and 100. Try to guess it.

A smart strategy is not to start with 1, then 2, then 3. Instead, most people guess 50. If it's too high, guess 25. If too low, guess 75. Each guess cuts the possible range in half.

This intuitive approach - narrowing down the search space by halves - is the core idea of binary search. Even children play this game naturally without knowing the algorithm’s name.

## How Does Binary Search Work?

The key prerequisite is: **The array must be sorted.**

Why? Because binary search depends on the order to decide which half of the array to discard.

For example, imagine looking up the word "Manuscript" in a dictionary. Since the dictionary is alphabetically sorted, you can open it near the middle and check the letter there. If the middle is "J" and you're looking for "M," you know to search the latter half. If the dictionary wasn’t sorted, you’d have to scan every page.

## Step-by-Step Binary Search

Suppose you have a sorted array:

```py
arr = [1, 3, 5, 7, 9, 11, 13, 15] # sorted array
target = 7
```

The algorithm proceeds as follows:
1. Find the middle element of the current search range.
2. Compare the middle element to the `target`
   - If equal, return the middle index.
   - If the target is smaller, search the left half.
   - If the target is larger, search the right half.
3. We repeat until the target is found, or the search range becomes empty

## Iterative Binary Search In Python

Here is an implementation of binary search in python:

```py
def binary_search(array, target):
    low = 0 # Lower end of the search range, which is initialised at the start of the array
    high = len(array) - 1 # Higher end of the search range, which is initialised at the end of the array

    while low <= high: # While the search range is not empty
        mid = (low + high) // 2  # Get the middle of the array. We make sure this is an integer, because this is an index

        if array[mid] == target:
            return mid  # Target found
        elif array[mid] < target:
            low = mid + 1  # Search the right half
        else:
            high = mid - 1  # Search the left half

    return -1  # Target is not in the array
```

## Recursive Binary Search in Python

As a recursive function, we introduce 2 new parameters `low` and `high` to represent the search range of the current function call:

```py
def binary_search_recursive(arr, target, low, high):
    if low > high:
        return -1  # Base case: target not found

    mid = (low + high) // 2

    if arr[mid] == target: # Found the target
        return mid
    elif arr[mid] < target: # Target should be in the right half of the search range
        return binary_search_recursive(arr, target, mid + 1, high)
    else: # Target should be in the left half of the search range
        return binary_search_recursive(arr, target, low, mid - 1)


arr = [1, 3, 5, 7, 9, 11, 13]
target = 7

index = binary_search_recursive(arr, target, 0, len(arr) - 1)
print(index)  # Output: 3
```
## Complexity Analysis

### Time Complexity

1. Best Case: $O(1)$ - when the target is found on the first try.
2. Average Case: $O(\log n)$
3. Worst Case: $O(\log n)$

Binary search cuts the array size in half with each iteration. For an array of size n, it takes at most $\log_2(n)$ steps to reduce the search space to one element.

### Space Complexity

- Iterative version: $O(1)$ - constant space.
- Recursive version: $O(\log n)$ - due to recursive call stack.

## Common Variants of Binary Search

The classic binary search finds an exact match. But what if the array contains duplicates, and you want to find the first or last occurrence of a target?

### Finding the First Occurrence in a Sorted Array with Duplicates

Given:

```py
arr = [1, 2, 2, 2, 3, 4]
target = 2
```

The classic binary search may return any index where `2` occurs, e.g., `2`. To find the **first** occurrence, we:
- Keep track of the index when we find the target.
- Continue searching the left half to see if an earlier occurrence exists.

```py
def find_first_occurrence(array, target):
    low = 0 # Lower end of the search range, initialised at the start of the array
    high = len(array) - 1 # Upper end of the search range, initialised at the end of the array
    result = -1 # Default to not found

    while low <= high:
        mid = (low + high) // 2 # Find the middle index

        if array[mid] == target:
            result = mid # Found the target
            high = mid - 1 # But we do not stop. We check the lower half to see if an earlier occurrence exists
        elif array[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return result
```

Now we can use our newly created `find_first_occurrence` function:

```py
arr = [1, 2, 2, 2, 3, 4]
print(find_first_occurrence(arr, 2))  # Output: 1
print(find_first_occurrence(arr, 5))  # Output: -1
```

### Finding the Last Occurrence in a Sorted Array with Duplicates

Similarly, if we want to find the **last** occurrence instead:

```py
def find_last_occurrence(array, target):
    low = 0 # Lower end of the search range, initialised at the start of the array
    high = len(array) - 1 # Upper end of the search range, initialised at the end of the array
    result = -1 # Default to not found

    while low <= high:
        mid = (low + high) // 2 # Find the middle index

        if array[mid] == target:
            result = mid # Record the index where the target is found
            low = mid + 1 # But we do not stop. We check the upper half to see if a later occurrence exists
        elif array[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return result
```

And now we can use it

```py
arr = [1, 2, 2, 2, 3, 4]
print(find_last_occurrence(arr, 2))  # Output: 3
print(find_last_occurrence(arr, 5))  # Output: -1
```

### Lower/Upper Bounds

- Lower Bound: Find the first position where the value is **greater than or equal to** the target.
- Upper Bound: Find the first position where the value is **strictly greater than** the target.

Both lower/upper bounds are useful for inserting into a sorted array:
- `lower_bound`: Inserting before duplicates.
- `upper_bound`: Inserting after duplicates.

#### Lower Bound Implementation

```py
def lower_bound(arr, target):
    low, high = 0, len(arr) - 1

    while low < high:
        mid = (low + high) // 2
        if arr[mid] < target: 
            # Everything before `mid` is too small, our lower bound should be on the right
            low = mid + 1
        elif arr[mid] > target:
            # Anything past `mid` is strictly greater than our target, which means our lower bound (the first element >= `target`) 
            # must be before mid
            high = mid - 1
        else: 
            # `arr[mid] == target`. This could possibly be our lower bound, but there might be an earlier occurrence
            # Hence we shrink to continue checking on the left of our search range
            high = mid

    return low
```

- If `arr[mid] < target`, we know `mid` and everything before it are too small
- If `arr[mid] >= target`, this *might* be the answer, but we look left to see if there is an earlier match
  - In the `elif arr[mid] > target` branch, since `arr[mid]` is **strictly greater** than `target`, `mid` cannot be the lower bound index if we are looking for the first element that is `>= target`. Hence, we can safely exclude `mid` by doing `high = mid - 1`
  - However, for the `else` branch, which is the case where `arr[mid] == target`, this means that `mid` could possibly be the first element which is `>= target` (because it is equal to `target`). However, it is also possible that there is an earlier occurrence on the left side. Therefore, **we do not exclude `mid`**. Instead, we set `high = mid` to **continue searching including `mid` itself**.
  - If we were to replace `high = mid` in the else branch with `high = mid - 1`, we might accidentally skip over the first occurrence of the target when it actually might be the correct answer. 
- At the end of the algorithm, `low` will point to the first index where `arr[low] >= target`

A slightly different version of the `lower_bound` function avoid the `elif` branch to simplify into:

```py
def lower_bound(arr, target):
    low, high = 0, len(arr) # Half-open interval

    while low < high:
        mid = (low + high) // 2
        if arr[mid] < target: # lower_bound is strictly on the right of `mid`
            low = mid + 1
        else: # `mid` could be the lower_bound, however there could be a possible earlier occurrence
            high = mid

    return low
```

Here, `high` is intialised to `len(arr)` (one past the last index), and the condition only checks `arr[mid] < target`, which makes the algorithm more elegant and less error-prone

> Why set `high = len(arr)` instead of `high = len(arr) - 1`?

1. This allows the search space to represent the entire range of possible insertion points
   - Think of `low` and `high` as pointers between elements, rather than indices of elements
   - `low` and `high` represent a range `[low, high)` (half-open interval), having `high = len(arr)` means that the search space includes the position after the last element. 
     - A **half-open interval** `[low, high)` is a range of integers that **includes** the lower bound (`low`), but excludes the upper bound (`high`)
     - In other words, it contains all elements starting from `low` **up to, but not including** `high`
   - Imagine the array `[1, 2, 3, 4, 5]`, and the target `10`. This lets the algorithm naturally handle cases where the target is inserted **at the end** of the array (i.e., after the last element)

#### Upper Bound Implementation

For finding the upper bound, we can use binary search to find the first position where the value is **strictly greater than the target**

```py
def upper_bound(arr, target):
    low, high = 0, len(arr) # Half-open serach interval

    while low < high:
        mid = (low + high) // 2
        if arr[mid] <= target: # Upper bound is strictly to the right of mid
            low = mid + 1
        else: # Upper bound could possibly be mid, but there could be an earlier occurrence
            high = mid
    return low
```

- `low = 0`, `high = len(arr)` defines the half-open search interval `[low, high)` to cover all possible insertion points
- If `arr[mid] <= target`, the upper bound must be **strictly to the right** of `mid`, so we set `low = mid + 1` to exclude `mid`
- Else (`arr[mid] > target`), `mid` could be the first element **strictly greater than** `target`, so we keep `mid` in the search by setting `high = mid`
- The loop runs until `low == high`, at which `low` points to the first position where `arr[low] > target`, which is the upper bound

Consider the array:

```py
arr = [1, 3, 5, 5, 5, 7, 9]
```

These are the results for using `lower_bound` and `upper_bound`:

| Target | `lower_bound` | `upper_bound` |
| ------ | ------------- | ------------- |
| 5      | 2             | 5             |
| 4      | 2             | 2             |
| 6      | 5             | 5             |
| 10     | 7 (end)       | 7 (end)       |
| 0      | 0             | 0             |

#### Why do `low` and `high` converge to the same value?

Our loop condition is:

```py
while low < high:
    # ...
```

The loop continues narrowing the range, and when `low == high`, the search ends. At that moment:
- `low` points to the smallest index where the condition (e.g. for lower bound: `arr[i] >= target`, for upper bound: `arr[i] > target`) holds true.
- `high` points to the same index, because the range `[low, high)` has zero length
- We can actually return `high` if we want to as well, since at the end of the function, `low == high`. However, most implementations usually prefer returning `low` because it is commonly used to represent the insertion point. Functionally, it does not matter

## Summary

- Binary search requires a `sorted` array.
- It repeatedly halves the search range, yielding a $O(\log n)$ time complexity.
- Variants such as first/last occurrence, lower/upper bounds are useful for handling duplicates or range queries.

## Leetcode Practice Questions

1. [704. Binary Search](https://leetcode.com/problems/binary-search/)
2. [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/?envType=problem-list-v2&envId=binary-search)
3. [374. Guess Number Higher or Lower](https://leetcode.com/problems/guess-number-higher-or-lower/description/?envType=problem-list-v2&envId=binary-search)
4. [35. Search Insert Position](https://leetcode.com/problems/search-insert-position/description/?envType=problem-list-v2&envId=binary-search)