---
title: Heaps
date: 2025-07-16
---

Heaps come in 2 forms:

1. Min-heap
2. Max-heap

# What is a Heap?

A heap is a tree-based data structure which satisfies the following **heap property**:

- In a max heap, for any node $P$, the value of $P$ is greater than or equal to any of its children (The largest element is the root of the heap).
- In a min heap, for any node $P$, the value of $P$ is smaller than or equal to any of its children (The smallest element is the root of the heap).

For example: This is a min-heap. Note how the smallest element is on top, and for every node, the value of its children is always larger.

```
       2
     /   \
    4     7
   / \   / \
  8  10 9  12
```

Every heap has the following characteristics:

1. A heap is a complete binary tree - all the levels are fully filled, except possibly the last. The last level is filled from left to right.
2. Heaps allow fast $O(1)$ access to the minimum (for a min-heap) and the maximum (for a max-heap).

On the heap, we can perform:

1. Insertion: Adding a new element while maintaining the heap property in $O(\log n)$ time.
2. Getting the minimum/maximum element in the heap: This takes $O(1)$ time.
3. Removing the minimum/maximum element in the heap while maintaining the heap property: $O(\log n)$ time.

# Implementation of a Heap

Typically, a heap is implemented as an array instead of a tree.

- For any node at index `i`:
    - The left child is at index `2i + 1`
    - The right child is at index `2i + 2`
    - The parent is at index `(i - 1) // 2`

Let's implement a heap from scratch in python. We need to support the following methods:

1. `peek()`: Look at the smallest value in the heap without removing it.
2. `insert(value)`: Insert a new value to the heap.
   - This requires an additional method `heapify_up` to help maintain the heap property.
3. `pop()`: Remove and return the smallest value from the heap. This requires us to **re-heapify** the heap to maintain the heap property.
    - This requires an additional method `heapify_down` to help maintain the heap property.

```py
class MinHeap:
    def __init__(self):
        """
        Initialize an empty min-heap using a list.
        """
        self.heap = [] # Keep track of the elements in the heap

    def peek(self):
        """
        Return the smallest element in the heap without removing it.

        Returns:
            The smallest element, or None if the heap is empty.
        """
        if not self.heap: # If there is nothing in the heap, do nothing
            return None

        return self.heap[0] # Return the root of the heap, which is the smallest element

    def insert(self, value):
        """
        Insert a new value into the heap and maintain the heap property.

        Args:
            value: The value to insert.
        """
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        """
        Remove and return the smallest element from the heap.

        Returns:
            The smallest element, or None if the heap is empty.
        """
        if not self.heap: # If there is nothing in the heap, do nothing
            return None

        if len(self.heap) == 1:
            return self.heap.pop()

        min_value = self.heap[0] # Store the original minimum value
        self.heap[0] = self.heap.pop() # Remove the last element, and place it in the root
        self._heapify_down(0) # Fix the heap

        return min_value # Return the original minimum value

    def _heapify_up(self, index):
        """
        Restore the heap property going up from the given index.

        Args:
            index: The index of the element to heapify up.
        """
        parent_index = (index - 1) // 2 # Find the parent in the heap
        # If the index is valid, and the value of the current node is smaller than the parent, we will need to re-heapify to maintain the heap property
        if index > 0 and self.heap[index] < self.heap[parent_index]:
            self._swap(index, parent_index) # We swap the parent and the child
            self._heapify_up(parent_index) # And then recursively heapify from the `parent_index`

    def _heapify_down(self, index):
        """
        Restore the heap property going down from the given index.

        Args:
            index: The index of the element to heapify down.
        """
        smallest = index # Keep track of the smallest element between the current node and its 2 children. Assume the original node is the smallest
        left = 2 * index + 1 # Get the left child's index
        right = 2 * index + 2 # Get the right child's index
        size = len(self.heap)

        # Find which element is the smallest
        if left < size and self.heap[left] < self.heap[smallest]:
            smallest = left

        if right < size and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != index: # If the original node is not the smallest
            self._swap(index, smallest) # Swap the original node with the smallest node
            self._heapify_down(smallest) # Recursively fix the heap downwards

    def _swap(self, i, j):
        """
        Swap the elements at indices i and j in the heap.

        Args:
            i: Index of the first element.
            j: Index of the second element.
        """
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
```

And an example usage of the `MinHeap`:

```py
heap = MinHeap()

heap.insert(5)
heap.insert(3)
heap.insert(8)
heap.insert(1)

print("Peek:", heap.peek())          # Should print: 1 (smallest element)
print("Extracted:", heap.pop())  # Should remove and return: 1
print("Peek after extract:", heap.peek())  # Should print: 3

heap.insert(2)
print("Extracted:", heap.pop())  # Should remove and return: 2

```

Let's explain how each of the methods work:

## `__init__`

This method initialises the heap.

```py
def __init__(self):
    self.heap = [] # Keep track of the elements in the heap
```

## `peek`

`peek` allows the user to get the smallest value in the heap, without removing anything from the heap. If the heap is empty, return `None`.

```py
def peek(self):
    if not self.heap: # Nothing in the heap
        return None

    return self.heap[0] # Return the smallest value in the heap
```

## `insert`

Insert an element into the heap while maintaining its heap property.

```py
def insert(self, value):
    self.heap.append(value)
    self._heapify_up(len(self.heap) - 1)
```

- We append `value` to the end of the list.
- And then we call `_heapify_up` to restore the heap property, by comparing the inserted value with its parent, and swapping upwards as needed.

## `_heapify_up`

This is one of the helper functions to help maintain the heap property. Starting from `index`, we move up the heap and swap as necessary.

```py
def _heapify_up(self, index):
    parent_index = (index - 1) // 2 # Find the parent in the heap
    # If the index is valid, and the value of the current node is smaller than the parent, we will need to re-heapify to maintain the heap property
    if index > 0 and self.heap[index] < self.heap[parent_index]:
        self._swap(index, parent_index) # We swap the parent and the child
        self._heapify_up(parent_index) # And then recursively heapify from the `parent_index`
```

1. From the node at `index`, we find the index of the parent `parent_index`.
2. If `index == 0`, we have reached the top of the tree, and do not need to continue. If `index < 0`, then we are out of bounds, and do not need to continue as well.
3. We also check if `self.heap[index] < self.heap[parent_index]`. If this is true, then we know that the child is smaller than the parent, which breaks our min-heap's heap property. We then swap the values between the child and the parent, and then recursively `_heapify_up` on the parent, to fix the heap all the way up.

A visualisation of how `insert` and `_heapify_up` work:

```txt
| Step | Array            | Tree Structure                       | Action Description                         |
|------|------------------|--------------------------------------|--------------------------------------------|
| 0    | []               | (empty)                              | Start with an empty heap                   |
| 1    | [5]              |         5                            | Insert 5                                   |
|      |                  |                                      | No heapify needed (it's the root)          |
| 2    | [5, 3]           |         5                            | Insert 3                                   |
|      |                  |       /                              |                                            |
|      |                  |      3                               | 3 < 5 → swap with parent                   |
| 3    | [3, 5]           |         3                            | After heapify_up                           |
|      |                  |       /                              |                                            |
|      |                  |      5                               |                                            |
| 4    | [3, 5, 8]        |         3                            | Insert 8                                   |
|      |                  |       /   \                          |                                            |
|      |                  |      5     8                         | 8 > 3 → no swap needed                     |
| 5    | [3, 5, 8, 1]     |         3                            | Insert 1                                   |
|      |                  |       /   \                          |                                            |
|      |                  |      5     8                         |                                            |
|      |                  |     /                                |                                            |
|      |                  |    1                                 | 1 < 5 → swap                               |
| 6    | [3, 1, 8, 5]     |         3                            | After first swap (1 < 5)                   |
|      |                  |       /   \                          |                                            |
|      |                  |      1     8                         |                                            |
|      |                  |     /                                |                                            |
|      |                  |    5                                 | 1 < 3 → swap again                         |
| 7    | [1, 3, 8, 5]     |         1                            | Final heap after heapify_up                |
|      |                  |       /   \                          |                                            |
|      |                  |      3     8                         |                                            |
|      |                  |     /                                |                                            |
|      |                  |    5                                 |                                            |
```

Regarding time complexity, we can see that the number of operations performed in an `insert` is at most the height of the tree (starting from the end of the tree, we `_heapify_up` until the top of the tree). This shows that the time complexity for `insert` is $O(\log N)$.

## `pop`

`pop` will remove the minimum node from the min-heap, and after that, rearrange the heap to maintain the heap property.

```py
def pop(self):
    if not self.heap: # If there is nothing in the heap, do nothing
        return None

    if len(self.heap) == 1:
        return self.heap.pop()

    min_value = self.heap[0] # Store the original minimum value
    self.heap[0] = self.heap.pop() # Remove the last element, and place it in the root
    self._heapify_down(0) # Fix the heap

    return min_value # Return the original minimum value
```

1. If the heap is empty, return `None`.
2. If the heap only has 1 element, we can directly remove it, and return the value.
3. If the heap has more than 1 element, we store the original value first in `min_value`. We remove the last element from the heap, place it in the root position, and then `_heapify_down` from the root to maintain the heap property.

## `_heapify_down`

`_heapify_down` maintains the heap property by fixing it from the root of the tree, and then recursively goes down the tree.

```py
def _heapify_down(self, index):
    smallest = index # Keep track of the smallest element between the current node and its 2 children. Assume the original node is the smallest
    left = 2 * index + 1 # Get the left child's index
    right = 2 * index + 2 # Get the right child's index
    size = len(self.heap)

    # Find which element is the smallest
    if left < size and self.heap[left] < self.heap[smallest]:
        smallest = left

    if right < size and self.heap[right] < self.heap[smallest]:
        smallest = right

    if smallest != index: # If the original node is not the smallest
        self._swap(index, smallest) # Swap the original node with the smallest node
        self._heapify_down(smallest) # Recursively fix the heap downwards
```

1. Use `smallest` to keep track of the index of the smallest node.
2. `left` is the index of the left child, while `right` is the index of the right child.
3. Check if `left` or `right` is smaller than the original node, and update `smallest` if necessary.
4. If the original node is not the smallest, swap the original node's value with the smallest node, and then recursively `_heapify_down`.

A visualisation of how `pop` and `_heapify_down` work:

```
| Step | Array            | Tree Structure                       | Action Description                         |
|------|------------------|--------------------------------------|--------------------------------------------|
| 0    | [1, 3, 8, 5, 10] |         1                            | Start: Initial heap                        |
|      |                  |       /   \                          |                                            |
|      |                  |      3     8                         |                                            |
|      |                  |     / \                              |                                            |
|      |                  |    5  10                             |                                            |
| 1    | [10, 3, 8, 5]    |         10                           | Move last element to root (remove 1)       |
|      |                  |       /   \                          |                                            |
|      |                  |      3     8                         |                                            |
|      |                  |     /                                |                                            |
|      |                  |    5                                 |                                            |
| 2    | [3, 10, 8, 5]    |         3                            | Swap 10 and 3                              |
|      |                  |       /   \                          |                                            |
|      |                  |     10     8                         |                                            |
|      |                  |     /                                |                                            |
|      |                  |    5                                 |                                            |
| 3    | [3, 5, 8, 10]    |         3                            | Swap 10 and 5                              |
|      |                  |       /   \                          |                                            |
|      |                  |      5     8                         |                                            |
|      |                  |     /                                |                                            |
|      |                  |   10                                 |                                            |
```

From this, we can see that `pop` has an $O(\log N)$ time complexity, because `_heapify_down` would at most go down the entire height of the tree.

# Max Heap

Here is an equivalent `MaxHeap` instead:

```py
class MaxHeap:
    def __init__(self):
        self.heap = []  # Keep track of the elements in the heap

    def peek(self):
        if not self.heap:  # If there is nothing in the heap, do nothing
            return None

        return self.heap[0]  # Return the root of the heap, which is the largest element

    def insert(self, value):
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        if not self.heap:  # If there is nothing in the heap, do nothing
            return None

        if len(self.heap) == 1:
            return self.heap.pop()

        max_value = self.heap[0]  # Store the original maximum value
        self.heap[0] = self.heap.pop()  # Remove the last element and place it in the root
        self._heapify_down(0)  # Fix the heap

        return max_value  # Return the original maximum value

    def _heapify_up(self, index):
        parent_index = (index - 1) // 2  # Find the parent in the heap
        # If the index is valid, and the current node is greater than the parent, we need to fix the heap
        if index > 0 and self.heap[index] > self.heap[parent_index]:
            self._swap(index, parent_index)  # Swap the child and the parent
            self._heapify_up(parent_index)  # Recursively heapify from the parent

    def _heapify_down(self, index):
        largest = index  # Assume the current node is the largest
        left = 2 * index + 1  # Get the left child's index
        right = 2 * index + 2  # Get the right child's index
        size = len(self.heap)

        # Find which element is the largest
        if left < size and self.heap[left] > self.heap[largest]:
            largest = left

        if right < size and self.heap[right] > self.heap[largest]:
            largest = right

        if largest != index:  # If the current node is not the largest
            self._swap(index, largest)  # Swap with the largest child
            self._heapify_down(largest)  # Recursively fix the heap downwards

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
```

And an example usage:

```py
heap = MaxHeap()
heap.insert(5)
heap.insert(3)
heap.insert(8)
heap.insert(1)

print(heap.peek())     # Output: 8 (largest)
print(heap.pop())      # Output: 8
print(heap.peek())     # Output: 5
```

# Use Cases of a Heap

1. Priority queues: A queue which is sorted based on some property (e.g. in Dijkstra's algorithm, where the node closest to the target is at the front of the queue).
2. Scheduling: Task scheduling with priorities put the highest-priority task at the front, allowing for easy access.
3. Heap sort: Sorting algorithm based on the heap structure.

# Python's built-in `heapq`

Python provides a built-in module `heapq` which implements a binary min-heap using a plain list. Note that python only provides a **min-heap** by default.

```py
import heapq

nums = [5, 3, 8, 1]
heapq.heapify(nums)
print(nums)  # Now a heap: [1, 3, 8, 5]

heapq.heappush(nums, 2)
print(nums)  # [1, 2, 8, 5, 3]

smallest = heapq.heappop(nums)
print(smallest)  # 1
print(nums)      # [2, 3, 8, 5]

result = heapq.heappushpop(nums, 4)
print(result)  # Smallest after insert: 2

result = heapq.heapreplace(nums, 7)
print(result)  # Old smallest element
```

To explain the most commonly used methods from `heapq`:

1. `heapify(list)`: Transforms a regular list into a heap **in-place**. Note how we do not have to reassign `nums` in the example above. We directly call `heapq.heapify(nums)`, and `nums` is now a min-heap.
2. `heappush(heap, item)`: Adds `item` into the heap, maintaining the heap property.
3. `heappop(heap)`: Removes and returns the **smallest** element from `heap`.
4. `heappushpop(heap, item)`: Pushes a new item, then pops and returns the smallest element. This is more efficient than doing `heappush` followed by `heappop`.
5. `heapreplace(heap, item)`: Pops the smallest element, and pushes the new item immediately. This is faster than separate calls, but will always pop, even if the new item is smaller.

One important tip: If you need a **max-heap** when using `heapq`, we can simulate a max-heap by inserting **negative values** instead:

```py
nums = []
heapq.heappush(nums, -5)
heapq.heappush(nums, -1)
heapq.heappush(nums, -10)

max_val = -heapq.heappop(nums)  # 10
```

Do take note, when popping from the heap, we must take the negative again, to get the original value.

## Why is `heappushpop` and `heapreplace` faster than making separate calls?

### 1. `heappushpop(heap, item)` vs `heappush(heap, item)` + `heappop(heap)`

Consider the following:

```py
heapq.heappush(heap, x)
heapq.heappop(heap)
```

This performs

1. A heapify up for `heappush(x)`
2. Followed by a heapify down for `heappop()`

Thus, you have to pay for both operations, each taking $O(\log N)$.

However, instead if we just did:

```py
heapq.heappushpop(heap, x)
```

This is more efficient because:

1. It only inserts if `x` is not smaller than the root.
2. If `x < heap[0]`, it just returns `x` immediately, skipping the heap entirely.
3. It avoids adding `x` then immediately removing it, which would be wasteful.
4. In the average case, it does fewer comparisons and swaps, even though the time complexity is still $O(\log N)$.

### 2. `heapreplace(heap, item)` vs `heappop(heap)` + `heappush(heap, item)`

For `heapreplace`:

```py
heapq.heappop(heap)
heapq.heappush(heap, x)
```

This

1. Removes the root (heapify down)
2. Inserts `x` (heapify up)

That is 2 full operations. However, just using

```py
heapq.heapreplace(heap, x)
```

This

1. Replaces `heap[0] = x` directly
2. Does only one heapify down

## Implementations for `heappushpop` and `heapreplace`

```py
def heappushpop(self, value):
    """
    Push `value` onto the heap, then pop and return the smallest element.
    More efficient than insert() followed by pop() when value is bigger than the root.
    """
    if not self.heap:
        return value  # Nothing to compare with; just return what we would have pushed

    if value <= self.heap[0]: # value is smaller than root, so it would be immediately popped anyway
        return value

    # Replace root with value, then heapify down
    result = self.heap[0]
    self.heap[0] = value
    self._heapify_down(0)
    return result

def heapreplace(self, value):
    """
    Pop and return the smallest element, and push `value` in its place.
    Assumes heap is non-empty.
    """
    if not self.heap:
        raise IndexError("heapreplace(): empty heap")

    min_value = self.heap[0]
    self.heap[0] = value
    self._heapify_down(0)
    return min_value
```

And example usages:

```py
heap = MinHeap()
for val in [3, 5, 7]:
    heap.insert(val)

print("Heap:", heap.heap)  # [3, 5, 7]

print("heappushpop(4):", heap.heappushpop(4))  # Pops 3, pushes 4
print("Heap after heappushpop:", heap.heap)    # [4, 5, 7]

print("heapreplace(10):", heap.heapreplace(10))  # Pops 4, pushes 10
print("Heap after heapreplace:", heap.heap)      # [5, 10, 7]
```

# LeetCode Questions Using Heaps

- [1882. Process Tasks Using Servers
](https://leetcode.com/problems/process-tasks-using-servers/description/)
- [215. Kth Largest Element in an Array
](https://leetcode.com/problems/kth-largest-element-in-an-array/description/?envType=problem-list-v2&envId=heap-priority-queue)
- [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/?envType=problem-list-v2&envId=heap-priority-queue)
- [407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/submissions/1513696090/?envType=problem-list-v2&envId=heap-priority-queue)

# Summary

- A heap is a specialized binary tree, used to efficiently retrieve the **minimum**/**maximum** element.
- There are 2 types of heaps:
    1. Min-Heap: The **smallest** element is at the root (index 0).
    2. Max-Heap: The **largest** element is at the root.
- A heap is implemented with an array, where for an element at index `i`:
    - Left child's index: `2 * i + 1`
    - Right child's index: `2 * i + 2`
    - Parent's index: `(i - 1) // 2`

There are a few operations for a min-heap:
| Operation         | Description                                    | Time Complexity |
| ----------------- | ---------------------------------------------- | --------------- |
| `insert()`        | Add element while maintaining heap property    | $O(\log N)$     |
| `peek()`          | Look at the root (min or max) without removing | $O(1)$          |
| `pop()`           | Remove and return root, reheapify              | $O(\log N)$     |
| `_heapify_up()`   | Restore order after insert                     | $O(\log N)$     |
| `_heapify_down()` | Restore order after removal                    | $O(\log N)$     |
| `heappushpop()`   | Efficient push + pop combined                  | $O(\log N)$     |
| `heapreplace()`   | Efficient pop + push combined                  | $O(\log N)$     |

# Resources

- https://rivea0.github.io/blog/leetcode-meditations-chapter-8-heap-and-priority-queue