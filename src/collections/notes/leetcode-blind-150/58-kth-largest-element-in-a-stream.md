# 58. Kth Largest Element in a Stream

https://leetcode.com/problems/kth-largest-element-in-a-stream/description/

Design a class to find the `kth` largest integer in a stream of values, including duplicates. E.g. the `2nd` largest from `[1, 2, 3, 3]` is `3`. The stream is not necessarily sorted.

Implement the following methods:

- `constructor(int k, int[] nums)` Initializes the object given an integer `k` and the stream of integers `nums`.
- `int add(int val)` Adds the integer `val` to the stream and returns the `kth` largest integer in the stream.

# Solution

Key insight: We use a min-heap to store all the numbers. This min-heap will have a size `k`. Everytime the min-heap exceeds the size, we pop from the top of the heap. Then, the top of the heap will be the `k`-th largest number.

```py
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.capacity = k

        self.nums = nums
        heapq.heapify(self.nums)

        while len(self.nums) > self.capacity:
            heapq.heappop(self.nums)

    def add(self, val: int) -> int:
        heapq.heappush(self.nums, val)
        while len(self.nums) > self.capacity:
            heapq.heappop(self.nums)

        return self.nums[0]
```

For $K$ and $N$ numbers to add:
- Time complexity: $O(N \log K)$
- Space complexity: $O(K)$