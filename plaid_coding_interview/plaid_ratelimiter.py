'''
题意还原（你可以这样复述给面试官）

系统有个队列，用户在“这一秒内”多次调用：

requestAPI(cost)：把一个请求（只记录 cost）加入等待队列（FIFO）

每秒末尾系统调用一次 emitAPI()：从队列头开始取出尽可能多的请求执行，要求：

Part1：本秒执行的 cost 总和 ≤ rate_limit

Part2：还要满足 任意连续 w 秒 内执行的 cost 总和 ≤ rate_limit（等价：sliding window budget）

'''

#  Answer to Q1:

from collections import deque
from typing import List, Deque


class RateLimiterPart1:
    """
    Part1: per-second budget.
    - requestAPI(cost): enqueue a request with given cost
    - emitAPI(): at end of each second, execute as many queued requests as possible
      such that sum(costs_this_second) <= rate_limit.
      Must be FIFO: cannot skip the head of the queue.
    """

    def __init__(self, rate_limit: int):
        if rate_limit < 0:
            raise ValueError("rate_limit must be non-negative")
        self.rate_limit = rate_limit
        self.pending: Deque[int] = deque()  # FIFO queue of request costs

    def requestAPI(self, cost: int) -> None:
        if cost < 0:
            raise ValueError("cost must be non-negative")
        # We just enqueue; actual execution happens in emitAPI()
        self.pending.append(cost)

    def emitAPI(self) -> List[int]:
        """
        Execute as many requests as possible under this second's budget.
        Return the list of executed costs in execution order.
        """
        executed: List[int] = []
        budget = self.rate_limit

        # FIFO: only pop from the left; if the head doesn't fit, we stop.
        while self.pending and self.pending[0] <= budget:
            cost = self.pending.popleft()
            executed.append(cost)
            budget -= cost

        return executed


rl = RateLimiterPart1(rate_limit=5)
rl.requestAPI(3)
rl.requestAPI(2)
rl.requestAPI(1)
rl.requestAPI(1)
print(rl.emitAPI())  # [3, 2]
rl.requestAPI(2)
print(rl.emitAPI())  # [1, 1, 2]

# Part2：还要满足 任意连续 w 秒 内执行的 cost 总和 ≤ rate_limit（等价：sliding window budget）
# Answer to Part 2


from collections import deque


class SlidingWindowRateLimiter:
    def __init__(self, rate_limit, window_size):
        self.rate_limit = rate_limit
        self.window_size = window_size

        self.pending = deque()      # requests waiting to run
        self.history = deque()      # (time_executed, cost) within window
        self.window_sum = 0         # total cost in the current window
        self.t = 0                  # current second

    def requestAPI(self, cost):
        # Just queue the request
        self.pending.append(cost)

    def _evict_old(self):
        # Remove executions that slid out of the last 'window_size' seconds
        cutoff = self.t - self.window_size
        while self.history and self.history[0][0] <= cutoff:
            _, old_cost = self.history.popleft()
            self.window_sum -= old_cost

    def emitAPI(self):
        # Step 1: slide the window
        self._evict_old()

        # Step 2: compute how much cost we can still execute
        budget = self.rate_limit - self.window_sum

        executed = []

        # Step 3: FIFO execution
        while self.pending and self.pending[0] <= budget:
            cost = self.pending.popleft()
            executed.append(cost)

            # Record this execution into window history
            self.history.append((self.t, cost))
            self.window_sum += cost
            budget -= cost

        # Step 4: move to next second
        self.t += 1
        return executed

# testing part 2

rl = SlidingWindowRateLimiter(rate_limit=5, window_size=3)

# t=0
rl.requestAPI(4)
rl.requestAPI(3)
rl.requestAPI(3)
print(rl.emitAPI())  # [4]

# t=1
print(rl.emitAPI())  # []

# t=2
print(rl.emitAPI())  # []

# t=3 (now t=0 is out of window of size 3? window at t=3 covers [1,3], so cost=4 evicted)
print(rl.emitAPI())  # [3]

print(rl.emitAPI())  # []
print(rl.emitAPI())  # []
print(rl.emitAPI())  # [3]

