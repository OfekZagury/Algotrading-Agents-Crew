import threading
import time
from collections import defaultdict, deque


class RateLimitExceededException(Exception):
    """Raised when a caller exceeds the allowed number of calls in the time window."""


class RateLimiter:
    """
    Sliding-window rate limiter, tracked separately per key (e.g. "alpaca_api").

    allow_request() returns True when the call is allowed and raises
    RateLimitExceededException when the key already used max_calls
    within the last period_seconds.
    """

    def __init__(self, max_calls: int, period_seconds: float):
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive")
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls = defaultdict(deque)
        self._lock = threading.Lock()

    def allow_request(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            window = self._calls[key]
            # Drop timestamps that are outside the current window
            while window and now - window[0] >= self.period_seconds:
                window.popleft()

            if len(window) >= self.max_calls:
                raise RateLimitExceededException(
                    f"Rate limit exceeded for '{key}': "
                    f"max {self.max_calls} calls per {self.period_seconds}s"
                )

            window.append(now)
            return True
