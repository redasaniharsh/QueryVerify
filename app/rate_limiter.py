"""
Lightweight in-memory sliding-window rate limiter.

Protects endpoints against request flooding without adding heavy dependencies
(Redis, Memcached, etc.). Thread-safe and keyed by client identifier (e.g. IP).
"""

import threading
import time
from collections import defaultdict
from typing import Dict, List, Set

LOOPBACK_HOSTS: Set[str] = {"127.0.0.1", "::1", "localhost", "testclient"}


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter per client key."""

    def __init__(
        self,
        limit: int = 10,
        window_seconds: float = 60.0,
        exempt_loopback: bool = True,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self.exempt_loopback = exempt_loopback
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request from client_id is allowed.

        Returns True if allowed, False if the rate limit is exceeded.
        Records the current timestamp on allowed requests.
        Loopback hosts are exempted when exempt_loopback is True.
        """
        if self.exempt_loopback and client_id in LOOPBACK_HOSTS:
            return True

        if self.limit <= 0:
            return True

        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            timestamps = self._requests[client_id]
            # Prune timestamps older than cutoff
            valid_start = 0
            while valid_start < len(timestamps) and timestamps[valid_start] <= cutoff:
                valid_start += 1
            if valid_start > 0:
                timestamps[:valid_start] = []

            if len(timestamps) >= self.limit:
                return False

            timestamps.append(now)
            return True

    def reset(self):
        """Clear all recorded request history (useful for testing)."""
        with self._lock:
            self._requests.clear()
