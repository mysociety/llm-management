"""
Deterministic time for public-web concurrency and cache tests.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class Clock:
    """
    Advance monotonic time while yielding to concurrent test tasks.
    """

    now: float = 0
    waits: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)
