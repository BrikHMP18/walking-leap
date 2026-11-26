"""Fixed-rate loop helper with achieved-rate and jitter statistics.

Used by the hand's control loop:

    rate = Rate(100.0)
    while running:
        work()
        rate.sleep()
    print(rate.stats)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RateStats:
  """Statistics over the last `window` loop periods."""

  hz: float  # achieved rate (1 / mean period)
  jitter_p50_ms: float  # |period - target period|, median
  jitter_p99_ms: float  # |period - target period|, 99th percentile
  overruns: int  # ticks whose work ran past the deadline (total)
  ticks: int  # completed sleep() calls (total)

  def to_dict(self) -> dict:
    """Plain dict, for logs."""
    return asdict(self)


class Rate:
  """Sleep so that successive `sleep()` calls return at a fixed rate.

  The schedule advances by one period per tick, so it does not drift. A tick whose work
  ran past its deadline counts as an overrun and does not sleep; when the loop falls more
  than one period behind, the missed ticks are dropped instead of run back to back.
  """

  def __init__(self, hz: float, window: int = 1000) -> None:
    if hz <= 0:
      raise ValueError(f"hz must be positive, got {hz}")
    self.hz = float(hz)
    self.period = 1.0 / self.hz
    self._lock = threading.Lock()
    self._periods: deque[float] = deque(maxlen=window)
    self._next = time.monotonic() + self.period
    self._last: float | None = None
    self._overruns = 0
    self._ticks = 0

  def remaining(self) -> float:
    """Seconds until the next deadline (negative when late)."""
    return self._next - time.monotonic()

  def sleep(self) -> None:
    """Sleep until the next deadline and record the achieved period."""
    now = time.monotonic()
    delay = self._next - now
    if delay > 0:
      time.sleep(delay)
      self._next += self.period
    else:
      self._next += self.period
      if self._next < now:
        self._next = now + self.period
    t = time.monotonic()
    with self._lock:
      if delay <= 0:
        self._overruns += 1
      if self._last is not None:
        self._periods.append(t - self._last)
      self._last = t
      self._ticks += 1

  @property
  def stats(self) -> RateStats:
    """Achieved rate and jitter over the window; safe to read from any thread."""
    with self._lock:
      periods = np.fromiter(self._periods, dtype=float, count=len(self._periods))
      overruns, ticks = self._overruns, self._ticks
    if periods.size == 0:
      return RateStats(0.0, 0.0, 0.0, overruns, ticks)
    jitter_ms = np.abs(periods - self.period) * 1e3
    return RateStats(
      hz=float(1.0 / periods.mean()),
      jitter_p50_ms=float(np.percentile(jitter_ms, 50)),
      jitter_p99_ms=float(np.percentile(jitter_ms, 99)),
      overruns=overruns,
      ticks=ticks,
    )
