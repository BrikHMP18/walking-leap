"""State and loop statistics of the hand (numpy and the standard library only)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

_ARRAYS = ("q", "dq", "current_ma", "q_cmd", "q_target", "hw_error")


@dataclass
class HandState:
  """One reading of the hand, all arrays (16,) in bus-ID order, canonical radians."""

  t: float  # time.monotonic() of the read, on the machine running the loop
  q: np.ndarray  # measured position, rad
  dq: np.ndarray  # measured velocity, rad/s
  current_ma: np.ndarray  # measured current, mA
  q_cmd: np.ndarray  # setpoint written this tick
  q_target: np.ndarray  # latest target from the caller (after clipping)
  hw_error: np.ndarray  # uint8 Hardware_Error_Status per motor (0 = fine)
  torque_on: bool
  watchdog_active: bool  # holding because no target arrived in time

  def to_dict(self) -> dict:
    """Plain types (lists, floats, bools), for JSON."""
    d = {name: getattr(self, name) for name in ("t", "torque_on", "watchdog_active")}
    d["t"] = float(d["t"])
    d["torque_on"] = bool(d["torque_on"])
    d["watchdog_active"] = bool(d["watchdog_active"])
    for name in _ARRAYS:
      d[name] = np.asarray(getattr(self, name)).tolist()
    return d

  @classmethod
  def from_dict(cls, d: dict) -> HandState:
    """Inverse of `to_dict`."""
    arrays = {name: np.asarray(d[name], dtype=float) for name in _ARRAYS}
    arrays["hw_error"] = arrays["hw_error"].astype(np.uint8)
    return cls(
      t=float(d["t"]),
      torque_on=bool(d["torque_on"]),
      watchdog_active=bool(d["watchdog_active"]),
      **arrays,
    )

  def copy(self) -> HandState:
    """Deep copy (the arrays are copied)."""
    return dataclasses.replace(self, **{n: getattr(self, n).copy() for n in _ARRAYS})


@dataclass
class LoopStats:
  """Control-loop health."""

  hz: float  # achieved loop rate over the last window
  jitter_p50_ms: float
  jitter_p99_ms: float
  overruns: int  # ticks whose work ran past the deadline
  ticks: int
  bus_ms_p50: float  # read + write time of one tick, median over the window
  bus_ms_p99: float
  comm_errors: int  # ticks whose bus transfer failed after the retry, total
  comm_error_streak: (
    int  # consecutive failed ticks (the loop stops at comm_error_limit)
  )
  bus_retries: int  # reads whose first packet was bad and whose retry succeeded
  watchdog_hits: int  # times the watchdog froze the setpoint
  hw_error_events: int  # times a non-zero Hardware_Error_Status appeared

  def to_dict(self) -> dict:
    """Plain dict, for logs."""
    return dataclasses.asdict(self)
