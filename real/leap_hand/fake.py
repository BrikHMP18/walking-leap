"""FakeLeapHand: LeapHand's API and control loop on simulated servos, no serial.

It subclasses `LeapHand` and overrides only the `_bus_*` methods, so the control loop,
limits, watchdog, hardware-error handling and torque-off paths under test are the real
ones. The simulated servos follow a first-order model in the servo's own units:

    q += clip(goal - q, +-v_max * dt)   (only while torque is on)

with optional position noise and a synthetic current proportional to the remaining error,
capped at the goal current. Like a real servo, they refuse an operating-mode (EEPROM)
write while torque is on (counted in `eeprom_writes_with_torque`) and a motor with a
hardware error drops torque and refuses to re-enable it.

Fault injection for tests: `fail_next_reads`, `raise_on_next_read`, `missing_ids`,
`inject_hw_error()`. Bus-level state for assertions: `sim_torque`, `sim_open`, `sim_mode`,
`sim_gains`, `sim_goal_current`, `torque_writes`.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import conventions as conv
from .config import LeapHandConfig
from .hand import MODE_CURRENT_BASED_POSITION, CommError, HandError, LeapHand

CURRENT_MA_PER_RAD = (
  2000.0  # synthetic current per rad of tracking error, before the cap
)
X_SERIES_DEFAULT_MODE = 3  # position control, the X-series factory default


class FakeLeapHand(LeapHand):
  """LeapHand on simulated servos. Same public API; for the tests and `--fake`."""

  def __init__(
    self,
    cfg: LeapHandConfig | None = None,
    *,
    v_max: float = 4.0,
    noise_std: float = 0.0,
    q0: np.ndarray | None = None,
    seed: int = 0,
  ) -> None:
    super().__init__(cfg)
    n = conv.NUM_JOINTS
    self.v_max = float(v_max)
    self.noise_std = float(noise_std)
    self._rng = np.random.default_rng(seed)
    self._sim_lock = threading.Lock()
    q0 = np.zeros(n) if q0 is None else np.asarray(q0, dtype=float)
    # Simulated servos, in servo units (LEAPhand rad) like the real bus.
    self.sim_q = conv.to_servo(q0)
    self.sim_dq = np.zeros(n)
    self.sim_cur = np.zeros(n)
    self.sim_goal = self.sim_q.copy()
    self.sim_torque = np.zeros(n, dtype=bool)
    self.sim_mode = np.full(n, X_SERIES_DEFAULT_MODE)
    self.sim_gains = {"p": np.zeros(n), "i": np.zeros(n), "d": np.zeros(n)}
    self.sim_goal_current = np.zeros(n)
    self.sim_hw_error = np.zeros(n, dtype=np.uint8)
    self.sim_open = False
    self._sim_t = time.monotonic()
    self.eeprom_writes_with_torque = 0
    self.torque_writes: list[bool] = []
    # Fault injection.
    self.fail_next_reads = 0
    self.raise_on_next_read: BaseException | None = None
    self.missing_ids: tuple[int, ...] = ()

  def inject_hw_error(self, motor_id: int, status: int = 0x04) -> None:
    """Set Hardware_Error_Status on one motor (default: overheating); it drops torque."""
    with self._sim_lock:
      self.sim_hw_error[motor_id] = status
      self.sim_torque[motor_id] = False

  def move_sim(self, q_sim: np.ndarray) -> None:
    """Teleport the simulated joints (canonical rad), e.g. a hand pushing a finger."""
    with self._sim_lock:
      self.sim_q = conv.to_servo(np.asarray(q_sim, dtype=float))

  # ------------------------------------------------------------- simulated bus

  def _need_open(self) -> None:
    if not self.sim_open:
      raise CommError("port is not open")

  def _bus_open(self) -> None:
    missing = [i for i in self._ids if i in self.missing_ids]
    if missing:
      raise HandError(f"IDs {missing} did not answer ping on {self.cfg.port} (fake)")
    self.sim_open = True
    self._sim_t = time.monotonic()

  def _bus_configure(self) -> None:
    self._need_open()
    with self._sim_lock:
      if self.sim_torque.any():
        self.eeprom_writes_with_torque += 1  # a real servo ignores this write
      else:
        self.sim_mode[:] = MODE_CURRENT_BASED_POSITION
      goal_current = round(self.cfg.current_limit_ma / self.cfg.current_unit_ma)
      self.sim_goal_current[:] = goal_current
    self._bus_write_gains(*self._gains)

  def _bus_write_gains(self, kp: int, ki: int, kd: int) -> None:
    self._need_open()
    ab = list(conv.ABDUCTION_IDS)
    scale = self.cfg.abduction_gain_scale
    p, i, d = (np.full(conv.NUM_JOINTS, float(g)) for g in (kp, ki, kd))
    p[ab] = int(kp * scale)  # the SDK's sync_write truncates with int()
    d[ab] = int(kd * scale)
    with self._sim_lock:
      self.sim_gains = {"p": p, "i": i, "d": d}

  def _bus_read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    self._need_open()
    if self.raise_on_next_read is not None:
      exc, self.raise_on_next_read = self.raise_on_next_read, None
      raise exc
    if self.fail_next_reads > 0:
      self.fail_next_reads -= 1
      raise CommError("simulated read failure")
    with self._sim_lock:
      self._simulate()
      q = self.sim_q.copy()
      if self.noise_std > 0:
        q += self._rng.normal(0.0, self.noise_std, q.shape)
      return q, self.sim_dq.copy(), self.sim_cur.copy()

  def _simulate(self) -> None:
    now = time.monotonic()
    dt = now - self._sim_t
    self._sim_t = now
    if dt <= 0:
      return
    on = self.sim_torque
    step = np.clip(self.sim_goal - self.sim_q, -self.v_max * dt, self.v_max * dt)
    q_new = np.where(on, self.sim_q + step, self.sim_q)
    self.sim_dq = (q_new - self.sim_q) / dt
    self.sim_q = q_new
    cap = self.sim_goal_current
    err = self.sim_goal - self.sim_q
    self.sim_cur = np.where(on, np.clip(CURRENT_MA_PER_RAD * err, -cap, cap), 0.0)

  def _bus_write(self, q_servo: np.ndarray) -> None:
    self._need_open()
    with self._sim_lock:
      self.sim_goal = np.asarray(q_servo, dtype=float).copy()

  def _bus_read_hw_error(self) -> np.ndarray:
    self._need_open()
    with self._sim_lock:
      return self.sim_hw_error.copy()

  def _bus_set_torque(self, enabled: bool, retries: int) -> list[int]:
    self._need_open()
    with self._sim_lock:
      refused = [i for i in self._ids if enabled and self.sim_hw_error[i]]
      for i in self._ids:
        if i not in refused:
          self.sim_torque[i] = enabled
      self.torque_writes.append(enabled)
    return refused

  def _bus_close(self) -> None:
    self.sim_open = False
