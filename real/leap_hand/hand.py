"""LeapHand: the LEAP hand's driver and its control loop.

The only code that drives the motors. Everything else (`wlc.Hand`, the loop, the panel)
goes through this class; `scan.py` only reads. No sockets, no recording, no sim.

    from leap_hand.config import load_rig
    from leap_hand.hand import LeapHand

    with LeapHand(load_rig().hand) as hand:                         # torque on, holds pose
        hand.set_target(q_sim)                                      # canonical rad, (16,)
        state = hand.read()
    # leaving the block stops the loop and disables torque (with retries)

Threading: one loop thread owns the bus for its whole life (open, configure, loop, torque
off, close). Public methods never touch the bus; requests that need it (`release`,
`engage`, `set_gains`) are queued and run by the loop thread between ticks.

Control loop, one tick:
1. read position, velocity, current with the SDK's `read_pos_vel_cur()`;
2. if torque is on, advance the setpoint toward the target by at most `max_step_rad`
   (or the slope implied by `duration_s`) and write it with `write_desired_pos`;
3. watchdog: no new target for `watchdog_s` (after the commanded duration) freezes the
   setpoint; torque stays on, so a hand holding an object does not drop it;
4. every `hw_error_check_every` ticks read Hardware_Error_Status; a non-zero value holds
   the hand and refuses new targets (torque is not disabled; the caller decides);
5. `comm_error_limit` consecutive failed ticks stop the loop, disable torque and make
   every later public call raise.

Torque off is guaranteed by `disconnect()`, `__exit__`, an `atexit` hook and (in scripts)
`raise_on_signals()`. All of them end in the loop thread's shutdown path, which disables
torque with retries before closing the port.
"""

from __future__ import annotations

import atexit
import functools
import logging
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from . import conventions as conv
from .config import LeapHandConfig, check_gain, find_port
from .rate import Rate
from .state import HandState, LoopStats
from .third_party.leap_hand_utils import dynamixel_client as dxc

if TYPE_CHECKING:  # typing.Self needs 3.11; annotations are never evaluated at runtime
  from typing_extensions import Self

log = logging.getLogger(__name__)

# Control-table addresses. 11, 80, 82, 84 and 102 are the ones LeapNode.__init__ writes;
# 64 (torque), 116 (goal position), 126-132 (present current/velocity/position) are used
# through the SDK's DynamixelClient; 70 is Hardware_Error_Status.
ADDR_RETURN_DELAY_TIME = 9  # EEPROM, unit 2 us: Return_Delay_Time
ADDR_OPERATING_MODE = 11  # EEPROM: only written with torque off
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_GOAL_CURRENT = (
  102  # LeapNode's "current limit"; the X-series table calls it Goal_Current
)
MODE_CURRENT_BASED_POSITION = 5

TORQUE_OFF_RETRIES = 3
TORQUE_RETRY_INTERVAL_S = 0.05
READ_ATTEMPTS = (
  2  # a bad status packet is retried once before the tick counts as failed
)
# A chained read stalls when a motor misses its turn; that motor stays deaf for a few ms,
# so a retry straight away stalls at the same motor every time (measured: 0 of 40 retries
# succeed at 0 ms, 29 at 2 ms, 38 at 5 ms, the base stall rate from there).
READ_RETRY_BACKOFF_S = 0.005
CONNECT_TIMEOUT_S = 10.0
JOIN_TIMEOUT_S = 5.0
REQUEST_TIMEOUT_S = 2.0
MOVE_TO_TOLERANCE_RAD = 0.05


class HandError(RuntimeError):
  """The hand cannot do what was asked (not connected, loop stopped, fault, refused)."""


class CommError(OSError):
  """A bus transfer failed. The SDK logs such failures instead of raising."""


def _clear_input_buffer(port_handler: Any) -> None:
  """Replacement for the Python SDK's `PortHandler.clearPort`.

  The Python SDK's `clearPort` is `ser.flush()`, a `tcdrain` that waits for the kernel's
  USB-serial "all sent" poll: 3 ms per transaction on the hand's PC (a one-byte read took
  4.0 ms, 1.0 ms with this). The C SDK's `clearPort` is `tcflush(TCIFLUSH)`: it only
  discards unread input, which is what the packet handler needs before a transaction.
  """
  port_handler.ser.reset_input_buffer()


def _set_packet_timeout(port_handler: Any, extra_ms: float, packet_length: int) -> None:
  """Replacement for `PortHandler.setPacketTimeout`: transmission time plus `extra_ms`.

  The SDK adds `2 * LATENCY_TIMER + 2` ms with LATENCY_TIMER fixed at 16 (about 34 ms
  in all), sized for a USB latency timer this rig does not have; a stalled chained read
  should fail fast and be retried instead.
  """
  port_handler.packet_start_time = port_handler.getCurrentTime()
  port_handler.packet_timeout = port_handler.tx_time_per_byte * packet_length + extra_ms


class CountingClient(dxc.DynamixelClient):
  """The SDK client, counting failed transfers so the loop (and the bench) can see them,
  with the input-only `clearPort` (see `_clear_input_buffer`) and, when
  `packet_timeout_ms` is given, a packet timeout sized for this bus."""

  def __init__(
    self, *args: Any, packet_timeout_ms: float | None = None, **kwargs: Any
  ) -> None:
    super().__init__(*args, **kwargs)
    self.failures = 0
    self.port_handler.clearPort = functools.partial(
      _clear_input_buffer, self.port_handler
    )
    if packet_timeout_ms is not None:
      self.port_handler.setPacketTimeout = functools.partial(
        _set_packet_timeout, self.port_handler, float(packet_timeout_ms)
      )

  def handle_packet_result(
    self, comm_result, dxl_error=None, dxl_id=None, context=None
  ):
    ok = super().handle_packet_result(comm_result, dxl_error, dxl_id, context)
    if not ok:
      self.failures += 1
    return ok


class _Request:
  """A function the loop thread runs between ticks for another thread."""

  def __init__(self, fn: Callable[[], None]) -> None:
    self.fn = fn
    self.done = threading.Event()
    self.error: BaseException | None = None

  def run(self) -> None:
    try:
      self.fn()
    except BaseException as exc:  # noqa: BLE001  (handed back to the calling thread)
      self.error = exc
    finally:
      self.done.set()

  def fail(self, error: BaseException) -> None:
    self.error = error
    self.done.set()


def raise_on_signals() -> None:
  """Turn SIGTERM and SIGHUP into SystemExit and SIGINT into KeyboardInterrupt.

  Called by scripts (never by the class), so that `with LeapHand(...)` blocks and
  `finally` clauses run, and the hand ends with torque off, whatever stops the process.
  """

  def _exit(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)

  signal.signal(signal.SIGINT, signal.default_int_handler)
  signal.signal(signal.SIGTERM, _exit)
  signal.signal(signal.SIGHUP, _exit)


class LeapHand:
  """The LEAP hand: 16 XL330 servos in current-based position mode, one control loop."""

  def __init__(self, cfg: LeapHandConfig | None = None) -> None:
    self.cfg = cfg if cfg is not None else LeapHandConfig()
    self.lower, self.upper = conv.resolve_limits(self.cfg.limits_rad)
    self._ids = list(self.cfg.motor_ids)
    self._gains = (self.cfg.kp, self.cfg.ki, self.cfg.kd)
    self._lock = threading.Lock()
    self._requests: queue.Queue[_Request] = queue.Queue()
    self._stop = threading.Event()
    self._ready = threading.Event()
    self._thread: threading.Thread | None = None
    self._fatal: BaseException | None = None
    self._connected = False
    self._client: CountingClient | None = None
    self._hw_reader: dxc.DynamixelReader | None = None
    self._rate: Rate | None = None
    self._reset_loop_state()

  def _reset_loop_state(self) -> None:
    n = conv.NUM_JOINTS
    self._state: HandState | None = None
    self._q_cmd = np.zeros(n)
    self._q_target = np.zeros(n)
    self._step = np.full(n, self.cfg.max_step_rad)
    self._torque_on = False
    self._watchdog_deadline: float | None = None  # None: disarmed
    self._watchdog_active = False
    self._hw_error = np.zeros(n, dtype=np.uint8)
    self._bus_ms: deque[float] = deque(maxlen=1000)
    self._ticks = 0
    self._comm_errors = 0
    self._comm_streak = 0
    self._bus_retries = 0
    self._watchdog_hits = 0
    self._hw_error_events = 0
    self._bus_is_open = False

  # ----------------------------------------------------------------- public API

  def connect(self) -> None:
    """Open the port, ping all IDs, configure, torque on at the present pose, start the loop.

    Raises HandError (with the bus closed and torque off) if any step fails.
    """
    if self._thread is not None:
      raise HandError("already connected")
    self._reset_loop_state()
    self._stop.clear()
    self._ready.clear()
    self._fatal = None
    self._thread = threading.Thread(
      target=self._run, name="leap-hand-loop", daemon=True
    )
    atexit.register(self.disconnect)
    self._thread.start()
    if not self._ready.wait(CONNECT_TIMEOUT_S):
      self.disconnect()
      raise HandError(f"connect did not finish within {CONNECT_TIMEOUT_S} s")
    if self._fatal is not None:
      fatal = self._fatal
      self.disconnect()
      raise HandError(f"connect failed: {fatal}") from fatal
    self._connected = True

  def disconnect(self) -> None:
    """Stop the loop, disable torque (with retries) and close the port. Idempotent."""
    thread = self._thread
    if thread is None:
      return
    self._stop.set()
    if thread is not threading.current_thread():
      thread.join(JOIN_TIMEOUT_S)
      if thread.is_alive():
        log.error(
          "hand loop did not stop within %s s; torque state unknown", JOIN_TIMEOUT_S
        )
    self._thread = None
    self._connected = False
    atexit.unregister(self.disconnect)

  def __enter__(self) -> Self:
    if self._thread is None:
      self.connect()
    return self

  def __exit__(self, *exc: object) -> None:
    self.disconnect()

  @property
  def is_connected(self) -> bool:
    """True while the loop runs and no fatal error happened."""
    thread = self._thread
    return (
      self._connected
      and self._fatal is None
      and thread is not None
      and thread.is_alive()
    )

  def read(self) -> HandState:
    """The last state published by the loop (no bus access)."""
    self._check()
    with self._lock:
      state = self._state
    assert state is not None  # published before connect() returns
    return state.copy()

  def set_target(
    self, q_sim: np.ndarray, duration_s: float | None = None
  ) -> np.ndarray:
    """Set the target pose (canonical rad, shape (16,)); returns it clipped to the limits.

    Without `duration_s` the setpoint moves at up to `max_step_rad` per tick; with it,
    all joints arrive together after `duration_s` (never faster than `max_step_rad`).
    Each call re-arms the watchdog: without a new call for `duration_s + watchdog_s`,
    the setpoint freezes where it is.
    """
    self._check()
    q = np.asarray(q_sim, dtype=float)
    if q.shape != (conv.NUM_JOINTS,):
      raise ValueError(f"target must have shape ({conv.NUM_JOINTS},), got {q.shape}")
    if not np.all(np.isfinite(q)):
      raise ValueError("target contains NaN or inf")
    if duration_s is not None and not (np.isfinite(duration_s) and duration_s >= 0):
      raise ValueError(f"duration_s must be finite and >= 0, got {duration_s}")
    clipped = np.clip(q, self.lower, self.upper)
    now = time.monotonic()
    with self._lock:
      if self._hw_error.any():
        raise HandError(
          f"hardware error on IDs {self._faulty_ids()}: holding; "
          "release() or disconnect(), then power-cycle the hand"
        )
      if not self._torque_on:
        raise HandError("torque is off: call engage() first")
      self._q_target = clipped.copy()
      if duration_s:
        ticks = max(1.0, duration_s * self.cfg.control_hz)
        self._step = np.minimum(
          self.cfg.max_step_rad, np.abs(clipped - self._q_cmd) / ticks
        )
      else:
        self._step = np.full(conv.NUM_JOINTS, self.cfg.max_step_rad)
      self._watchdog_deadline = now + (duration_s or 0.0) + self.cfg.watchdog_s
      self._watchdog_active = False
    return clipped

  def move_to(
    self, q_sim: np.ndarray, duration_s: float, timeout_s: float | None = None
  ) -> bool:
    """`set_target`, then block until the hand is there; False on timeout.

    "There" means the setpoint equals the target and every joint is within
    MOVE_TO_TOLERANCE_RAD of it. The default timeout is `duration_s + 1` s.
    """
    target = self.set_target(q_sim, duration_s)
    deadline = time.monotonic() + (duration_s + 1.0 if timeout_s is None else timeout_s)
    while True:
      state = self.read()
      if (
        np.max(np.abs(state.q_cmd - target)) < 1e-9
        and np.max(np.abs(state.q - target)) < MOVE_TO_TOLERANCE_RAD
      ):
        return True
      now = time.monotonic()
      if now >= deadline:
        return False
      with self._lock:  # the caller is alive: keep the watchdog from freezing the move
        if self._watchdog_deadline is not None:
          self._watchdog_deadline = max(
            self._watchdog_deadline, now + self.cfg.watchdog_s
          )
      time.sleep(1.0 / self.cfg.control_hz)

  def hold(self) -> None:
    """Freeze: target = the current commanded setpoint (torque stays as it is)."""
    self._check()
    with self._lock:
      self._q_target = self._q_cmd.copy()
      self._watchdog_deadline = None
      self._watchdog_active = False  # an explicit hold is not a watchdog hold

  def release(self) -> None:
    """Torque off on all motors; the loop keeps reading. Raises if not confirmed."""

    def job() -> None:
      with self._lock:
        self._torque_on = False
        self._watchdog_deadline = None
        self._watchdog_active = False
      failed = self._bus_set_torque(False, TORQUE_OFF_RETRIES)
      if failed:
        raise HandError(f"torque off not confirmed for IDs {failed}")

    self._call(job)

  def engage(self) -> None:
    """Torque on at the present pose (goal = present position first, then torque)."""

    def job() -> None:
      pos, _, _ = self._bus_read()
      q = conv.from_servo(pos)
      with self._lock:
        self._q_cmd = q.copy()
        self._q_target = q.copy()
        self._step = np.full(conv.NUM_JOINTS, self.cfg.max_step_rad)
        self._watchdog_deadline = None
        self._watchdog_active = False
      self._bus_write(conv.to_servo(q))
      failed = self._bus_set_torque(True, 0)
      if failed:
        off_failed = self._bus_set_torque(False, TORQUE_OFF_RETRIES)
        raise HandError(
          f"torque on failed for IDs {failed}; torque disabled again"
          + (f" (not confirmed for {off_failed})" if off_failed else "")
        )
      with self._lock:
        self._torque_on = True

    self._call(job)

  def set_gains(self, kp: int, ki: int, kd: int) -> None:
    """Write position P/I/D gains to all motors (abduction IDs scaled, as at connect)."""
    for name, value in (("kp", kp), ("ki", ki), ("kd", kd)):
      check_gain(name, value)
    gains = (int(kp), int(ki), int(kd))

    def job() -> None:
      self._bus_write_gains(*gains)
      self._gains = gains

    self._call(job)

  @property
  def gains(self) -> tuple[int, int, int]:
    """The (kp, ki, kd) last written to the servos."""
    return self._gains

  @property
  def stats(self) -> LoopStats:
    """Loop rate, jitter, communication errors, watchdog and hardware-error counts."""
    rate = self._rate.stats if self._rate is not None else None
    with self._lock:
      bus_ms = np.fromiter(self._bus_ms, dtype=float, count=len(self._bus_ms))
    return LoopStats(
      hz=rate.hz if rate else 0.0,
      jitter_p50_ms=rate.jitter_p50_ms if rate else 0.0,
      jitter_p99_ms=rate.jitter_p99_ms if rate else 0.0,
      overruns=rate.overruns if rate else 0,
      ticks=self._ticks,
      bus_ms_p50=float(np.percentile(bus_ms, 50)) if bus_ms.size else 0.0,
      bus_ms_p99=float(np.percentile(bus_ms, 99)) if bus_ms.size else 0.0,
      comm_errors=self._comm_errors,
      comm_error_streak=self._comm_streak,
      bus_retries=self._bus_retries,
      watchdog_hits=self._watchdog_hits,
      hw_error_events=self._hw_error_events,
    )

  # ----------------------------------------------------------------- loop thread

  def _check(self) -> None:
    if self._fatal is not None:
      raise HandError(f"hand loop stopped: {self._fatal!r}") from self._fatal
    if not self._connected or self._thread is None:
      raise HandError("hand is not connected")

  def _call(self, fn: Callable[[], None]) -> None:
    """Run `fn` in the loop thread between ticks; re-raise what it raised."""
    self._check()
    if threading.current_thread() is self._thread:
      fn()
      return
    request = _Request(fn)
    self._requests.put(request)
    if not request.done.wait(REQUEST_TIMEOUT_S):
      self._check()
      raise HandError(
        f"the hand loop did not run the request within {REQUEST_TIMEOUT_S} s"
      )
    if request.error is not None:
      raise request.error

  def _run(self) -> None:
    try:
      self._bus_open()
      self._bus_is_open = True
      self._configure()
      self._rate = Rate(self.cfg.control_hz)
      self._ready.set()
      while not self._stop.is_set():
        self._tick()
        self._rate.sleep()
    except BaseException as exc:  # stored; raised by the next public call
      self._fatal = exc
      log.exception("hand loop stopped")
    finally:
      with self._lock:
        self._torque_on = False
      if self._bus_is_open:
        self._shutdown_bus()
      self._fail_pending_requests()
      self._ready.set()

  def _configure(self) -> None:
    """LeapNode.__init__'s writes, with the rig's values, torque enabled last.

    Torque off first (EEPROM writes are refused while it is on), operating mode 5 at 11,
    gains at 84/82/80 (abduction IDs scaled), goal current at 102, then goal = present
    position, then torque on. LeapNode enables torque right after the mode and writes a
    fixed open pose as the goal; here torque comes last so connecting does not move
    the hand.
    """
    failed = self._bus_set_torque(False, TORQUE_OFF_RETRIES)
    if failed:
      raise HandError(f"could not disable torque before configuring IDs {failed}")
    self._bus_configure()
    pos, vel, cur = self._bus_read()
    q = conv.from_servo(pos)
    self._q_cmd = q.copy()
    self._q_target = q.copy()
    self._bus_write(conv.to_servo(self._q_cmd))
    failed = self._bus_set_torque(True, 0)
    if failed:
      raise HandError(f"torque on failed for IDs {failed}")
    with self._lock:
      self._torque_on = True
    self._publish(time.monotonic(), q, vel, cur)
    outside = (q < self.lower - 1e-6) | (q > self.upper + 1e-6)
    if outside.any():
      log.warning(
        "present pose outside the limits on IDs %s; the setpoint moves "
        "inside at max_step_rad once a target is set",
        np.flatnonzero(outside).tolist(),
      )

  def _tick(self) -> None:
    self._run_requests()
    t_bus = time.perf_counter()
    try:
      pos, vel, cur = self._bus_read()
    except CommError as exc:
      self._comm_failure(exc)
      return
    now = time.monotonic()
    q = conv.from_servo(pos)
    with self._lock:
      torque_on = self._torque_on
      if torque_on:
        self._advance(now)
      else:  # the setpoint follows the hand, so engage() starts from the present pose
        self._q_cmd = q.copy()
        self._q_target = q.copy()
      q_cmd = self._q_cmd.copy()
    if torque_on:
      try:
        self._bus_write(conv.to_servo(q_cmd))
      except CommError as exc:
        self._comm_failure(exc)
        return
    with self._lock:
      self._bus_ms.append((time.perf_counter() - t_bus) * 1e3)
    self._ticks += 1
    if self._ticks % self.cfg.hw_error_check_every == 0:
      try:
        self._update_hw_error(self._bus_read_hw_error())
      except CommError as exc:
        self._comm_failure(exc)
        return
    self._comm_streak = 0
    self._publish(now, q, vel, cur)

  def _advance(self, now: float) -> None:
    """Watchdog, then one rate-limited step of the setpoint (lock held)."""
    if self._watchdog_deadline is not None and now > self._watchdog_deadline:
      self._q_target = self._q_cmd.copy()
      self._watchdog_deadline = None
      self._watchdog_active = True
      self._watchdog_hits += 1
      log.warning("watchdog: no target for %.2f s, holding", self.cfg.watchdog_s)
    delta = self._q_target - self._q_cmd
    self._q_cmd = self._q_cmd + np.clip(delta, -self._step, self._step)

  def _update_hw_error(self, hw: np.ndarray) -> None:
    hw = np.asarray(hw, dtype=np.uint8)
    with self._lock:
      new_fault = hw.any() and not self._hw_error.any()
      self._hw_error = hw.copy()
      if new_fault:
        self._q_target = self._q_cmd.copy()
        self._watchdog_deadline = None
        self._hw_error_events += 1
    if new_fault:
      log.error(
        "hardware error status %s on IDs %s: holding, new targets refused",
        hw[hw != 0].tolist(),
        self._faulty_ids(),
      )

  def _faulty_ids(self) -> list[int]:
    return [self._ids[i] for i in np.flatnonzero(self._hw_error)]

  def _comm_failure(self, exc: BaseException) -> None:
    self._comm_errors += 1
    self._comm_streak += 1
    if self._comm_streak >= self.cfg.comm_error_limit:
      raise HandError(
        f"{self._comm_streak} consecutive bus errors, last: {exc}"
      ) from exc

  def _publish(self, t: float, q: np.ndarray, dq: np.ndarray, cur: np.ndarray) -> None:
    with self._lock:
      self._state = HandState(
        t=t,
        q=np.asarray(q, dtype=float).copy(),
        dq=np.asarray(dq, dtype=float).copy(),
        current_ma=np.asarray(cur, dtype=float).copy(),
        q_cmd=self._q_cmd.copy(),
        q_target=self._q_target.copy(),
        hw_error=self._hw_error.copy(),
        torque_on=self._torque_on,
        watchdog_active=self._watchdog_active,
      )

  def _run_requests(self) -> None:
    while True:
      try:
        request = self._requests.get_nowait()
      except queue.Empty:
        return
      request.run()

  def _fail_pending_requests(self) -> None:
    while True:
      try:
        request = self._requests.get_nowait()
      except queue.Empty:
        return
      request.fail(HandError("the hand loop stopped before running the request"))

  def _shutdown_bus(self) -> None:
    """Torque off with retries, then close the port. Never raises."""
    try:
      failed = self._bus_set_torque(False, TORQUE_OFF_RETRIES)
      if failed:
        log.error("torque off NOT confirmed for IDs %s", failed)
      else:
        log.info("torque off on all motors")
    except Exception:
      log.exception("torque off failed")
    try:
      self._bus_close()
    except Exception:
      log.exception("closing the port failed")
    self._bus_is_open = False

  # ----------------------------------------------------------------- bus layer
  # Called from the loop thread only. FakeLeapHand overrides exactly these methods.

  def _bus_open(self) -> None:
    """Open the port at the rig baudrate and ping every motor ID."""
    cfg = self.cfg
    port = find_port(cfg.port)
    client = CountingClient(
      self._ids,
      port,
      cfg.baudrate,
      cur_scale=cfg.current_unit_ma,
      packet_timeout_ms=cfg.packet_timeout_ms,
    )
    client.connect()  # OSError if the port cannot be opened or set to the baudrate
    try:
      missing, models = [], set()
      for motor_id in self._ids:
        model, comm, _ = client.packet_handler.ping(client.port_handler, motor_id)
        if comm == client.dxl.COMM_SUCCESS:
          models.add(model)
        else:
          missing.append(motor_id)
      if missing:
        raise HandError(
          f"IDs {missing} did not answer ping on {port} at {cfg.baudrate}"
        )
      self._hw_reader = dxc.DynamixelReader(
        client, self._ids, ADDR_HARDWARE_ERROR_STATUS, 1
      )
    except BaseException:
      client.port_handler.closePort()  # nothing was written yet: no torque to undo
      dxc.DynamixelClient.OPEN_CLIENTS.discard(client)
      raise
    self._client = client
    log.info(
      "all %d IDs answered ping on %s (model numbers %s)",
      len(self._ids),
      port,
      sorted(models),
    )

  def _bus_configure(self) -> None:
    """Operating mode, gains, goal current: LeapNode's sync writes (torque is off)."""
    c, n = self._sdk(), len(self._ids)
    c.sync_write(
      self._ids, np.full(n, MODE_CURRENT_BASED_POSITION), ADDR_OPERATING_MODE, 1
    )
    if self.cfg.return_delay_time is not None:  # EEPROM, written with torque off
      c.sync_write(
        self._ids, np.full(n, self.cfg.return_delay_time), ADDR_RETURN_DELAY_TIME, 1
      )
    self._bus_write_gains(*self._gains)
    goal_current = round(self.cfg.current_limit_ma / self.cfg.current_unit_ma)
    c.sync_write(self._ids, np.full(n, goal_current), ADDR_GOAL_CURRENT, 2)

  def _bus_write_gains(self, kp: int, ki: int, kd: int) -> None:
    c, n = self._sdk(), len(self._ids)
    ab = list(conv.ABDUCTION_IDS)
    scale = self.cfg.abduction_gain_scale
    before = c.failures
    c.sync_write(self._ids, np.full(n, kp), ADDR_POSITION_P_GAIN, 2)
    c.sync_write(ab, np.full(len(ab), kp * scale), ADDR_POSITION_P_GAIN, 2)
    c.sync_write(self._ids, np.full(n, ki), ADDR_POSITION_I_GAIN, 2)
    c.sync_write(self._ids, np.full(n, kd), ADDR_POSITION_D_GAIN, 2)
    c.sync_write(ab, np.full(len(ab), kd * scale), ADDR_POSITION_D_GAIN, 2)
    if c.failures != before:
      raise CommError("writing gains failed (see the SDK log)")

  def _bus_read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position (servo rad), velocity (rad/s), current (mA); one retry on a bad packet."""
    c = self._sdk()
    pos, vel, cur = self._read_with_retry(c._pos_vel_cur_reader, "read_pos_vel_cur")
    return pos.astype(float), vel.astype(float), cur.astype(float)

  def _read_with_retry(self, reader: dxc.DynamixelReader, what: str):
    """`reader.read()` with READ_ATTEMPTS single attempts; CommError when all fail.

    The SDK's own retry returns stale data silently; counting the attempts here keeps a
    recovered read (counted in `bus_retries`) apart from a failed tick (`comm_errors`).
    """
    c = self._sdk()
    for attempt in range(READ_ATTEMPTS):
      before = c.failures
      data = reader.read(retries=0)
      if c.failures == before:
        return data
      if attempt < READ_ATTEMPTS - 1:
        self._bus_retries += 1
        time.sleep(READ_RETRY_BACKOFF_S)
    raise CommError(f"{what} failed {READ_ATTEMPTS} times in a row (see the SDK log)")

  def _bus_write(self, q_servo: np.ndarray) -> None:
    """Goal position (servo rad) on all motors."""
    c = self._sdk()
    before = c.failures
    c.write_desired_pos(self._ids, np.asarray(q_servo, dtype=float))
    if c.failures != before:
      raise CommError("write_desired_pos failed (see the SDK log)")

  def _bus_read_hw_error(self) -> np.ndarray:
    assert self._hw_reader is not None
    values = self._read_with_retry(self._hw_reader, "reading Hardware_Error_Status")
    return values.astype(np.uint8)

  def _bus_set_torque(self, enabled: bool, retries: int) -> list[int]:
    """Torque on/off on all motors; returns the IDs that did not acknowledge.

    Uses the SDK's `write_byte` (as its `set_torque_enabled` does) but returns the
    failed IDs and does not sleep after a success.
    """
    c = self._sdk()
    remaining = list(self._ids)
    for attempt in range(retries + 1):
      remaining = list(c.write_byte(remaining, int(enabled), dxc.ADDR_TORQUE_ENABLE))
      if not remaining:
        break
      if attempt < retries:
        time.sleep(TORQUE_RETRY_INTERVAL_S)
    return remaining

  def _bus_close(self) -> None:
    if self._client is not None:
      self._client.disconnect()  # the SDK disables torque once more, then closes
      self._client = None

  def _sdk(self) -> CountingClient:
    if self._client is None:
      raise CommError("port is not open")
    return self._client
