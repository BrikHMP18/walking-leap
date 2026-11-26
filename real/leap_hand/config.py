"""Hand configuration, loaded from a rig file (`leap_right.yaml`, beside this file).

Port, baudrate, gains, current limit, joint limits: every number that reaches the motors is
in the rig file; the dataclass defaults only mirror the SDK's `LeapNode` values. Unknown
keys are an error, so a typo in a rig file cannot silently fall back to a default.

Rig file layout:

    name: leap_right
    hand: {port: ..., baudrate: ..., kp: ..., ...}   # LeapHandConfig fields
"""

from __future__ import annotations

import dataclasses
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

from .conventions import MOTOR_IDS, resolve_limits

DEFAULT_RIG = Path(__file__).with_name("leap_right.yaml")
GAIN_MAX = 16383  # X-series Position P/I/D Gain range is 0 to 16,383
# The U2D2 (an FTDI FT232H) by its stable name, whatever ttyUSB number it was given.
PORT_GLOB = "/dev/serial/by-id/usb-FTDI_*-if00-port0"


@dataclass
class LeapHandConfig:
  """Everything `LeapHand` needs to open, configure and run the hand."""

  port: str = "auto"  # a device path, or "auto": find_port()
  baudrate: int = 4_000_000
  motor_ids: tuple[int, ...] = MOTOR_IDS
  kp: int = 600
  ki: int = 0
  kd: int = 200
  abduction_gain_scale: float = (
    0.75  # applied to kp and kd on ABDUCTION_IDS, as LeapNode
  )
  current_limit_ma: int = (
    350  # Goal Current (address 102); 550 only for full-size motors
  )
  current_unit_ma: float = (
    1.0  # mA per current unit; XL330/XC330: 1.0 (the SDK default is XH's)
  )
  # Return Delay Time (register 9, unit 2 us), written at connect when set. The factory
  # value 250 (500 us) is paid once per motor in every sync read, 8 ms over 16 motors;
  # 0 removes that wait. None leaves the servos as they are.
  return_delay_time: int | None = None
  # Time allowed for a status packet beyond its transmission time, ms. The SDK's default
  # (about 34 ms) is sized for a 16 ms USB latency timer; with the U2D2 at 1 ms a full
  # fast sync read of 16 motors arrives within 2.2 ms, so a stalled read (a motor that
  # misses its turn in the chain) should give up and be retried quickly. None keeps the
  # SDK's formula.
  packet_timeout_ms: float | None = None
  control_hz: float = 100.0  # background loop rate
  max_step_rad: float = 0.05  # per-tick change of the commanded setpoint
  watchdog_s: float = 0.5  # no new target for this long -> hold
  limits_rad: tuple[tuple[float, float], ...] | None = None  # None -> SDK defaults
  comm_error_limit: int = 5  # consecutive bus errors before the loop stops
  hw_error_check_every: int = 10  # ticks between Hardware_Error_Status reads

  def __post_init__(self) -> None:
    self.motor_ids = tuple(int(i) for i in self.motor_ids)
    if self.motor_ids != MOTOR_IDS:
      raise ValueError(f"motor_ids must be {MOTOR_IDS} (bus ID == array index)")
    if self.limits_rad is not None:
      self.limits_rad = tuple((float(lo), float(hi)) for lo, hi in self.limits_rad)
    resolve_limits(self.limits_rad)  # raises if a rig widens the SDK limits
    for name in ("kp", "ki", "kd"):
      check_gain(name, getattr(self, name))
    if not 0.0 < self.abduction_gain_scale <= 1.0:
      raise ValueError("abduction_gain_scale must be in (0, 1]")
    if self.return_delay_time is not None and not (
      isinstance(self.return_delay_time, int) and 0 <= self.return_delay_time <= 254
    ):
      raise ValueError(
        f"return_delay_time must be an int in [0, 254] or null, "
        f"got {self.return_delay_time!r}"
      )
    if self.packet_timeout_ms is not None and not self.packet_timeout_ms > 0:
      raise ValueError(
        f"packet_timeout_ms must be positive or null, got {self.packet_timeout_ms!r}"
      )
    positive = (
            "baudrate", "current_limit_ma", "current_unit_ma", "control_hz", "max_step_rad",
            "watchdog_s", "comm_error_limit", "hw_error_check_every",
        )  # fmt: skip
    for name in positive:
      if not getattr(self, name) > 0:
        raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


@dataclass
class Rig:
  """One physical setup: a named hand."""

  name: str
  hand: LeapHandConfig = field(default_factory=LeapHandConfig)
  path: str | None = None


def check_gain(name: str, value: int) -> None:
  """Raise ValueError unless `value` is an integer in the servo's gain range."""
  if isinstance(value, bool) or int(value) != value or not 0 <= value <= GAIN_MAX:
    raise ValueError(f"{name} must be an integer in [0, {GAIN_MAX}], got {value!r}")


def _build(cls: type, data: dict | None, where: str):
  data = dict(data or {})
  known = {f.name for f in dataclasses.fields(cls)}
  unknown = sorted(set(data) - known)
  if unknown:
    raise ValueError(f"{where}: unknown keys {unknown}; valid keys: {sorted(known)}")
  return cls(**data)


def rig_from_dict(data: dict, path: str | None = None) -> Rig:
  """Build a Rig from the parsed YAML mapping."""
  where = path or "rig"
  unknown = sorted(set(data) - {"name", "hand"})
  if unknown:
    raise ValueError(f"{where}: unknown top-level keys {unknown}")
  if "name" not in data:
    raise ValueError(f"{where}: missing 'name'")
  return Rig(
    name=str(data["name"]),
    hand=_build(LeapHandConfig, data.get("hand"), f"{where}: hand"),
    path=path,
  )


def load_rig(path: str | Path = DEFAULT_RIG) -> Rig:
  """Load and validate a rig file (by default the right hand's, beside this file)."""
  import yaml  # here, not at module level: the fake hand and the tests need no rig

  path = Path(path)
  with path.open() as f:
    data = yaml.safe_load(f)
  if not isinstance(data, dict):
    raise TypeError(f"{path}: expected a mapping at the top level")
  return rig_from_dict(data, str(path))


def find_port(port: str = "auto") -> str:
  """The U2D2's serial port: `port` as given, unless it is "auto"; then $WL_LEAP_PORT if
  set, else the one FTDI adapter under /dev/serial/by-id/. OSError if none or several."""
  if port != "auto":
    return port
  if os.environ.get("WL_LEAP_PORT"):
    return os.environ["WL_LEAP_PORT"]
  found = sorted(glob.glob(PORT_GLOB))
  if len(found) == 1:
    return found[0]
  if not found:
    raise OSError(
      "no U2D2 under /dev/serial/by-id/ (plugged in? `ls /dev/serial/by-id/`); "
      "or set WL_LEAP_PORT to its port"
    )
  raise OSError(f"several FTDI adapters {found}: set WL_LEAP_PORT to the U2D2's")
