#!/usr/bin/env python3
# ruff: noqa: B905, E741
"""Reader for the TM171 IMU (SYD Dynamics TransducerM) of Walking Leap, on USB.

Linux, Python 3.8 or newer, standard library only: nothing to install.

In a control loop:

    from tm171 import TM171

    with TM171() as imu:
        imu.wait()               # the first reading, or TimeoutError
        while running:
            s = imu.latest       # the newest reading, palm axes
            if not imu.fresh():  # unplugged or frozen: stop the hand
                break

    s.gyro     rotation rate, rad/s                  the simulation's base_ang_vel
    s.gravity  which way is down, unit vector        the simulation's projected_gravity

In a terminal: ``wlimu`` for the live view, ``wlimu --help`` for the rest.

Axes. The IMU's own are the maker's: x to the pins, z out of the lid, y to the left
seen with the pins up. On the hand it sits turned by 90 degrees, pins to the thumb side:

    palm x = -imu y        palm y = +imu x        palm z = imu z

The palm's axes are the simulation's: x to the fingertips, y to the thumb side, z out of
the back of the hand. Palm down and level, gravity reads (0, 0, -1).
"""

import argparse
import collections
import csv
import errno
import fcntl
import glob
import json
import math
import os
import queue
import select
import statistics
import struct
import sys
import termios
import threading
import time
from datetime import datetime
from typing import NamedTuple, Optional, Tuple

HERE = os.path.dirname(os.path.realpath(__file__))
SERIAL = os.environ.get(
  "TM171_SERIAL", ""
)  # Its USB serial number, if several are plugged in.
# Where the logs and the settings' backups go (one before every change).
OUT = os.environ.get(
  "TM171_OUT",
  os.path.join(os.path.dirname(os.path.dirname(HERE)), "outputs", "real", "imu"),
)
PORTS = "/dev/serial/by-id/*STM32_Virtual_ComPort*"

# Objects of the maker's protocol (EasyProtocol; its EasyProfile library is public, BSD 2-clause).
REQUEST, SETTINGS, STATUS = 12, 21, 22
QUAT, RPY, GRAVITY, RAW, COMBO = 32, 35, 36, 41, 43
HOST = 2  # The PC's id. 0 is everyone.

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------------------
# Axes


def to_palm(v: Vec3) -> Vec3:
  """A vector from the IMU's axes to the palm's."""
  return (-v[1], v[0], v[2])


def quat_to_palm(q):
  """(w, x, y, z) of the IMU in the world -> of the palm in the world."""
  w, x, y, z = q
  c = math.sqrt(0.5)  # Right-multiplied by a turn of -90 degrees about z.
  return (c * (w + z), c * (x - y), c * (x + y), c * (z - w))


def gravity_from_quat(q) -> Vec3:
  """World down (0, 0, -1) in the axes of the body with orientation ``q``."""
  w, x, y, z = q
  return (-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y)))


# --------------------------------------------------------------------------------------
# Packets: AA 55, length, payload, CRC. Payload: 4 bytes of address, then the values.


def _crc_table():
  table = []
  for c in range(256):
    for _ in range(8):
      c = (c >> 1) ^ 0xA001 if c & 1 else c >> 1
    table.append(c)
  return table


_CRC = _crc_table()


def crc16(data) -> int:
  """CRC-16 Modbus."""
  c = 0xFFFF
  for b in data:
    c = (c >> 8) ^ _CRC[(c ^ b) & 0xFF]
  return c


def packet(cmd: int, body: bytes, to: int = 0) -> bytes:
  payload = struct.pack("<I", cmd | HOST << 10 | to << 21) + body
  data = bytes([len(payload)]) + payload
  return b"\xaa\x55" + data + struct.pack("<H", crc16(data))


class Parser:
  """Cuts a byte stream into packets. Anything broken is skipped and counted."""

  def __init__(self):
    self.buf = bytearray()
    self.bad_crc = 0
    self.junk = 0

  def feed(self, data):
    """Yields (object number, sender's id, values)."""
    buf = self.buf
    buf += data
    while True:
      start = buf.find(b"\xaa\x55")
      if start < 0:
        drop = max(
          0, len(buf) - 1
        )  # The last byte may be the first of the next header.
        self.junk += drop
        del buf[:drop]
        return
      if start:
        self.junk += start
        del buf[:start]
      if len(buf) < 3 or len(buf) < buf[2] + 5:
        return
      n = buf[2]
      body = bytes(buf[2 : 3 + n])
      if n < 4 or crc16(body) != buf[3 + n] | buf[4 + n] << 8:
        self.bad_crc += 1
        del buf[:2]
        continue
      del buf[: n + 5]
      head = struct.unpack_from("<I", body, 1)[0]
      if head >> 7 & 7:  # Reserved bits: not a packet we know.
        self.bad_crc += 1
        continue
      yield head & 127, head >> 10 & 2047, body[5:]


# --------------------------------------------------------------------------------------
# Settings (object 21). On the TM171 the last four bytes are silent time and output rate,
# and bit 20 is the USB output: read on our IMU and the same in the maker's ImuAssistant 3.9.23.

_FIELDS = (
  "switches",
  "uart2",
  "uart1",
  "can",
  "gain_acc",
  "gain_mag",
  "inhibit",
  "silent",
  "odr",
)
_SWITCHES = {  # name: (first bit, number of bits)
  "gyro": (0, 1),
  "acc": (1, 1),
  "mag": (2, 1),
  "gyro_err_filter": (3, 1),
  "mag_self_adapt": (4, 1),
  "out_status": (5, 1),
  "out_raw": (6, 1),
  "out_quat": (8, 1),
  "out_euler": (10, 1),
  "out_rpy": (11, 1),
  "out_gravity": (12, 1),
  "out_combo": (27, 1),
  "binary": (13, 1),
  "checksum": (14, 3),
  "ack": (17, 1),
  "save": (18, 1),
  "uart1_on": (19, 1),
  "usb_on": (20, 1),
  "boot": (24, 2),
}
_NUMBERS = ("gain_acc", "gain_mag", "silent", "odr")  # May be changed. The bauds never.
_LOCKED = ("binary", "checksum", "ack", "save", "usb_on")  # Would cut our own link.
_RATES = (1, 5, 10, 25, 50, 100, 200, 400, 800)

# What Walking Leap wants, and why, in plain words.
WANTED = {
  "gyro": (1, "needed: how fast the palm turns"),
  "acc": (1, "needed: keeps down pointing down"),
  "mag": (0, "it only helps the heading, which nothing here uses"),
  "gyro_err_filter": (1, "takes the gyro's drift out while it runs"),
  "gain_acc": (208, "the maker's value for a robot (290 drone, 280 strong vibration)"),
  "boot": (0, "auto: ready in 3 s, the hand may move meanwhile"),
  "silent": (0, "talks as soon as it's on"),
  "odr": (400, "8 readings per 50 Hz step, the newest at most 2.5 ms old"),
  "out_raw": (1, "the gyro, for the policy"),
  "out_gravity": (1, "down, for the policy"),
  "out_quat": (1, "the full orientation, for the viewer and the logs"),
  "out_rpy": (0, "angles break down when the hand stands on its side"),
  "out_euler": (0, "not used"),
  "out_combo": (0, "not used"),
  "out_status": (1, "its own quality grade, without having to ask"),
  "usb_on": (1, "the cable we read"),
}


class Settings:
  def __init__(self, content: bytes):
    if len(content) != 20:
      raise ValueError(
        f"settings of {len(content)} bytes, 20 expected: not touching them"
      )
    self._v = dict(zip(_FIELDS, struct.unpack("<I8H", content)))
    if self["binary"] != 1 or self["checksum"] != 2:
      raise ValueError("these settings aren't laid out as expected: not touching them")

  def __getitem__(self, key):
    if key in _SWITCHES:
      bit, n = _SWITCHES[key]
      return self._v["switches"] >> bit & ((1 << n) - 1)
    return self._v[key]

  def __setitem__(self, key, value):
    if key in _SWITCHES:
      bit, n = _SWITCHES[key]
      if not 0 <= value < 1 << n:
        raise ValueError(f"{key}: {value}")
      mask = ((1 << n) - 1) << bit
      self._v["switches"] = self._v["switches"] & ~mask | value << bit
    else:
      self._v[key] = value

  def content(self) -> bytes:
    return struct.pack("<I8H", *(self._v[f] for f in _FIELDS))

  def same_as(self, other) -> bool:
    """Equal, the two one-time flags (ack, save) aside."""
    a, b = Settings(self.content()), Settings(other.content())
    for s in (a, b):
      s["ack"] = s["save"] = 0
    return a.content() == b.content()

  def change(self, key, value):
    """Change one setting, refusing what could cut the link or make no sense."""
    if key in _LOCKED or key not in (*_SWITCHES, *_NUMBERS):
      raise ValueError(f"{key} can't be changed here")
    if key == "odr" and value not in _RATES:
      raise ValueError(f"rate {value}: the IMU knows {_RATES}")
    if key in ("gain_acc", "gain_mag") and not 100 <= value <= 400:
      raise ValueError(f"{key} {value}: 100 to 400 (100 = 1.00)")
    if key == "silent" and not 0 <= value <= 65:
      raise ValueError(f"silent {value}: 0 to 65 s")
    self[key] = value


# --------------------------------------------------------------------------------------
# Readings


class Sample(NamedTuple):
  t: float  # When it reached this PC, time.monotonic(), s.
  t_dev: float  # When the IMU took it, on its own clock (starts at its power-on), s.
  gyro: Vec3  # rad/s
  acc: Vec3  # g; still and level it reads (0, 0, -1), the same sense as gravity
  gravity: Vec3  # unit vector
  quat: Optional[
    Tuple[float, float, float, float]
  ]  # body to world (w, x, y, z), heading drifts

  def age(self) -> float:
    """Seconds since it reached this PC."""
    return time.monotonic() - self.t

  def tilt(self) -> float:
    """Angle to level, degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -self.gravity[2]))))


def find_port() -> str:
  found = sorted(glob.glob(PORTS))
  mine = [p for p in found if SERIAL and SERIAL in p]
  if mine:
    return mine[0]
  if len(found) == 1:
    return found[0]
  if not found:
    raise FileNotFoundError(
      "the IMU isn't plugged in (no STM32 port in /dev/serial/by-id)"
    )
  raise FileNotFoundError(
    "several STM32 ports, say which with --port:\n  " + "\n  ".join(found)
  )


def open_port(path: str) -> int:
  """The port in raw mode, closed to every other program."""
  fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
  try:
    fcntl.ioctl(fd, termios.TIOCEXCL)
    a = termios.tcgetattr(fd)
    a[0] = a[1] = a[3] = 0  # No translation, no echo.
    a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    a[4] = a[5] = termios.B921600  # Means nothing on USB.
    a[6][termios.VMIN] = a[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, a)
    termios.tcflush(fd, termios.TCIOFLUSH)
  except BaseException:
    os.close(fd)
    raise
  return fd


class TM171:
  """The IMU, read by a thread. Only listens, unless asked to read or change settings."""

  def __init__(self, port: Optional[str] = None, axes: str = "palm"):
    if axes not in ("palm", "imu"):
      raise ValueError(axes)
    self.port = port or find_port()
    self.axes = axes
    self.latest: Optional[Sample] = None
    self.history = collections.deque(maxlen=4000)
    self.error: Optional[str] = None
    self.count = self.lost = self.incomplete = 0
    self.quality = self.temperature = self.inner_rate = self.rpy = self.node = None
    self.parser = Parser()
    self._fd = None
    self._thread = None
    self._running = False
    self._news = threading.Condition()
    self._waiting = {}
    self._greeted = False
    self._relearn = False
    self._cycle = None
    self._parts = {}
    self._last_ts = None
    self._t_dev = 0.0
    self._learn()

  # Opening and closing

  def open(self, patience: float = 5.0):
    end = (
      time.monotonic() + patience
    )  # ModemManager may look at a port just plugged in.
    while True:
      try:
        self._fd = open_port(self.port)
        break
      except OSError as e:
        if e.errno != errno.EBUSY or time.monotonic() > end:
          raise
        time.sleep(0.25)
    self._running = True
    self._thread = threading.Thread(target=self._loop, name="tm171", daemon=True)
    self._thread.start()
    return self

  def close(self):
    self._running = False
    if self._thread is not None:
      self._thread.join(timeout=1.0)
      self._thread = None
    if self._fd is not None:
      os.close(self._fd)
      self._fd = None

  __enter__ = open

  def __exit__(self, *exc):
    self.close()

  # Using it

  def wait(self, timeout: float = 6.0) -> Sample:
    """The next new reading. After a cold start the IMU needs 3.2 s."""
    with self._news:
      before = self.count
      if not self._news.wait_for(lambda: self.count > before or self.error, timeout):
        raise TimeoutError("the IMU sends nothing (no gyro and gravity packets)")
      if self.error:
        raise OSError(self.error)
      return self.latest

  def fresh(self, limit: float = 0.05) -> bool:
    """True when the newest reading is younger than ``limit`` seconds."""
    s = self.latest
    return s is not None and not self.error and s.age() < limit

  def gyro_mean(self, seconds: float = 0.01) -> Vec3:
    """The gyro averaged over the last ``seconds``: what a 50 Hz loop misses in between."""
    last = self.latest
    picked = []
    for s in reversed(list(self.history)):
      if last.t_dev - s.t_dev >= seconds - 1e-6:
        break
      picked.append(s.gyro)
    return tuple(sum(c) / len(picked) for c in zip(*picked))

  # The thread

  def _loop(self):
    start = time.monotonic()
    while self._running:
      try:
        if not select.select([self._fd], [], [], 0.1)[0]:
          continue
        data = os.read(self._fd, 65536)
      except BlockingIOError:
        continue
      except (OSError, ValueError) as e:
        self.error = f"the port closed ({e})"
        break
      if not data:
        self.error = "the IMU was unplugged"
        break
      now = time.monotonic()
      for cmd, node, values in self.parser.feed(data):
        self.node = node
        # What waited in the IMU's buffer while nobody listened is seconds old.
        if cmd in (RAW, GRAVITY, QUAT, RPY) and now - start < 0.05:
          continue
        try:
          self._take(cmd, values, now)
        except struct.error:
          self.parser.bad_crc += 1
        except Exception as e:  # Never die in silence: fresh() turns false.
          self.error = f"reader stopped: {e!r}"
          self._running = False
          break
    with self._news:
      self._news.notify_all()

  def _learn(self):
    """What it sends and how often: learned from what arrives."""
    self._need = {RAW, GRAVITY}
    self._quats = self._quat_misses = 0
    self._steps = []
    self._step = None

  def _take(self, cmd, values, now):
    if self._relearn:  # After a change of settings.
      self._relearn = False
      self._learn()
    waiting = self._waiting.get(cmd)
    if waiting is not None:
      waiting.put(values)
    if cmd == STATUS:
      _, self.temperature, self.inner_rate, state = struct.unpack_from("<IfHH", values)
      self.quality = state & 7
    elif cmd == RPY:
      self.rpy = struct.unpack_from("<3f", values, 4)
    if cmd not in (RAW, GRAVITY, QUAT):
      return
    ts = struct.unpack_from("<I", values)[0]
    if ts != self._cycle:
      if self._parts and not self._need <= self._parts.keys():
        if self._last_ts is not None:
          self.incomplete += 1
        if self._need - self._parts.keys() == {QUAT}:  # It stopped sending quaternions.
          self._quat_misses += 1
          if self._quat_misses > 5:
            self._need.discard(QUAT)
            self._quats = self._quat_misses = 0
      self._cycle, self._parts = ts, {}
    if cmd in self._parts:
      return
    if cmd == QUAT and QUAT not in self._need:
      self._quats += 1
      if self._quats > 5:
        self._need.add(QUAT)
    ready = self._need <= self._parts.keys()
    self._parts[cmd] = values
    if not ready and self._need <= self._parts.keys() and ts != self._last_ts:
      self._emit(ts, now)

  def _emit(self, ts, now):
    if self._last_ts is not None:
      step = (ts - self._last_ts) & 0xFFFFFFFF  # Its clock starts over every 1.19 h.
      self._t_dev += step * 1e-6
      if self._step is None:
        self._steps.append(step)
        if len(self._steps) == 21:
          self._step = statistics.median(self._steps)
      elif step > 1.5 * self._step:
        self.lost += round(step / self._step) - 1
    else:
      self._t_dev = ts * 1e-6
    self._last_ts = ts
    raw = struct.unpack_from("<9f", self._parts[RAW], 4)
    gyro, acc = raw[:3], raw[3:6]
    grav = struct.unpack_from("<3f", self._parts[GRAVITY], 4)
    size = math.sqrt(sum(g * g for g in grav)) or 1.0
    grav = tuple(g / size for g in grav)
    quat = self._parts.get(QUAT)
    quat = struct.unpack_from("<4f", quat, 4) if quat is not None else None
    if self.axes == "palm":
      gyro, acc, grav = to_palm(gyro), to_palm(acc), to_palm(grav)
      quat = quat_to_palm(quat) if quat is not None else None
    sample = Sample(now, self._t_dev, gyro, acc, grav, quat)
    self.history.append(sample)
    with self._news:
      self.latest = sample
      self.count += 1
      self._news.notify_all()

  # Asking and changing

  def _send(self, data: bytes):
    if not self._greeted:  # Ends a packet it may have half received, then is ignored.
      self._greeted = True
      data = bytes(64) + data
    while data:
      try:
        data = data[os.write(self._fd, data) :]
      except BlockingIOError:
        select.select([], [self._fd], [], 0.1)

  def request(self, cmd: int, timeout: float = 2.0) -> bytes:
    """Ask for one object. Changes nothing on the IMU."""
    box = self._waiting[cmd] = queue.Queue()
    try:
      end = time.monotonic() + timeout
      while time.monotonic() < end and not self.error:
        self._send(packet(REQUEST, bytes([cmd, 0, 0, 0])))
        try:
          return box.get(timeout=0.1)
        except queue.Empty:
          pass
    finally:
      del self._waiting[cmd]
    raise TimeoutError(f"the IMU didn't answer (object {cmd})")

  def read_settings(self) -> Settings:
    return Settings(self.request(SETTINGS))

  def write_settings(self, new: Settings, save: bool = False) -> Settings:
    """Send settings and read them back. ``save``: also into its permanent memory.

    Without it they last until the IMU is unplugged. Nothing comes back that says the
    permanent memory took them: unplug, plug in, read them.
    """
    out = Settings(new.content())
    out["ack"], out["save"] = (
      0,
      int(save),
    )  # An ack flag would stay set: reading back is better.
    if self.node is None:
      self.wait()
    self._send(
      packet(SETTINGS, out.content(), to=self.node)
    )  # It ignores one sent to everyone.
    time.sleep(1.5 if save else 0.3)
    self._relearn = True
    back = self.read_settings()
    if not back.same_as(new):
      raise OSError("the IMU kept other settings than the ones sent")
    return back


# --------------------------------------------------------------------------------------
# Commands


def _bar(v, half=12):
  n = int(round(max(-1.0, min(1.0, v)) * half))
  left = " " * (half + min(0, n)) + "=" * -min(0, n)
  return left + "|" + "=" * max(0, n) + " " * (half - max(0, n))


def _row(name, v, scale):
  nums = "".join(f"{c:+9.3f}" for c in v)
  return f"  {name:<14}{nums}   " + "  ".join(_bar(c / scale) for c in v)


def cmd_live(imu, args):
  tty = sys.stdout.isatty()
  imu.wait()
  began, first, last_count, last_t, rate = (
    time.monotonic(),
    True,
    imu.count,
    time.monotonic(),
    0.0,
  )
  while not imu.error:
    time.sleep(0.1 if tty else 1.0)
    now = time.monotonic()
    if now - last_t >= 1.0:
      rate, last_count, last_t = (
        (imu.count - last_count) / (now - last_t),
        imu.count,
        now,
      )
    s = imu.latest
    up = int(now - began)
    quality = "?" if imu.quality is None else f"{imu.quality} of 5"
    heat = "" if imu.temperature is None else f"   {imu.temperature:.1f} C"
    raw = {
      "gravity": (s.gravity[1], -s.gravity[0], s.gravity[2]),
      "gyro": (s.gyro[1], -s.gyro[0], s.gyro[2]),
    }
    lines = [
      f"Walking Leap IMU   TM171 (node {imu.node})   {os.path.realpath(imu.port)}",
      f"{rate:5.0f} readings a second   lost {imu.lost}   newest {1e3 * s.age():4.1f} ms old"
      f"   quality {quality}{heat}   {up // 3600}:{up // 60 % 60:02}:{up % 60:02}",
      "",
      "PALM AXES (once it's on the hand)   x fingertips   y thumb side   z out of the back",
      _row("gravity", s.gravity, 1.0),
      _row("gyro rad/s", s.gyro, 4.0),
      _row("accel g", s.acc, 2.0),
      f"  tilt {s.tilt():6.2f} deg     accel size {math.sqrt(sum(a * a for a in s.acc)):.3f} g",
      "",
      "IMU'S OWN AXES (as drawn on it)     x pins         y left         z out of the lid",
      _row("gravity", raw["gravity"], 1.0),
      _row("gyro rad/s", raw["gyro"], 4.0),
      "",
      "Level, lid up: gravity (0, 0, -1) in both.   Ctrl+C stops.",
    ]
    if not imu.fresh(0.5):
      lines.append("!! NOTHING NEW FROM THE IMU: unplugged?")
    if tty:
      sys.stdout.write(
        ("\x1b[2J" if first else "")
        + "\x1b[H"
        + "".join(l + "\x1b[K\n" for l in lines)
        + "\x1b[J"
      )
      first = False
    else:
      print(
        lines[1]
        + f"   gravity {tuple(round(c, 3) for c in s.gravity)}   gyro {tuple(round(c, 3) for c in s.gyro)}"
      )
    sys.stdout.flush()
  print(imu.error)
  return 1


def _collect(imu, seconds):
  imu.wait()
  time.sleep(0.2)
  out, seen = [], imu.latest.t_dev
  lost0, inc0, crc0 = imu.lost, imu.incomplete, imu.parser.bad_crc
  end = time.monotonic() + seconds
  while time.monotonic() < end and not imu.error:
    time.sleep(0.2)
    new = [s for s in list(imu.history) if s.t_dev > seen]
    if new:
      seen = new[-1].t_dev
      out += new
  return out, imu.lost - lost0, imu.incomplete - inc0, imu.parser.bad_crc - crc0


def _pct(values, p):
  values = sorted(values)
  return values[min(len(values) - 1, int(p / 100 * len(values)))]


def cmd_check(imu, args):
  print(
    f"Listening for {args.seconds:.0f} s. Leave the IMU still for the noise numbers."
  )
  data, lost, incomplete, bad = _collect(imu, args.seconds)
  if len(data) < 50:
    print("almost nothing arrived:", imu.error)
    return 1
  span = data[-1].t - data[0].t
  rate = (len(data) - 1) / (data[-1].t_dev - data[0].t_dev)
  gaps = [1e3 * (b.t - a.t) for a, b in zip(data, data[1:])]
  lag = [s.t - s.t_dev for s in data]
  # The two clocks run at slightly different speeds: take the slow drift out first.
  third = max(1, len(lag) // 3)
  a, b = min(lag[:third]), min(lag[-third:])
  ta, tb = data[lag.index(a)].t, data[len(lag) - third + lag[-third:].index(b)].t
  slope = (b - a) / (tb - ta) if tb > ta else 0.0
  late = [1e3 * (l - a - slope * (s.t - ta)) for l, s in zip(lag, data)]
  low = min(late)
  late = [x - low for x in late]
  ok = lost == 0 and incomplete == 0 and bad == 0 and max(gaps) < 20
  print(
    f"\nreadings      {len(data)} in {span:.1f} s, {rate:.1f} a second by the IMU's clock"
  )
  print(f"lost          {lost}   incomplete {incomplete}   damaged packets {bad}")
  print(
    f"between two   typically {_pct(gaps, 50):.2f} ms, 1 in 100 over {_pct(gaps, 99):.2f}, worst {max(gaps):.2f}"
  )
  print(
    f"extra delay   typically {_pct(late, 50):.2f} ms, 1 in 100 over {_pct(late, 99):.2f}, worst {max(late):.2f} (above its fastest)"
  )
  print(
    f"clocks        the IMU's runs {1e6 * -slope:+.0f} millionths against this PC's"
  )
  print(
    f"quality       {imu.quality} of 5 (its own grade), inside it runs at {imu.inner_rate} Hz, {imu.temperature:.1f} C"
  )
  sd = [statistics.pstdev(c) for c in zip(*(s.gyro for s in data))]
  mean = [statistics.fmean(c) for c in zip(*(s.gyro for s in data))]
  pp = [max(c) - min(c) for c in zip(*(s.gyro for s in data))]
  if max(sd) > 0.02:
    print("noise         not judged, it was moving")
  else:
    g = [statistics.fmean(c) for c in zip(*(s.gravity for s in data))]
    n = math.sqrt(sum(c * c for c in g))
    off = [
      math.degrees(
        math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(s.gravity, g)) / n)))
      )
      for s in data
    ]
    acc = [math.sqrt(sum(c * c for c in s.acc)) for s in data]
    print("gyro          offset " + " ".join(f"{c:+.5f}" for c in mean) + " rad/s")
    print(
      "              noise  "
      + " ".join(f"{c:.5f}" for c in sd)
      + " rad/s (standard deviation)"
    )
    print(
      "              whole range "
      + " ".join(f"{math.degrees(c):.2f}" for c in pp)
      + " deg/s"
    )
    print(
      "gravity       "
      + " ".join(f"{c:+.4f}" for c in g)
      + f", wanders {max(off):.3f} deg at most"
    )
    print(f"accel size    {statistics.fmean(acc):.4f} g")
  print("\n" + ("GOOD: nothing lost." if ok else "LOOK: something was lost or late."))
  return 0 if ok else 1


_AXES = [
  # what to do, quantity, palm axis it must show on, sign
  ("Hand LEVEL, palm down (lid up), still", "gravity", 2, -1),
  ("FINGERTIPS DOWN: the fingers point to the floor", "gravity", 0, +1),
  ("THUMB SIDE DOWN", "gravity", 1, +1),
  ("UPSIDE DOWN (lid to the floor)", "gravity", 2, +1),
  ("Back to level, then TIP THE FINGERTIPS DOWN, briskly", "gyro", 1, +1),
  ("Level again, then RAISE THE THUMB SIDE, briskly", "gyro", 0, +1),
  ("Level again, then TURN LEFT as seen from above, briskly", "gyro", 2, +1),
]


class AxisJudge:
  """Looks at one move and says whether it showed on the right axis, with the right sign."""

  def __init__(self, name, axis, sign):
    self.name, self.axis, self.sign = name, axis, sign
    self.level = name == "gravity" and axis == 2 and sign < 0
    self.start = None  # Gravity when the step began.
    self.rest = False
    self.hold = (None, 0)

  def feed(self, s: Sample):
    """None until it's clear, then (right?, what it read)."""
    if self.start is None:
      self.start = s.gravity
    v = getattr(s, self.name)
    order = sorted(range(3), key=lambda k: -abs(v[k]))
    k, clean = order[0], abs(v[order[1]]) < 0.6 * abs(v[order[0]])
    seen = (k, v[k] > 0)
    if self.name == "gravity":
      moved = sum(a * b for a, b in zip(v, self.start)) < 0.7  # 45 degrees.
      still = max(abs(c) for c in s.gyro) < 0.5
      neutral = seen == (
        2,
        False,
      )  # Level is where one comes back to between two poses.
      ok = abs(v[k]) > 0.9 and still and (self.level or (moved and not neutral))
      need = 160
    else:  # A turn is judged from rest, on its first part.
      if not self.rest:
        self.rest = max(abs(c) for c in v) < 0.15
        return None
      if (
        abs(v[k]) > 0.6 and not clean
      ):  # Not around one axis: wait for rest, take the next.
        self.rest, self.hold = False, (None, 0)
        return None
      ok, need = abs(v[k]) > 1.0, 8
    if not ok:
      self.hold = (None, 0)
      return None
    self.hold = (seen, self.hold[1] + 1 if self.hold[0] == seen else 1)
    if self.hold[1] < need:
      return None
    return seen == (self.axis, self.sign > 0), v


def cmd_axes(imu, args):
  print(
    "The IMU on the hand, the hand's power OFF, the hand in your hands. Enter skips a step."
  )
  print(
    "Not on the hand yet? Hold the board lid up, pins away from you: the fingertips would be"
  )
  print("to your right, the thumb side away from you.\n")
  imu.wait()
  results = []
  for i, (what, name, axis, sign) in enumerate(_AXES, 1):
    print(f"{i}/7  {what}")
    judge, verdict, seen, shown = (
      AxisJudge(name, axis, sign),
      None,
      imu.latest.t_dev,
      0.0,
    )
    while verdict is None:
      if select.select([sys.stdin], [], [], 0.02)[0]:
        sys.stdin.readline()
        verdict = "skipped"
        print("     skipped" + " " * 60)
        break
      if imu.error:
        print(imu.error)
        return 1
      for s in [s for s in list(imu.history) if s.t_dev > seen]:
        seen = s.t_dev
        said = judge.feed(s)
        if said:
          verdict = "OK" if said[0] else "WRONG"
          print(
            f"     {verdict}: {name} reads "
            + " ".join(f"{a} {c:+.2f}" for a, c in zip("xyz", said[1]))
            + " " * 30
          )
          break
      if verdict is None and sys.stdout.isatty() and time.monotonic() - shown > 0.2:
        shown = time.monotonic()
        s = imu.latest
        print(
          "     gravity "
          + " ".join(f"{c:+.2f}" for c in s.gravity)
          + "   gyro "
          + " ".join(f"{c:+.2f}" for c in s.gyro)
          + "  ",
          end="\r",
          flush=True,
        )
    results.append(verdict)
  print()
  if results.count("OK") == 7:
    print("ALL SEVEN OK: the axes are the simulation's.")
    return 0
  print("Not all OK: " + ", ".join(f"{i} {r}" for i, r in enumerate(results, 1)))
  print(
    "With one axis WRONG the hand falls (every time, in the simulation): no policy until it is right."
  )
  return 1


def _angle(a, b):
  return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))))


def cmd_shake(imu, args):
  before = None
  if args.gain:
    before = imu.read_settings()
    new = Settings(before.content())
    new.change("gain_acc", args.gain)
    imu.write_settings(new)
    print(f"Accelerometer gain {args.gain / 100:.2f} for this test only.")
  try:
    print("Flat on the table, STILL for 5 s...", flush=True)
    time.sleep(2.0)
    rest, *_ = _collect(imu, 3.0)
    if max(statistics.pstdev(c) for c in zip(*(s.gyro for s in rest))) > 0.02:
      print("It moved: once more.")
      return 1
    down = [statistics.fmean(c) for c in zip(*(s.gravity for s in rest))]
    size = math.sqrt(sum(c * c for c in down))
    down = [c / size for c in down]
    pull = [statistics.fmean(c) for c in zip(*(s.acc for s in rest))]
    print(
      f"NOW slide it around and tap it for {args.seconds:.0f} s, always flat on the table.",
      flush=True,
    )
    data, lost, *_ = _collect(imu, args.seconds)
    print("Stop.\n")
    push = [math.sqrt(sum((a - b) ** 2 for a, b in zip(s.acc, pull))) for s in data]
    wrong = [_angle(s.gravity, down) for s in data]
    print(
      f"shaken by     {_pct(push, 50):.2f} g typically, 1 in 20 over {_pct(push, 95):.2f}, at most {max(push):.2f}"
    )
    print(
      f"down is off   {_pct(wrong, 50):.2f} deg typically, 1 in 20 over {_pct(wrong, 95):.2f}, at most {max(wrong):.2f}"
    )
    print(f"lost          {lost}")
    print(
      "It should be 0 deg: flat on the table, down doesn't move (lifting a corner counts as wrong)."
    )
  finally:
    if before is not None:
      imu.write_settings(before)
      print("The gain is back as it was.")
  return 0


def cmd_log(imu, args):
  path = args.file or os.path.join(
    OUT, "logs", datetime.now().strftime("imu-%Y%m%d-%H%M%S.csv")
  )
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  imu.wait()
  seen, n, began = imu.latest.t_dev, 0, time.monotonic()
  with open(path, "w", newline="") as f:
    out = csv.writer(f)
    out.writerow(
      [
        "t_pc",
        "t_imu",
        *(f"gyro_{c}" for c in "xyz"),
        *(f"acc_{c}" for c in "xyz"),
        *(f"gravity_{c}" for c in "xyz"),
        *(f"quat_{c}" for c in "wxyz"),
      ]
    )
    print(f"Writing {path} (palm axes). Ctrl+C stops.")
    try:
      while not imu.error and (
        not args.seconds or time.monotonic() - began < args.seconds
      ):
        time.sleep(0.1)
        for s in [s for s in list(imu.history) if s.t_dev > seen]:
          seen = s.t_dev
          n += 1
          out.writerow(
            [
              f"{s.t:.6f}",
              f"{s.t_dev:.6f}",
              *(f"{c:.6f}" for c in (*s.gyro, *s.acc, *s.gravity)),
              *(f"{c:.6f}" for c in (s.quat or ())),
            ]
          )
    except KeyboardInterrupt:
      pass
  print(f"{n} readings, {imu.lost} lost")
  return 0


def _say(key, value):
  if key == "boot":
    return ("auto", "static", "dynamic", "?")[value]
  if key == "odr":
    return f"{value} Hz"
  if key.startswith("gain"):
    return f"{value / 100:.2f}"
  if key == "silent":
    return f"{value} s"
  return "on" if value else "off"


_NAMES = {
  "gyro": "gyro",
  "acc": "accelerometer",
  "mag": "magnetometer",
  "gyro_err_filter": "gyro error filter",
  "gain_acc": "accelerometer gain",
  "boot": "boot mode",
  "silent": "silent time at start",
  "odr": "readings a second",
  "out_raw": "sends gyro+accel (raw)",
  "out_gravity": "sends gravity",
  "out_quat": "sends quaternion",
  "out_rpy": "sends roll/pitch/yaw",
  "out_euler": "sends Euler angles",
  "out_combo": "sends combo",
  "out_status": "sends status",
  "usb_on": "sends on USB",
  "uart1_on": "sends on its pins",
  "gain_mag": "magnetometer gain",
  "mag_self_adapt": "mag self-adapt filter",
}


def _show(s: Settings):
  differs = 0
  print(f"  {'setting':<24}{'now':<9}{'wanted':<9}why")
  for key, (want, why) in WANTED.items():
    mark = " " if s[key] == want else "*"
    differs += s[key] != want
    print(f"{mark} {_NAMES[key]:<24}{_say(key, s[key]):<9}{_say(key, want):<9}{why}")
  print(
    "  not used by us: "
    + ", ".join(
      f"{_NAMES[k]} {_say(k, s[k])}" for k in ("uart1_on", "gain_mag", "mag_self_adapt")
    )
    + f", pins at {s['uart1'] * 100} baud"
  )
  return differs


def cmd_settings(imu, args):
  s = imu.read_settings()
  differs = _show(s)
  print(f"\n{differs} of them not as wanted (*)." if differs else "\nAll as wanted.")
  return 0


def _backup(s: Settings, why):
  folder = os.path.join(OUT, "settings")
  os.makedirs(folder, exist_ok=True)
  path = os.path.join(folder, datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{why}.json")
  values = {k: s[k] for k in (*_SWITCHES, *_FIELDS[1:])}
  with open(path, "w") as f:
    json.dump({"content_hex": s.content().hex(), "values": values}, f, indent=1)
  return path


def _write(imu, new, args):
  now = imu.read_settings()
  if now.same_as(new) and not args.save:
    print("Already so, nothing sent.")
    return 0
  for key in (*_SWITCHES, *_NUMBERS):
    if now[key] != new[key] and key not in ("ack", "save"):
      print(f"  {_NAMES.get(key, key)}: {_say(key, now[key])} -> {_say(key, new[key])}")
  if args.save:
    print("This goes in the IMU's permanent memory. Don't unplug it meanwhile.")
    if not args.yes and input("Type yes: ").strip().lower() != "yes":
      print("Nothing sent.")
      return 1
  print("Before, kept in", _backup(now, "before"))
  imu.write_settings(new, save=args.save)
  print(
    "Done, read back the same."
    + ("" if args.save else " It lasts until the IMU is unplugged (--save keeps it).")
  )
  return 0


def cmd_apply(imu, args):
  new = imu.read_settings()
  for key, (want, _) in WANTED.items():
    if key not in _LOCKED:
      new.change(key, want)
  return _write(imu, new, args)


def cmd_set(imu, args):
  new = imu.read_settings()
  for item in args.change:
    key, _, value = item.partition("=")
    new.change(key, int(value))
  return _write(imu, new, args)


def cmd_restore(imu, args):
  with open(args.file) as f:
    new = Settings(bytes.fromhex(json.load(f)["content_hex"]))
  new["ack"] = new["save"] = 0
  return _write(imu, new, args)


def main():
  parser = argparse.ArgumentParser(
    prog="wlimu", description="The TM171 IMU of Walking Leap."
  )
  parser.add_argument("--port", help="its port (found by itself)")
  sub = parser.add_subparsers(dest="cmd", metavar="command")
  sub.add_parser("live", help="what it reads, live (the default)")
  p = sub.add_parser("check", help="rate, losses, delay and noise")
  p.add_argument("--seconds", type=float, default=20)
  sub.add_parser("axes", help="the seven moves that prove the axes, on the hand")
  p = sub.add_parser("shake", help="how wrong its down gets when it's knocked around")
  p.add_argument("--seconds", type=float, default=15)
  p.add_argument(
    "--gain",
    type=int,
    help="try another accelerometer gain, e.g. 280 for 2.80, for this test only",
  )
  p = sub.add_parser("log", help="write the readings in a CSV file")
  p.add_argument("file", nargs="?")
  p.add_argument("--seconds", type=float)
  sub.add_parser(
    "settings", help="its settings, against the ones we want (changes nothing)"
  )
  for name, text in (
    ("apply", "give it the settings we want"),
    ("set", "change some, e.g. odr=200"),
    ("restore", "put a backup back"),
  ):
    p = sub.add_parser(name, help=text)
    if name == "set":
      p.add_argument("change", nargs="+", metavar="name=value")
    if name == "restore":
      p.add_argument("file")
    p.add_argument("--save", action="store_true", help="also in its permanent memory")
    p.add_argument("--yes", action="store_true", help="don't ask before --save")
  args = parser.parse_args()
  cmd = args.cmd or "live"
  try:
    with TM171(port=args.port) as imu:
      return globals()["cmd_" + cmd](imu, args)
  except KeyboardInterrupt:
    print()
    return 0
  except PermissionError as e:
    print(
      f"not allowed to open {e.filename}: this user isn't in the port's group ('ls -l' it)"
    )
  except (OSError, ValueError) as e:
    busy = getattr(e, "errno", None) == 16
    print(
      "another program is already reading the IMU: close it first" if busy else f"{e}"
    )
  return 1


if __name__ == "__main__":
  sys.exit(main())
