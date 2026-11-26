"""Checks of the TM171 IMU's reader (real/imu/tm171.py) that need no IMU. Linux or macOS."""

# ruff: noqa: B905, B007, B023
import math
import os
import struct
import sys
import time

sys.path.insert(
  0,
  os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "real", "imu"
  ),
)
from tm171 import (  # noqa: E402
  GRAVITY,
  QUAT,
  RAW,
  SETTINGS,
  STATUS,
  TM171,
  Parser,
  Settings,
  crc16,
  gravity_from_quat,
  packet,
  quat_to_palm,
  to_palm,
)

# Packets made here, in the maker's format: one of each that the reader takes.
EXAMPLES = {
  "raw": packet(
    RAW,
    struct.pack(
      "<I9f",
      2199820972,
      0.000703433,
      0.25,
      -0.5,
      0.01,
      -1.00012,
      0.02,
      0.1,
      0.2,
      0.790234,
    ),
  ),
  "quat": packet(QUAT, struct.pack("<I4f", 7, 0.9, 0.1, -0.3, 0.2)),
  "gravity": packet(GRAVITY, struct.pack("<I3f", 7, 0.0, -0.6, -0.8)),
  "status": packet(STATUS, struct.pack("<IfHH", 7, 20.0, 800, 5)),
}


def parse(data, size=None):
  p = Parser()
  out = []
  for i in range(0, len(data), size or len(data)):
    out += list(p.feed(data[i : i + (size or len(data))]))
  return p, out


def test_packets():
  for name, data in EXAMPLES.items():
    p, out = parse(data)
    assert len(out) == 1 and p.bad_crc == 0, name
  cmd, node, v = parse(EXAMPLES["raw"])[1][0]
  assert cmd == RAW
  ts, *g = struct.unpack("<I9f", v)
  assert (
    ts == 2199820972
    and abs(g[0] - 0.000703433) < 1e-8
    and abs(g[4] + 1.00012) < 1e-5
    and abs(g[8] - 0.790234) < 1e-6
  )
  assert struct.unpack_from("<IfHH", parse(EXAMPLES["status"])[1][0][2])[3] & 7 == 5


def test_broken_stream():
  good = EXAMPLES["raw"]
  bad = bytearray(good)
  bad[20] ^= 1
  stream = (
    b"\x01\xaa" + good[7:] + good + bytes(bad) + b"\xaa\x55\x03" + good + good[:10]
  )
  for size in (1, 3, 64, None):
    p, out = parse(stream, size)
    assert len(out) == 2, (size, len(out))
  assert crc16(b"") == 0xFFFF


def test_axes():
  # The IMU's x (pins) points to the thumb side: the palm's y. Its y points to the wrist.
  assert to_palm((1, 0, 0)) == (0, 1, 0)
  assert to_palm((0, 1, 0)) == (-1, 0, 0)
  assert to_palm((0, 0, 1)) == (0, 0, 1)
  # A quaternion and its gravity vector must turn together, for any pose.
  for q in (
    (0.9, 0.1, -0.3, 0.2),
    (0.1, 0.7, 0.6, -0.4),
    (-0.2, 0.3, 0.1, 0.9),
    (1, 0, 0, 0),
  ):
    n = math.sqrt(sum(c * c for c in q))
    q = tuple(c / n for c in q)
    a = to_palm(gravity_from_quat(q))
    b = gravity_from_quat(quat_to_palm(q))
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-9, q
    assert abs(sum(c * c for c in quat_to_palm(q)) - 1) < 1e-9
  assert gravity_from_quat((1, 0, 0, 0)) == (0, 0, -1)


# What our IMU answered, as found.
FOUND = bytes.fromhex("7bb8180000001027c409d0005d000a0000009001")


def test_settings():
  s = Settings(FOUND)
  assert s.content() == FOUND
  assert (s["gyro"], s["acc"], s["mag"]) == (1, 1, 0)
  assert (
    s["gyro_err_filter"],
    s["gain_acc"],
    s["gain_mag"],
    s["odr"],
    s["boot"],
    s["silent"],
  ) == (1, 208, 93, 400, 0, 0)
  assert (
    s["out_status"],
    s["out_raw"],
    s["out_quat"],
    s["out_euler"],
    s["out_rpy"],
    s["out_gravity"],
    s["out_combo"],
  ) == (1, 1, 0, 0, 1, 1, 0)
  assert (s["binary"], s["checksum"], s["uart1_on"], s["usb_on"], s["uart1"]) == (
    1,
    2,
    1,
    1,
    10000,
  )
  s.change("out_rpy", 0)
  s.change("out_quat", 1)
  assert s.content() == bytes.fromhex("7bb1180000001027c409d0005d000a0000009001")
  assert not s.same_as(Settings(FOUND))
  t = Settings(FOUND)
  t["ack"] = t["save"] = 1
  assert t.same_as(Settings(FOUND))
  for key, value in (
    ("usb_on", 0),
    ("binary", 0),
    ("save", 1),
    ("odr", 300),
    ("gain_acc", 20),
    ("uart1", 1152),
    ("silent", 99),
  ):
    try:
      s.change(key, value)
      raise SystemExit(f"{key}={value} was accepted")
    except ValueError:
      pass
  try:
    Settings(FOUND + b"\0")
    raise SystemExit("21 bytes were accepted")
  except ValueError:
    pass


def cycle(ts, quat=True, gyro=(0.1, 0.2, 0.3)):
  down = (0.0, -0.6, -0.8)
  out = packet(RAW, struct.pack("<I9f", ts & 0xFFFFFFFF, *gyro, *down, 0, 0, 0))
  if quat:
    out += packet(QUAT, struct.pack("<I4f", ts & 0xFFFFFFFF, 1, 0, 0, 0))
  out += packet(GRAVITY, struct.pack("<I3f", ts & 0xFFFFFFFF, *down))
  return out + packet(STATUS, struct.pack("<IfHH", ts & 0xFFFFFFFF, 20.0, 800, 5))


def feed(imu, data, now):
  for cmd, node, values in imu.parser.feed(data):
    imu._take(cmd, values, now)


def test_readings():
  imu = TM171(port="none")
  ts = 0xFFFFFFFF - 20000  # Goes over the 1.19 h roll-over.
  for k in range(400):
    if k not in (150, 151):  # Two lost.
      feed(imu, cycle(ts + 2500 * k), 0.0025 * k)
  assert (imu.count, imu.lost, imu.incomplete) == (398, 2, 0), (
    imu.count,
    imu.lost,
    imu.incomplete,
  )
  s = imu.latest
  assert abs(s.t_dev - (ts * 1e-6 + 399 * 0.0025)) < 1e-6
  assert (
    max(abs(a - b) for a, b in zip(s.gravity, (0.6, 0.0, -0.8))) < 1e-6
  )  # Fingertips down.
  assert max(abs(a - b) for a, b in zip(s.gyro, (-0.2, 0.1, 0.3))) < 1e-6
  assert s.quat is not None and imu.quality == 5
  # Without quaternions it keeps going.
  for k in range(400, 440):
    feed(imu, cycle(ts + 2500 * k, quat=False), 0.0025 * k)
  assert imu.latest.quat is None and imu.count >= 398 + 30
  # The mean over the last 10 ms takes 4 readings.
  imu = TM171(port="none", axes="imu")
  for k in range(100):
    feed(imu, cycle(2500 * k, gyro=(float(k), 0, 0)), 0.0025 * k)
  assert abs(imu.gyro_mean(0.01)[0] - 97.5) < 1e-6
  # A status in the wrong place or a cut cycle never gives a reading.
  imu = TM171(port="none")
  for k in range(50):
    feed(imu, packet(RAW, struct.pack("<I9f", 2500 * k, *[0.0] * 9)), 0.0025 * k)
  assert imu.count == 0 and imu.latest is None and not imu.fresh()


def turned(axis, sign, size, n):
  v = [0.0, 0.0, 0.0]
  v[axis] = sign * size
  return [tuple(v)] * n


def test_axis_judge():
  """Right, flipped and swapped moves. Coming back to level in between isn't judged."""
  from tm171 import _AXES, AxisJudge, Sample

  level = (0.0, 0.0, -1.0)
  still = (0.0, 0.0, 0.0)
  for what, name, axis, sign in _AXES:
    for k in range(3):
      for s in (1, -1):
        if name == "gravity":
          first = axis == 2 and sign < 0  # The first step: level itself.
          if not first and (k, s) == (2, -1):
            continue
          poses = ([] if first else [level] * 400) + turned(k, s, 1.0, 400)
          data = [Sample(0, 0, still, g, g, None) for g in poses]
        else:
          gyro = [still] * 40 + turned(k, s, 0.8, 5) + turned(k, s, 2.0, 60)
          data = [Sample(0, 0, w, level, level, None) for w in gyro]
        judge, said = AxisJudge(name, axis, sign), None
        for d in data:
          said = said or judge.feed(d)
        assert said is not None and said[0] == ((k, s) == (axis, sign)), (
          what,
          k,
          s,
          said,
        )
  # Around no clear axis, or not from rest: nothing is said.
  for before, w in ((40, (1.5, 1.5, 0.2)), (0, (0.0, 2.0, 0.0))):
    judge = AxisJudge("gyro", 1, 1)
    for d in [still] * before + [w] * 100:
      assert judge.feed(Sample(0, 0, d, level, level, None)) is None


def test_request_and_write():
  """Against a pretend IMU on a pseudo-terminal: it answers and obeys, or doesn't obey."""
  import pty
  import threading

  for obeys in (True, False):
    master, slave = pty.openpty()
    state = {"content": FOUND, "saved": 0, "stop": False}

    def device():
      p, k = Parser(), 0
      os.set_blocking(master, False)
      while not state["stop"]:
        try:
          data = os.read(master, 4096)
        except BlockingIOError:
          data = b""
        except OSError:
          return
        for cmd, _, v in p.feed(data):
          to = (
            struct.unpack_from("<I", data, data.index(bytes([0xAA, 0x55])) + 3)[0] >> 21
          )
          if cmd == 12 and v[0] == SETTINGS:
            os.write(master, packet(SETTINGS, state["content"]))
          elif (
            cmd == SETTINGS and to and obeys
          ):  # Like the real one: not if sent to everyone.
            s = Settings(v)
            state["saved"] += s["save"]
            assert s["ack"] == 0
            s["save"] = 0
            state["content"] = s.content()
        try:
          os.write(master, cycle(2500 * k))
        except BlockingIOError:
          pass
        k += 1
        time.sleep(0.0025)

    thread = threading.Thread(target=device, daemon=True)
    thread.start()
    with TM171(port=os.ttyname(slave)) as imu:
      imu.wait(2.0)
      assert imu.read_settings().content() == FOUND
      new = Settings(FOUND)
      new.change("odr", 200)
      try:
        assert imu.write_settings(new)["odr"] == 200 and obeys and state["saved"] == 0
        imu.write_settings(Settings(FOUND), save=True)
        assert state["saved"] == 1 and imu.read_settings().content() == FOUND
      except OSError:
        assert not obeys
      imu.wait(2.0)
      assert imu.fresh() and imu.error is None
      state["stop"] = True
      thread.join()
      os.close(master)  # Unplugged: it says so instead of giving old readings.
      time.sleep(0.4)
      assert imu.error and not imu.fresh()
    os.close(slave)


if __name__ == "__main__":
  for name, fn in list(globals().items()):
    if name.startswith("test_"):
      fn()
      print("ok ", name)
