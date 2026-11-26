"""Walking Leap control: one loop for scripts and policies on the real hand, with guards and full logs.

Joint order everywhere here: the repo's (JOINT_NAMES below). Only `Hand` knows the motors' order.
Angles in rad (0 = flat hand, + closes to the palm), gravity and gyro in the palm's axes (x to the
fingertips, y to the thumb side, z up), as in the simulation.

The motors are driven through `Hand`, the only class that talks to them. It uses dexhw, the LEAP
driver of the lab's hand PC (not part of this repo). Another driver fits if it keeps `Hand`'s methods
and, like dexhw, puts the motors in current-based position mode (5) with a 350 mA limit before torque.
"""

# ruff: noqa: B905
import dataclasses
import hashlib
import json
import os
import sys
import threading
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE + "/imu")
# The driver's settings for this hand (its limits, the 350 mA, the watchdog); the gains are set below.
RIG = os.environ.get("WL_RIG", os.path.expanduser("~/dexhw/rigs/leap_right.yaml"))
STOP = HERE + "/STOP"
CLIPS = os.environ.get("WL_CLIPS", REPO + "/docs/data/real_clips")
LOGS = os.environ.get("WL_LOGS", REPO + "/outputs/real/logs")
POLICIES = os.environ.get("WL_POLICIES", REPO + "/outputs/real/policies")
HZ = 50
JOINT_NAMES = (
  "if_mcp",
  "if_rot",
  "if_pip",
  "if_dip",
  "mf_mcp",
  "mf_rot",
  "mf_pip",
  "mf_dip",
  "rf_mcp",
  "rf_rot",
  "rf_pip",
  "rf_dip",
  "th_cmc",
  "th_axl",
  "th_mcp",
  "th_ipl",
)
MOTOR_OF = np.array(
  (1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15)
)  # repo joint i -> motor id
JOINT_OF = np.argsort(MOTOR_OF)  # motor id -> repo joint
# Where the thumb pressed on the table in a first pose (motor 12 at -0.17, motor 14 at -0.90, both at the
# current limit): kept out of reach. Each joint alone goes further in the air (-0.30, -1.11).
SAFE_LOW = {"th_cmc": -0.10, "th_mcp": -0.70}


class Abort(Exception):
  """Something is wrong: hold, then torque off."""


class SoftStop(Exception):
  """Asked to stop: lie down if it can, then torque off."""


def ease(u):
  u = min(max(u, 0.0), 1.0)
  return u**3 * (10 - 15 * u + 6 * u * u)


# ---------------------------------------------------------------------------- the hand
class Hand:
  """dexhw's LeapHand (or its fake) in the repo's joint order. 350 mA, its limits, its watchdog.

  What the loop uses: connect and disconnect (`with`), read() -> state (t, hw_error), q, dq, current in
  mA and the target sent, all in the repo's order; target(q); hold(); temps(); lo and hi; h.stats.
  """

  def __init__(self, gains=(1200, 0, 600), fake=False, q0=None, probe=False):
    from dexhw.leap.config import load_rig

    cfg = dataclasses.replace(load_rig(RIG).hand, kp=gains[0], ki=gains[1], kd=gains[2])
    assert cfg.current_limit_ma == 350, "the current limit is 350 mA"
    self.cfg, self.gains, self.fake = cfg, gains, fake
    if fake:
      from dexhw.leap.fake import FakeLeapHand

      b = None if q0 is None else np.asarray(q0)[JOINT_OF]
      self.h = FakeLeapHand(cfg, q0=b)
    else:
      from dexhw.leap.hand import LeapHand

      self.h = LeapHand(cfg)
    self.lo, self.hi = self.h.lower[MOTOR_OF].copy(), self.h.upper[MOTOR_OF].copy()
    if not probe:
      for n, v in SAFE_LOW.items():
        i = JOINT_NAMES.index(n)
        self.lo[i] = max(self.lo[i], v)

  def __enter__(self):
    self.h.connect()
    return self

  def __exit__(self, *e):
    self.h.disconnect()

  def read(self):
    s = self.h.read()
    return s, s.q[MOTOR_OF], s.dq[MOTOR_OF], s.current_ma[MOTOR_OF], s.q_cmd[MOTOR_OF]

  def target(self, q):
    q = np.clip(q, self.lo, self.hi)
    self.h.set_target(q[JOINT_OF])
    return q

  def hold(self):
    self.h.hold()

  def temps(self):
    """Motor temperatures (C), read by the hand's own loop between two ticks."""
    if self.fake:
      return np.full(16, 25.0)
    from dexhw.leap.third_party.leap_hand_utils import dynamixel_client as dxc

    out = {}

    def job():
      r = dxc.DynamixelReader(self.h._sdk(), self.h._ids, 146, 1)
      out["t"] = np.array(self.h._read_with_retry(r, "temperatures"), float)

    try:
      self.h._call(job)
    except Exception as e:  # a lost packet here is no reason to fall
      print("   temperatures not read:", e)
      return np.full(16, np.nan)
    return out["t"][MOTOR_OF]


class FakeImu:
  """Always level and still."""

  quality, error, lost, count = 5, None, 0, 0

  def __init__(self):
    from tm171 import Sample

    self.S, self.history, self.g = Sample, deque(maxlen=4000), (0.0, 0.0, -1.0)
    self.silent = False

  def __enter__(self):
    return self

  def __exit__(self, *e):
    pass

  def wait(self, timeout=0):
    return self.latest

  @property
  def latest(self):
    t = time.monotonic()
    return self.S(t, t, (0.0, 0.0, 0.0), self.g, self.g, (1.0, 0.0, 0.0, 0.0))

  def fresh(self, limit=0.05):
    return not self.silent

  def gyro_mean(self, s=0.01):
    return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------- the logs
class Recorder:
  """Every motor state the bus gives (100 Hz), every IMU reading (400 Hz), every command (50 Hz)."""

  MOTOR = "t q*16 dq*16 cur_mA*16 setpoint*16"
  IMU = "t t_dev gyro*3 acc*3 gravity*3 quat*4"
  CMD = "t clip frame sent*16 clip_q*16 sim_gravity*3 policy_q*16 loop_late_ms"

  def __init__(self, hand, imu, name, meta):
    self.hand, self.imu, self.name, self.meta = hand, imu, name, meta
    self.m, self.i, self.c, self.events = [], [], [], []
    self.t0 = time.monotonic()
    self._run, self._seen = True, 0.0
    self.th = threading.Thread(target=self._loop, daemon=True)
    self.th.start()

  def _loop(self):
    last = -1.0
    while self._run:
      try:
        s, q, dq, cur, qc = self.hand.read()
        if s.t != self._seen:
          self._seen = s.t
          self.m.append(np.r_[s.t - self.t0, q, dq, cur, qc])
      except Exception:
        pass
      try:
        new = [x for x in list(self.imu.history) if x.t_dev > last]
      except RuntimeError:  # grew while we looked: next time
        new = []
      for x in new:
        last = x.t_dev
        self.i.append(
          np.r_[
            x.t - self.t0, x.t_dev, x.gyro, x.acc, x.gravity, x.quat or (np.nan,) * 4
          ]
        )
      time.sleep(0.004)

  def event(self, text):
    t = time.monotonic() - self.t0
    self.events.append((t, text))
    print(f"{t:6.1f} s  {text}", flush=True)

  def close(self):
    self._run = False
    self.th.join(1.0)
    os.makedirs(LOGS, exist_ok=True)
    p = f"{LOGS}/{time.strftime('%Y%m%d_%H%M%S')}_{self.name}.npz"
    self.meta["events"] = self.events
    np.savez_compressed(
      p,
      motor=np.array(self.m, np.float32),
      imu=np.array(self.i, np.float64),
      cmd=np.array(self.c, np.float32),
      meta=json.dumps(self.meta),
    )
    print(
      "log",
      p,
      f"({len(self.m)} motor rows, {len(self.i)} IMU rows, {len(self.c)} commands)",
    )
    return p


# ---------------------------------------------------------------------------- clips
class Clip:
  """Joint targets at 50 Hz, with what the sim's IMU read for them."""

  def __init__(self, name):
    path = name if name.endswith(".npz") else f"{CLIPS}/{name}.npz"
    d = np.load(path, allow_pickle=False)
    self.name = os.path.basename(path)[:-4]
    self.q, self.grav = d["q"].astype(float), d["grav"].astype(float)
    self.ref_grav = d["ref_grav"].astype(float) if "ref_grav" in d else self.grav
    self.meta = json.loads(str(d["meta"]))
    assert tuple(self.meta["joint_names"]) == JOINT_NAMES and self.meta["fps"] == HZ
    assert self.q.shape == (len(self.grav), 16) and np.isfinite(self.q).all()
    v = np.abs(np.diff(self.q, axis=0)).max() * HZ if len(self.q) > 1 else 0
    assert v < 6.0, f"{name}: {v:.1f} rad/s between two frames"
    self.start, self.end = self.meta["start"], self.meta["end"]
    self.sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


# ---------------------------------------------------------------------------- the loop
class Control:
  def __init__(self, hand, imu, rec, max_tilt=40.0, max_off=15.0, sat_s=1.5):
    self.hand, self.imu, self.rec = hand, imu, rec
    self.max_tilt, self.max_off, self.sat_s = max_tilt, max_off, sat_s
    self.q = None  # last target sent
    self.sat = np.zeros(16)  # s each motor has been at its cap
    self.off_for = 0.0
    self.next = None
    self.where = "lying"
    self.policy, self.live = None, False
    self.worst_off = 0.0
    self.clip_no = -1

  def guard(self, pred):
    s, q, dq, cur, qc = self.hand.read()
    if s.hw_error.any():
      raise Abort(f"motor error {s.hw_error.tolist()}")
    if not self.imu.fresh(0.06):
      raise Abort("the IMU is silent")
    if time.monotonic() - s.t > 0.1:
      raise Abort("the motors are silent")
    g = np.array(self.imu.latest.gravity)
    tilt = float(np.degrees(np.arccos(np.clip(-g[2], -1, 1))))
    if tilt > self.max_tilt:
      raise Abort(f"tilt {tilt:.0f} deg")
    if pred is not None:
      off = float(np.degrees(np.arccos(np.clip(g @ pred, -1, 1))))
      self.worst_off = max(self.worst_off, off)
      self.off_for = self.off_for + 1 / HZ if off > self.max_off else 0.0
      if self.off_for > 0.06:
        raise Abort(f"{off:.0f} deg away from the sim")
    self.sat = np.where(np.abs(cur) > 330, self.sat + 1 / HZ, 0.0)
    if self.sat.max() > self.sat_s:
      raise Abort(
        f"{JOINT_NAMES[self.sat.argmax()]} at its current limit for {self.sat_s} s"
      )
    if os.path.exists(STOP):
      os.remove(STOP)
      raise SoftStop("STOP file")
    return s, q, dq, cur, g

  def step(self, target, pred=None, clip=-1, frame=-1, policy_q=None):
    """Send one target, wait for the 50 Hz tick, check everything."""
    now = time.monotonic()
    if self.next is None or now - self.next > 0.5:
      self.next = now
    late = max(0.0, now - self.next) * 1e3
    self.q = self.hand.target(target)
    nan3, nan16 = (np.nan,) * 3, (np.nan,) * 16
    self.rec.c.append(
      np.r_[
        now - self.rec.t0,
        clip,
        frame,
        self.q,
        target,
        nan3 if pred is None else pred,
        nan16 if policy_q is None else policy_q,
        late,
      ]
    )
    self.next += 1 / HZ
    time.sleep(max(0.0, self.next - time.monotonic()))
    return self.guard(pred)

  def start(self):
    s, q, *_ = self.hand.read()
    self.q = np.clip(q, self.hand.lo, self.hand.hi)
    for _ in range(HZ // 2):
      self.step(self.q)

  def hold(self, T):
    for _ in range(int(T * HZ)):
      self.step(self.q)

  def ramp(self, q, speed=0.5):
    q = np.clip(q, self.hand.lo, self.hand.hi)
    a, d = self.q.copy(), float(np.abs(q - self.q).max())
    n = max(1, int(1.9 * d / speed * HZ))
    for k in range(1, n + 1):
      self.step(a + (q - a) * ease(k / n))

  def play(self, clip, speed=1.0):
    if clip.start != self.where:
      raise Abort(f"{clip.name} starts {clip.start}, the hand is {self.where}")
    far = float(np.abs(clip.q[0] - self.q).max())
    if self.where == "stand" and far > 0.4:
      raise SoftStop(f"{clip.name} starts {far:.2f} rad from the pose now")
    self.ramp(clip.q[0], 0.3 if self.where == "stand" else 0.5)
    self.worst_off, self.clip_no = 0.0, self.clip_no + 1
    n0, t0 = len(self.rec.m), time.monotonic()
    policy = (
      self.policy
      if (self.policy is not None and clip.name.startswith(self.policy.clip))
      else None
    )
    if policy is not None:
      policy.reset(clip)
    pol = []
    n = len(clip.q) if speed == 1.0 else int((len(clip.q) - 1) / speed) + 1
    self.where = "moving"
    for k in range(n):
      x = min(k * speed, len(clip.q) - 1)
      i = int(x)
      j = min(i + 1, len(clip.q) - 1)
      tgt = clip.q[i] + (clip.q[j] - clip.q[i]) * (x - i)
      pq = None
      if policy is not None:
        s, q, dq, cur, qc = self.hand.read()
        pq = policy.act(i, q, dq, self.imu.gyro_mean(0.01), self.imu.latest.gravity)
        pol.append(np.abs(pq - tgt))
        if self.live:
          tgt = pq
      self.step(tgt, clip.grav[i] if speed == 1.0 else None, self.clip_no, i, pq)
    self.where = clip.end
    cur = np.array([r[33:49] for r in self.rec.m[n0:]])
    g = self.imu.latest.gravity
    tilt = np.degrees(np.arccos(np.clip(-g[2], -1, 1)))
    msg = f"{clip.name}: done in {time.monotonic() - t0:.1f} s, tilt {tilt:.1f} deg"
    if speed == 1.0:
      msg += f", at most {self.worst_off:.1f} deg from the sim"
    if len(cur):
      j = int(np.abs(cur).max(0).argmax())
      msg += f", peak {np.abs(cur).max():.0f} mA ({JOINT_NAMES[j]})"
    if pol:
      msg += (
        f", policy {'SENT' if self.live else 'in the shadow'}: {np.mean(pol):.3f} rad from the clip on average, "
        f"worst {np.max(pol):.2f}"
      )
    t = self.hand.temps()
    msg += f", {np.nanmin(t):.0f}-{np.nanmax(t):.0f} C"
    self.next = None
    self.rec.event(msg)
    if np.nanmax(t) > 60:
      raise SoftStop(f"a motor is at {np.nanmax(t):.0f} C")


def soft_down(ctl):
  """Level and not fallen: go to the stand and lie down."""
  down = Clip("lie_down")
  if np.degrees(np.arccos(np.clip(-ctl.imu.latest.gravity[2], -1, 1))) > 15:
    raise Abort("too tilted to lie down")
  ctl.ramp(down.q[0], 0.6)
  ctl.where, ctl.live, ctl.policy = "stand", False, None
  ctl.play(down)


def run(
  names,
  gains=(1200, 0, 600),
  fake=False,
  speed=1.0,
  policy=None,
  live=False,
  label="",
  max_off=15.0,
  rest=0.4,
  hooks=None,
  devices=None,
  start="lying",
):
  """Play clips one after the other, from lying. Torque is off when it returns."""
  clips = [Clip(n) for n in names]
  for a, b in zip(clips, clips[1:]):
    if a.end != b.start:
      raise SystemExit(f"{a.name} ends {a.end}, {b.name} starts {b.start}")
    if a.end == "stand" and np.abs(a.q[-1] - b.q[0]).max() > 0.05:
      raise SystemExit(
        f"{a.name} ends {np.abs(a.q[-1] - b.q[0]).max():.2f} rad from where {b.name} starts: not starting"
      )
  if clips[0].start != start:
    raise SystemExit(f"{clips[0].name} starts {clips[0].start}, the hand is {start}")
  if os.path.exists(STOP):
    os.remove(STOP)
  if devices is not None:
    imu, hand = devices
  elif fake:
    imu, hand = FakeImu(), Hand(gains, True, clips[0].q[0])
  else:
    from dexhw.leap.hand import raise_on_signals
    from tm171 import TM171

    raise_on_signals()
    imu, hand = TM171(), Hand(gains)
  name = label or "_".join(dict.fromkeys(c.name for c in clips))[:60]
  with imu:
    imu.wait()
    time.sleep(0.0 if (fake or devices) else 1.2)
    if (imu.quality or 0) < 4:
      raise SystemExit(f"IMU quality {imu.quality}: leave it still half a minute")
    if not fake and devices is None and imu.latest.tilt() > 10:
      raise SystemExit(f"not lying level ({imu.latest.tilt():.0f} deg): not starting")
    with hand:
      meta = dict(
        when=time.strftime("%Y-%m-%d %H:%M:%S"),
        clips=[(c.name, c.sha) for c in clips],
        gains=gains,
        speed=speed,
        fake=fake,
        live_policy=live,
        policy=getattr(policy, "name", None),
        joint_names=JOINT_NAMES,
        current_limit_ma=350,
        columns=dict(motor=Recorder.MOTOR, imu=Recorder.IMU, cmd=Recorder.CMD),
      )
      rec = Recorder(hand, imu, name, meta)
      ctl = Control(
        hand, imu, rec, max_off=180.0 if fake else max_off
      )  # the fake IMU is always level
      ctl.policy, ctl.live, ctl.where = policy, live, start
      ok = False
      if hooks:
        hooks(ctl)
      try:
        ctl.start()
        meta["temps_before"] = hand.temps().tolist()
        rec.event(
          f"torque on, tilt {np.degrees(np.arccos(min(1, -imu.latest.gravity[2]))):.1f} deg, "
          f"{np.nanmin(meta['temps_before']):.0f}-{np.nanmax(meta['temps_before']):.0f} C"
        )
        for c in clips:
          ctl.play(c, speed)
          ctl.hold(rest)
        ok = True
      except SoftStop as e:
        rec.event(f"stopping: {e}")
        try:
          soft_down(ctl)
        except (Abort, SoftStop) as e2:
          rec.event(f"ABORT on the way down: {e2}")
      except Abort as e:
        rec.event(f"ABORT: {e} -> hold, then torque off")
        try:
          hand.hold()
          time.sleep(0.3)
        except Exception as x:
          print(x)
      except (KeyboardInterrupt, SystemExit):
        rec.event("interrupted -> torque off")
      finally:
        st = hand.h.stats
        meta.update(
          ok=ok,
          loop_hz=st.hz,
          comm_errors=st.comm_errors,
          retries=st.bus_retries,
          ticks=st.ticks,
          imu_lost=imu.lost,
          where=ctl.where,
        )
        path = rec.close()
        late = np.array([c[-1] for c in rec.c]) if rec.c else np.zeros(1)
        print(
          f"motors {st.hz:.1f} Hz, lost ticks {st.comm_errors} of {st.ticks}, retries {st.bus_retries} | "
          f"our loop late: typically {np.median(late):.1f} ms, worst {late.max():.1f} | IMU lost {imu.lost}"
        )
  print("torque off.", "OK" if ok else "NOT finished")
  return ok, path
