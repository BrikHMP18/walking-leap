"""Walking Leap control: one loop for scripts and policies on the real hand, with guards and full logs.

Joint order everywhere here: the repo's (JOINT_NAMES below). Only `Hand` knows the motors' order.
Angles in rad (0 = flat hand, + closes to the palm), gravity and gyro in the palm's axes (x to the
fingertips, y to the thumb side, z up), as in the simulation.

The motors are driven through `Hand`, the only class here that talks to them. It uses the repo's
LEAP driver, `leap_hand/` (its rig file `leap_hand/leap_right.yaml`): current-based position mode (5)
with a 450 mA limit before torque, the joint limits, a 0.5 s watchdog, torque off on every way out.
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
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/imu")
# The driver's settings for this hand (its limits, the 450 mA, the watchdog); the gains are set below.
# The current limit (the rig's, checked at connect) and the guards: a run stops (hold, then torque
# off) past MAX_TILT of tilt, or a motor over SAT_MA for SAT_S, or a clip's "down" MAX_OFF from the
# sim's for OFF_S. Loosened on 2026-11-27 (350 mA, 40 deg, 330 mA for 1.5 s, 15 deg for 0.06 s) so a
# policy has time to react to a push or a slip before the loop gives up.
CURRENT_LIMIT_MA = 450
SAT_MA = CURRENT_LIMIT_MA - 20
SAT_S = 3.0
MAX_TILT = 70.0
MAX_OFF = 25.0
OFF_S = 0.3
RIG = os.environ.get("WL_RIG", HERE + "/leap_hand/leap_right.yaml")
STOP = HERE + "/STOP"
# Where clips are looked for, in order: the panel's moves and emotes, then the ones the real
# hand has played (WL_CLIPS: one more folder, looked in first).
CLIP_DIRS = [HERE + "/clips", REPO + "/docs/data/real_clips"]
if os.environ.get("WL_CLIPS"):
  CLIP_DIRS.insert(0, os.environ["WL_CLIPS"])
CLIPS = CLIP_DIRS[-1]
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
  """The driver's LeapHand (or its fake) in the repo's joint order. 450 mA, its limits, its watchdog.

  What the loop uses: connect and disconnect (`with`), read() -> state (t, hw_error), q, dq, current in
  mA and the target sent, all in the repo's order; target(q); hold(); temps(); lo and hi; h.stats.
  """

  def __init__(self, gains=(1200, 0, 600), fake=False, q0=None, probe=False):
    from leap_hand.config import load_rig

    cfg = dataclasses.replace(load_rig(RIG).hand, kp=gains[0], ki=gains[1], kd=gains[2])
    assert cfg.current_limit_ma == CURRENT_LIMIT_MA, f"the current limit is {CURRENT_LIMIT_MA} mA"
    self.cfg, self.gains, self.fake = cfg, gains, fake
    if fake:
      from leap_hand.fake import FakeLeapHand

      b = None if q0 is None else np.asarray(q0)[JOINT_OF]
      self.h = FakeLeapHand(cfg, q0=b)
    else:
      from leap_hand.hand import LeapHand

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
    from leap_hand.third_party.leap_hand_utils import dynamixel_client as dxc

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
def clip_path(name):
  if name.endswith(".npz"):
    return name
  for folder in CLIP_DIRS:
    if os.path.exists(f"{folder}/{name}.npz"):
      return f"{folder}/{name}.npz"
  raise FileNotFoundError(f"no clip {name} in {CLIP_DIRS}")


def clip_names():
  """Every clip the loop can find, the first of each name."""
  seen = {}
  for folder in CLIP_DIRS:
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
      if f.endswith(".npz"):
        seen.setdefault(f[:-4], f"{folder}/{f}")
  return seen


class Clip:
  """Joint targets at 50 Hz, with what the sim's IMU read (or the plan's palm) for them.

  Optional: ``palm``, the planned palm path (frames x 3: x, y in m along the start's
  axes, heading in rad), for the panel's map.
  """

  def __init__(self, name):
    path = clip_path(name)
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
    self.palm = d["palm"].astype(float) if "palm" in d else None
    self.sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


class Steer:
  """``steer`` in the list of clips: a velocity policy, from the keyboard (Control.steer)."""

  name, start, end, sha = "steer", "stand", "stand", "-"


# ---------------------------------------------------------------------------- the loop
class Control:
  def __init__(self, hand, imu, rec, max_tilt=MAX_TILT, max_off=MAX_OFF, sat_s=SAT_S):
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
    self.on_tick = (
      None  # called after every tick with what was sent and read (the panel)
    )
    self.stop_asked = None  # a reason: lie down at the next tick (the panel's STOP)
    self.keep_policy = False  # clips back to back: the policy keeps its last action

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
      if self.off_for > OFF_S:
        raise Abort(f"{off:.0f} deg away from the sim")
    self.sat = np.where(np.abs(cur) > SAT_MA, self.sat + 1 / HZ, 0.0)
    if self.sat.max() > self.sat_s:
      raise Abort(
        f"{JOINT_NAMES[self.sat.argmax()]} at its current limit for {self.sat_s} s"
      )
    if os.path.exists(STOP):
      os.remove(STOP)
      raise SoftStop("STOP file")
    if self.stop_asked:
      why, self.stop_asked = self.stop_asked, None
      raise SoftStop(why)
    return s, q, dq, cur, g

  def step(self, target, pred=None, clip=-1, frame=-1, policy_q=None, clip_q=None):
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
        target if clip_q is None else clip_q,
        nan3 if pred is None else pred,
        nan16 if policy_q is None else policy_q,
        late,
      ]
    )
    self.next += 1 / HZ
    time.sleep(max(0.0, self.next - time.monotonic()))
    read = self.guard(pred)
    if self.on_tick is not None:
      s, q, dq, cur, g = read
      self.on_tick(
        dict(sent=self.q, clip_q=target if clip_q is None else clip_q, pred=pred,
             frame=frame, policy_q=policy_q, q=q, cur=cur, g=g, late=late)
      )  # fmt: skip
    return read

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
      if (
        self.policy is not None
        and self.policy.clip is not None
        and self.policy.covers(clip.name)
      )
      else None
    )
    if policy is not None:
      policy.reset(clip, keep_last=self.keep_policy)
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
      clip_q = clip.q[i] + (clip.q[j] - clip.q[i]) * (x - i)
      self.step(
        tgt, clip.grav[i] if speed == 1.0 else None, self.clip_no, i, pq, clip_q
      )
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


  def steer(self, keys, stand):
    """A velocity policy walks as the keys say until they quit, then the hand stands
    still for a second and eases back to ``stand`` (the next clip's first targets). In
    the shadow (not live) the stance is held while the policy is computed and logged."""
    if self.where != "stand":
      raise Abort(f"steering starts standing, the hand is {self.where}")
    policy, hold = self.policy, self.q.copy()
    if policy is None or policy.clip is not None:
      raise SystemExit("steer needs a velocity policy (--policy)")
    policy.reset()
    self.clip_no += 1
    self.where, name, t0, pol = "moving", "stand", time.monotonic(), []
    self.rec.event(f"steer: {'LIVE' if self.live else 'in the shadow'}, {keys.HELP}")
    quit_at = None
    while quit_at is None or time.monotonic() - quit_at < 1.0:
      key = keys.read()
      if key == "quit" and quit_at is None:
        quit_at, key = time.monotonic(), "stand"
      if key is not None and key != "quit":
        name = "stand" if key == name else key
        self.rec.event(f"steer: {name}")
      command = (
        keys.vector() if hasattr(keys, "vector") else policy.commands.get(name, np.zeros(3))
      )
      s, q, dq, cur, qc = self.hand.read()
      pq = policy.act(command, q, dq, self.imu.gyro_mean(0.01), self.imu.latest.gravity)
      pol.append(np.abs(pq - hold))
      self.step(pq if self.live else hold, None, self.clip_no, -1, pq, hold)
    self.rec.event(
      f"steer: done in {time.monotonic() - t0:.1f} s, policy {np.mean(pol):.3f} rad from"
      f" the stance on average, worst {np.max(pol):.2f}"
    )
    self.ramp(stand, 0.3)
    self.where = "stand"


class Keys:
  """The terminal's keys, without waiting: arrows move, ',' and '.' turn, the same key
  again (or space) stands still, q ends. Nothing without a terminal."""

  HELP = "arrows move, ',' '.' turn, the same key again or space: stand, q: end"
  NAMES = {
    "\x1b[A": "forward",
    "\x1b[B": "back",
    "\x1b[D": "left",
    "\x1b[C": "right",
    ",": "turn_left",
    ".": "turn_right",
    " ": "stand",
    "q": "quit",
  }

  def __enter__(self):
    import termios
    import tty

    self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    if self.fd is not None:
      self.saved = termios.tcgetattr(self.fd)
      tty.setcbreak(self.fd)
    return self

  def __exit__(self, *e):
    if self.fd is not None:
      import termios

      termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

  def read(self):
    """The last key pressed since the previous call, by name, or None."""
    import select

    if self.fd is None:
      return None
    text = ""
    while select.select([self.fd], [], [], 0)[0]:
      text += os.read(self.fd, 32).decode(errors="ignore")
    return self.parse(text)

  @classmethod
  def parse(cls, text):
    found, i = None, 0
    while i < len(text):
      for seq, name in cls.NAMES.items():
        if text.startswith(seq, i):
          found, i = name, i + len(seq)
          break
      else:
        i += 1
    return found


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
  max_off=MAX_OFF,
  rest=0.4,
  hooks=None,
  devices=None,
  start="lying",
):
  """Play clips one after the other, from lying. Torque is off when it returns."""
  clips = [Steer() if n == "steer" else Clip(n) for n in names]
  for a, b in zip(clips, clips[1:]):
    if isinstance(a, Steer) or isinstance(b, Steer):
      continue  # steering starts where stand_up ends and eases back to what follows
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
    from leap_hand.hand import raise_on_signals
    from tm171 import TM171

    raise_on_signals()
    imu, hand = TM171(), Hand(gains)
  name = label or "_".join(dict.fromkeys(c.name for c in clips))[:60]
  keys = Keys()
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
        current_limit_ma=CURRENT_LIMIT_MA,
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
        for i, c in enumerate(clips):
          if isinstance(c, Steer):
            after = clips[i + 1].q[0] if i + 1 < len(clips) else ctl.q
            with keys:
              ctl.steer(keys, after)
          else:
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
