"""Reference animations for motion imitation, written as keyframes.

A clip is a path of the palm plus finger gestures. Every foot that is not lifted stays
planted: the four joints of its digit are solved, frame by frame, so that the foot keeps
its place on the ground. The result is kinematic, at the policy rate (50 Hz), in mjlab's
motion format. ``report`` says how close a clip comes to the joint limits, the motors'
speed and losing its balance; only physics says whether it can be done.

The real hand's scripted clips (docs/data/real_clips/, joint targets that ran open
loop on the hand) become motions too: ``replay`` plays them open loop in plain MuJoCo
on the real hand's model, and the palm follows from physics. So does ``gait``: the
scripted crawl of ``gait.py`` (one cycle per command, 150 random commands), the
reference of the gait task.

  uv run python -m walking_leap.clips          # rebuild assets/clips/*.npz and report
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import Entity
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from walking_leap import robot

CLIPS_DIR: Path = Path(__file__).parent / "assets" / "clips"
# The real hand's scripted clips (git checkout only: they live outside the package).
REAL_CLIPS_DIR: Path = Path(__file__).parents[2] / "docs" / "data" / "real_clips"
# Real clips turned into motions by ``replay``.
REAL_MOTIONS: tuple[str, ...] = ("crawl",)
FPS = 50
DIGITS = ("if", "mf", "rf", "th")

##
# Home stance: palm level at HOME_HEIGHT (m), feet on these spots of the ground (x, y in
# the palm frame, m). The thumb is folded under the palm the other way round from
# robot.INIT_JOINT_POS (its base turned ~110 deg), which leaves the palm far more room
# on planted feet: -60/+27 mm up and down instead of -18/+6, +-41 deg of yaw.
##

HOME_HEIGHT = 0.130
HOME_FEET = dict(zip(DIGITS, robot.FOLDED_FEET, strict=True))
# Where the solver starts, and the pose it stays close to when a foot leaves it a choice
# (four joints for three constraints). Joint order: mcp rot pip dip / cmc axl mcp ipl.
HOME_GUESS: dict[str, tuple[float, float, float, float]] = {
  "if": (0.56, 0.0, 0.89, 0.63),
  "mf": (0.56, 0.0, 0.89, 0.63),
  "rf": (0.56, 0.0, 0.89, 0.63),
  "th": (0.24, 1.91, 1.46, 0.31),
}


def ease(u: float | np.ndarray) -> np.ndarray:
  """Minimum-jerk step from 0 to 1."""
  u = np.clip(u, 0.0, 1.0)
  return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


##
# Writing a clip.
##

Pose = np.ndarray  # x, y, z (m), roll, pitch, yaw (rad): the palm's offset from home.


def pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0) -> Pose:
  """Palm offset from home: mm and degrees. +pitch is nose down, +yaw to the thumb."""
  return np.r_[np.array([x, y, z]) * 1e-3, np.radians([roll, pitch, yaw])]


@dataclass
class Gesture:
  """A digit leaves the ground and comes back.

  Either a tap (``height``, m: the foot goes straight up and down) or a joint pose
  (``joints(t)``, rad, t in seconds since the start) reached after ``blend`` seconds.
  """

  digit: str
  start: float
  duration: float
  height: float = 0.0
  joints: Callable[[float], np.ndarray] | None = None
  blend: float = 0.5


@dataclass
class Clip:
  """Palm path as consecutive segments, plus gestures at given times."""

  name: str
  segments: list[tuple[float, Callable[[float], Pose]]] = field(default_factory=list)
  gestures: list[Gesture] = field(default_factory=list)
  _at: Pose = field(default_factory=pose)

  @property
  def now(self) -> float:
    return sum(duration for duration, _ in self.segments)

  def to(self, duration: float, **offset: float) -> Clip:
    """Move the palm to an offset from home, starting and ending still."""
    a, b = self._at.copy(), pose(**offset)
    self.segments.append((duration, lambda u: a + (b - a) * ease(u)))
    self._at = b
    return self

  def hold(self, duration: float) -> Clip:
    a = self._at.copy()
    self.segments.append((duration, lambda u: a))
    return self

  def dip(self, duration: float, **offset: float) -> Clip:
    """Go to an offset and back in one smooth move."""
    a, b = self._at.copy(), pose(**offset)
    self.segments.append((duration, lambda u: a + (b - a) * ease(1 - abs(2 * u - 1))))
    return self

  def sway(self, duration: float, turns: float, **radius: float) -> Clip:
    """Circle: each offset swings as sin (x, roll) or cos (the rest), eased in and out."""
    a, r = self._at.copy(), pose(**radius)
    phase = np.array([0.0, 0.5, 0.5, 0.5, 0.0, 0.5]) * np.pi

    def path(u: float) -> Pose:
      envelope = ease(min(u, 1 - u) / 0.2)
      return a + envelope * r * np.sin(2 * np.pi * turns * u + phase)

    self.segments.append((duration, path))
    return self

  def palm(self, t: float) -> Pose:
    for duration, path in self.segments:
      if t <= duration:
        return path(t / duration)
      t -= duration
    return self._at


def hello() -> Clip:
  """Stand, push-ups, hip circles, look around, wave, drum the fingers, bow."""
  c = Clip("hello").hold(0.5)
  # Two push-ups and a stretch.
  c.dip(1.2, z=-45).dip(1.2, z=-45).dip(1.0, z=18).hold(0.1)
  # Hip circles, then look left and right.
  c.sway(2.6, 2, x=14, y=16, z=0, yaw=5).hold(0.1)
  c.to(0.6, yaw=20, roll=4).hold(0.15).to(0.9, yaw=-20, roll=-4).hold(0.15).to(0.6)
  # Lean back to the ring side, nose up, and wave with the index held straight out.
  c.to(0.7, x=-8, y=-22, z=-4, pitch=-7)

  def wave(t: float) -> np.ndarray:
    # Around -0.12 rad: away from the middle finger.
    swing = -0.12 + 0.42 * np.sin(2 * np.pi * 1.5 * (t - 0.7)) if t > 0.7 else 0.0
    return np.array([0.0, swing * ease((t - 0.7) / 0.3), 0.0, 0.0])

  c.gestures.append(Gesture("if", c.now, 3.4, joints=wave, blend=0.7))
  c.hold(3.4).to(0.7)
  # Drum: index, ring, middle, twice, swaying off the finger that lifts.
  for _ in range(2):
    c.to(0.4, y=-20)
    c.gestures.append(Gesture("if", c.now, 0.55, height=0.015))
    c.hold(0.55).to(0.55, y=24)
    c.gestures.append(Gesture("rf", c.now, 0.55, height=0.015))
    c.hold(0.55).to(0.4)
    c.gestures.append(Gesture("mf", c.now, 0.55, height=0.015))
    c.hold(0.55)
  # Bow and stand.
  c.to(0.8, x=-6, z=-10, pitch=15).hold(0.4).to(0.8).hold(0.5)
  return c


CLIPS: dict[str, Callable[[], Clip]] = {"hello": hello}


##
# Solving it.
##


class Kinematics:
  """The as-built hand with planted feet (plain MuJoCo, CPU)."""

  def __init__(self) -> None:
    self.model = Entity(robot.get_leap_hand_cfg(as_built=True)).spec.compile()
    self.data = mujoco.MjData(self.model)
    m = self.model
    self.palm = m.body(robot.ROOT_BODY).id
    names = {k: [n for n in robot.JOINT_NAMES if n.startswith(k)] for k in DIGITS}
    self.qadr = {k: np.array([m.joint(n).qposadr[0] for n in names[k]]) for k in DIGITS}
    self.limits = {
      k: np.array([m.jnt_range[m.joint(n).id] for n in names[k]]) for k in DIGITS
    }
    self.body = {k: m.body(f"{k}_foot").id for k in DIGITS}
    self.foot = {k: m.geom(f"{k}_foot").id for k in DIGITS}
    self.hull = {}
    for k, g in self.foot.items():
      mesh = m.geom_dataid[g]
      start, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
      self.hull[k] = m.mesh_vert[start : start + n].copy()
    self.home = {k: self._home(k) for k in DIGITS}

  def set_palm(self, offset: Pose) -> None:
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, offset[3:], "xyz")
    self.data.qpos[:3] = offset[:3] + (0.0, 0.0, HOME_HEIGHT)
    self.data.qpos[3:7] = quat

  def foot_points(self, digit: str) -> tuple[np.ndarray, np.ndarray]:
    """World positions of the foot's centre and of its lowest point."""
    d, g = self.data, self.foot[digit]
    hull = d.geom_xpos[g] + self.hull[digit] @ d.geom_xmat[g].reshape(3, 3).T
    return d.xpos[self.body[digit]].copy(), hull[hull[:, 2].argmin()]

  def plant(self, digit: str, height: float, start: np.ndarray) -> np.ndarray:
    """Joints that put the foot over its spot, lowest point at ``height`` (m).

    Uses the palm pose set in ``data``. Stays 0.05 rad inside the joint limits, and as
    close to the home pose as the spot allows.
    """
    lo, hi = self.limits[digit][:, 0] + 0.05, self.limits[digit][:, 1] - 0.05
    spot, rest = np.array(HOME_FEET[digit]), np.array(HOME_GUESS[digit])

    def residual(q: np.ndarray) -> np.ndarray:
      self.data.qpos[self.qadr[digit]] = q
      mujoco.mj_kinematics(self.model, self.data)
      center, low = self.foot_points(digit)
      return np.r_[center[:2] - spot, low[2] - height, 0.002 * (q - rest) / (hi - lo)]

    start = np.clip(start, lo + 1e-5, hi - 1e-5)
    fit = least_squares(
      residual, start, bounds=(lo, hi), max_nfev=40, xtol=1e-9, ftol=1e-11
    )
    self.data.qpos[self.qadr[digit]] = fit.x
    self.error = float(np.linalg.norm(fit.fun[:3]))
    return fit.x

  def _home(self, digit: str) -> np.ndarray:
    self.set_palm(pose())
    return self.plant(digit, 0.0, np.array(HOME_GUESS[digit]))


@dataclass
class Motion:
  """A solved clip. ``qpos``: palm position, quaternion, 16 joints; one row a frame."""

  name: str
  qpos: np.ndarray
  lifted: np.ndarray  # frames x DIGITS, bool.
  ik_error: np.ndarray  # m, worst planted foot of each frame.


def solve(clip: Clip, kin: Kinematics | None = None) -> Motion:
  kin = kin or Kinematics()
  times = np.arange(0.0, clip.now, 1.0 / FPS)
  qpos = np.zeros((len(times), kin.model.nq))
  lifted = np.zeros((len(times), len(DIGITS)), dtype=bool)
  errors = np.zeros(len(times))
  last = {k: v.copy() for k, v in kin.home.items()}
  for i, t in enumerate(times):
    kin.set_palm(clip.palm(t))
    for j, digit in enumerate(DIGITS):
      active = [
        g
        for g in clip.gestures
        if g.digit == digit and g.start < t < g.start + g.duration
      ]
      height = sum(
        g.height * np.sin(np.pi * ease((t - g.start) / g.duration)) ** 2 for g in active
      )
      q = kin.plant(digit, height, last[digit])
      last[digit] = q
      errors[i] = max(errors[i], kin.error)
      for g in active:
        lifted[i, j] = True
        if g.joints is not None:
          up = min(t - g.start, g.start + g.duration - t) / g.blend
          # The knuckle leads on the way up and trails on the way down.
          s = np.array([ease(up / 0.8)] + [ease((up - 0.2) / 0.8)] * 3)
          q = q + s * (g.joints(t - g.start) - q)
          kin.data.qpos[kin.qadr[digit]] = q
    qpos[i] = kin.data.qpos
  return Motion(clip.name, qpos, lifted, errors)


class RealHand:
  """The real hand as it runs now, on a floor (plain MuJoCo, CPU): IMU, stock
  fingertips, LEAP's thumb limits, 350 mA, measured stiffness, torsional friction,
  the walking tasks' physics options. The model the real clips are replayed on."""

  def __init__(self) -> None:
    from walking_leap.env_cfg import walking_leap_flat_env_cfg

    cfg = robot.get_leap_hand_cfg(
      as_built=True,
      folded_thumb=True,
      feet=False,
      effort_limit=robot.API_EFFORT_LIMIT,
      stiffness=robot.MEASURED_STIFFNESS,
      torsion=robot.TORSION,
    )
    spec = Entity(cfg).spec
    spec.worldbody.add_geom(
      name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01)
    )
    self.model = spec.compile()
    walking_leap_flat_env_cfg().sim.mujoco.apply(self.model)
    self.data = mujoco.MjData(self.model)


def replay(
  name: str, hand: RealHand, settle_s: float = 0.5, q: np.ndarray | None = None
) -> Motion:
  """A real clip's joint targets, played open loop from the folded-thumb stance.

  One row per target (50 Hz): the state when that target is sent. The stance is
  settled first, with the clip's first targets. ``lifted`` and ``ik_error`` do not
  apply (no planted-feet solve) and are left empty. ``q`` (frames x 16, rad) plays
  these targets instead of the named clip's.
  """
  if q is None:
    q = np.load(REAL_CLIPS_DIR / f"{name}.npz")["q"]
  m, d = hand.model, hand.data
  mujoco.mj_resetDataKeyframe(m, d, m.key("init_state").id)
  d.qpos[7:] = q[0]
  d.ctrl[:] = q[0]
  mujoco.mj_forward(m, d)
  for _ in range(round(settle_s / m.opt.timestep)):
    mujoco.mj_step(m, d)
  steps = round(1.0 / (FPS * m.opt.timestep))
  qpos = np.zeros((len(q), m.nq))
  for i, target in enumerate(q):
    qpos[i] = d.qpos
    d.ctrl[:] = target
    for _ in range(steps):
      mujoco.mj_step(m, d)
  lifted = np.zeros((len(q), len(DIGITS)), dtype=bool)
  return Motion(name, qpos, lifted, np.zeros(len(q)))


def save(
  motion: Motion,
  kin: Kinematics | RealHand,
  path: Path,
  extra: dict[str, np.ndarray] | None = None,
) -> None:
  """Write mjlab's motion format: joints, and every body's pose and velocity, plus
  ``extra`` arrays (one row a frame; mjlab's loader ignores them)."""
  m, d = kin.model, kin.data
  n = len(motion.qpos)
  qvel = np.zeros((n, m.nv))
  for i in range(n):
    a, b = max(i - 1, 0), min(i + 1, n - 1)
    mujoco.mj_differentiatePos(
      m, qvel[i], (b - a) / FPS, motion.qpos[a], motion.qpos[b]
    )
  nb = m.nbody - 1  # All bodies but the world, in mjlab's order.
  pos, quat = np.zeros((n, nb, 3)), np.zeros((n, nb, 4))
  lin, ang = np.zeros((n, nb, 3)), np.zeros((n, nb, 3))
  velocity = np.zeros(6)
  for i in range(n):
    d.qpos[:], d.qvel[:] = motion.qpos[i], qvel[i]
    mujoco.mj_forward(m, d)
    pos[i], quat[i] = d.xpos[1:], d.xquat[1:]
    for b in range(nb):
      mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b + 1, velocity, 0)
      ang[i, b], lin[i, b] = velocity[:3], velocity[3:]
  path.parent.mkdir(parents=True, exist_ok=True)
  arrays = dict(
    joint_pos=motion.qpos[:, 7:],
    joint_vel=qvel[:, 6:],
    body_pos_w=pos,
    body_quat_w=quat,
    body_lin_vel_w=lin,
    body_ang_vel_w=ang,
  )
  arrays.update(extra or {})
  arrays = {k: v.astype(np.float32) for k, v in arrays.items()}
  np.savez_compressed(path, fps=np.array([FPS]), **arrays)


def report(motion: Motion, kin: Kinematics) -> dict[str, float]:
  """How close the clip comes to what the hand can't do (kinematics only)."""
  m, d = kin.model, kin.data
  ranges = m.jnt_range[1:]
  q = motion.qpos[:, 7:]
  speed = np.abs(np.gradient(q, 1.0 / FPS, axis=0))
  balance, clearance, touching = [], [], 0
  for i in range(len(q)):
    d.qpos[:] = motion.qpos[i]
    mujoco.mj_fwdPosition(m, d)
    down = [k for j, k in enumerate(DIGITS) if not motion.lifted[i, j]]
    hull = ConvexHull(np.array([kin.foot_points(k)[1][:2] for k in down]))
    com = d.subtree_com[kin.palm][:2]
    balance.append(-(hull.equations[:, :2] @ com + hull.equations[:, 2]).max())
    clearance += [kin.foot_points(k)[1][2] for k in DIGITS if k not in down]
    touching += any(c.dist < -1e-4 for c in d.contact[: d.ncon])
  return {
    "seconds": len(q) / FPS,
    "ik_error_mm": 1e3 * float(motion.ik_error.max()),
    "limit_margin_rad": float(np.minimum(q - ranges[:, 0], ranges[:, 1] - q).min()),
    "peak_joint_speed_rad_s": float(speed.max()),
    "balance_margin_mm": 1e3 * float(min(balance)),
    "lowest_lifted_foot_mm": 1e3 * float(min(clearance, default=0.0)),
    "frames_touching_itself": float(touching),
  }


def main() -> None:
  kin = Kinematics()
  for name, write in CLIPS.items():
    motion = solve(write(), kin)
    save(motion, kin, CLIPS_DIR / f"{name}.npz")
    print(name, {k: round(v, 3) for k, v in report(motion, kin).items()})
  hand = RealHand()
  for name in REAL_MOTIONS:
    motion = replay(name, hand)
    save(motion, hand, CLIPS_DIR / f"{name}.npz")
    moved = 1e3 * (motion.qpos[-1, :2] - motion.qpos[0, :2])
    print(name, {"frames": len(motion.qpos), "moved_mm": moved.round(1).tolist()})
  write_gait(hand)
  write_gait(hand, smooth=True)
  write_gait(hand, fast_turns=True)


def write_gait(
  hand: RealHand, smooth: bool = False, fast_turns: bool = False
) -> Motion:
  """assets/clips/gait.npz: the scripted crawl's targets (``gait.training_targets``)
  replayed on the real hand's model, with each frame's planned velocity
  (``command``: forward, left m/s; turn rad/s). ``smooth``: gait_smooth.npz, planned
  with gait.SMOOTH_IK (targets under the real loop's 6 rad/s)."""
  from walking_leap import gait

  name = "gait_smooth2" if fast_turns else "gait_smooth" if smooth else "gait"
  extra = {}
  if smooth or fast_turns:  # also each frame's gait phase (``gait.plan``)
    table = gait.FAST_TURN_COMMANDS if fast_turns else None
    targets, velocity, extra["phase"] = gait.training_targets(
      hand.model, gait.SMOOTH_IK, with_phase=True, table=table
    )
  else:
    targets, velocity = gait.training_targets(hand.model)
  motion = replay(name, hand, q=targets)
  save(motion, hand, CLIPS_DIR / f"{name}.npz", extra={"command": velocity, **extra})
  moved = 1e3 * (motion.qpos[-1, :2] - motion.qpos[0, :2])
  print(name, {"frames": len(motion.qpos), "moved_mm": moved.round(1).tolist()})
  return motion


GAIT_KEYS = (  # command, axis of (forward, left, turn), sign
  ("forward", 0, 1),
  ("back", 0, -1),
  ("left", 1, 1),
  ("right", 1, -1),
  ("turn_left", 2, 1),
  ("turn_right", 2, -1),
)


def gait_command_velocities(clip: str = "gait") -> dict[str, np.ndarray]:
  """Each command's velocity as a gait student saw it: gait.npz's planned velocity
  (forward m/s, left m/s, turn rad/s) averaged over the frames of that command (its
  cycles last 1.08-1.10 s). What a keyboard sends for that command."""
  planned = np.load(CLIPS_DIR / f"{clip}.npz")["command"]
  out = {}
  for name, axis, sign in GAIT_KEYS:
    alone = np.count_nonzero(planned, axis=1) == 1
    out[name] = planned[alone & (np.sign(planned[:, axis]) == sign)].mean(0)
  return out


if __name__ == "__main__":
  main()


def gait_cycle_s() -> float:
  """The smooth crawl's cycle, s (gait_smooth.npz's phase: the median cycle, 50-57
  frames for every command): the period of a clock policy's own gait clock."""
  phase = np.load(CLIPS_DIR / "gait_smooth.npz")["phase"]
  starts = np.flatnonzero(np.diff(phase) < 0) + 1
  return float(np.median(np.diff(starts))) / FPS


def trained_clip(run: Path) -> str:
  """The clip a run trained on (its params/env.yaml's motion file), e.g. "gait_smooth":
  whose planned velocities its keys send. "gait" when the file says nothing."""
  import re

  env = Path(run) / "params" / "env.yaml"
  text = env.read_text() if env.exists() else ""
  match = re.search(r"motion_file:\s*['\"]?\S*?/(\w+)\.npz", text)
  return match.group(1) if match else "gait"


def gait_cycle_periods(clip: str = "gait_smooth") -> dict[str, float]:
  """Each command's gait cycle, s (median over the clip's cycles of that command alone):
  a clock policy's clock period for that key. In gait_smooth all are 1.02-1.14 s; in
  gait_smooth2 the larger turns take 1.83 s (left) and 1.54 s (right)."""
  data = np.load(CLIPS_DIR / f"{clip}.npz")
  phase, planned = data["phase"], data["command"]
  starts = np.flatnonzero(np.diff(phase) < 0) + 1
  out: dict[str, list[float]] = {}
  for a, b in zip(starts[:-1], starts[1:], strict=True):
    c = planned[a]
    for name, axis, sign in GAIT_KEYS:
      if np.count_nonzero(c) == 1 and np.sign(c[axis]) == sign:
        out.setdefault(name, []).append((b - a) / FPS)
  return {name: float(np.median(v)) for name, v in out.items()}
