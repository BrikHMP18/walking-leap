"""The scripted four-digit crawl: a planner of servo targets, one cycle at a time.

Each cycle moves the hand by a command (forward, left, turn) in its heading frame: the
palm moves over the three planted fingers while the thumb slides along (or, going
ahead with ``thumb_drag=False``, steps), then ring, middle and index each step. The
centre of mass stays inside every support, FAST_MARGIN from its edges. Keyframes come
from differential inverse kinematics on the hand's model; the servo targets add the
servos' static sag. Played open loop at 50 Hz they make the reference clip of
``Walking-Leap-Motion-Gait-Flat-v0`` (``clips.py`` writes it); scripts/scripted_gaits.py
steers the same planner from the keyboard, scripts/gait_clips.py tests it on random
simulated hands.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import lsq_linear, minimize

from walking_leap import robot

DIGITS = ("if", "mf", "rf", "th")
G = 9.81
JOINT_MARGIN = 0.05  # rad, keyframes stay this far inside the joint limits
KEYFRAME_S = 0.02  # s between keyframes (50 Hz, as the real clips)
STAND_S = 1.0  # s a "stand" command holds the targets
FIRST_DRAG_S = 1.5  # s, the first drag out of the folded-thumb stance


# One crawl cycle per command: (forward m, left m, turn rad), in the heading frame.
# Back and right are shorter: with 30 mm back 10 of 45 random simulated hands fell,
# with 15 mm right 16 of 45 (the thumb sits on the left, the ring alone holds the
# right side); at 20 and 8 mm none did (scripts/gait_clips.py).
GAIT_COMMANDS = {
  "forward": (0.030, 0.0, 0.0),
  "back": (-0.020, 0.0, 0.0),
  "left": (0.0, 0.015, 0.0),
  "right": (0.0, -0.008, 0.0),
  "turn_left": (0.0, 0.0, np.radians(20.0)),
  "turn_right": (0.0, 0.0, -np.radians(20.0)),
  "stand": (0.0, 0.0, 0.0),
}

FRONT = ("if", "mf", "rf")
FAST_MARGIN = 0.004  # m, the CoM's distance from every support edge
THUMB_LEAN = 0.003  # m, CoM behind the front row while the thumb steps
THUMB_LIFT = 0.008  # m
SLIDE_SPEED = 0.1  # m/s, fastest thumb slide
COM_SPEED = 0.08  # m/s, fastest CoM move with all four down
# The thumb steps only going ahead, turning at most this per cycle.
STEP_TURN_MAX = np.radians(3.0)
# Smooth keyframes: each IK solve also stays near the previous keyframe (weight per
# rad). A planted tip is its mesh's lowest vertex, which switches vertex as the tip
# rolls; without this term the pose jumped up to 69 rad/s between keyframes (the real
# hand's loop refuses targets faster than 6 rad/s); with it, 5.1 at most over all
# commands. 0 keeps the targets of gait.npz (Gait-v0/v1).
SMOOTH_IK = 1.0


def lowest_point(m: mujoco.MjModel, d: mujoco.MjData, digit: str) -> np.ndarray:
  """World position of the lowest vertex of a digit's tip mesh."""
  g = m.geom(f"{digit}_tip").id
  mesh = m.geom_dataid[g]
  start, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
  verts = (
    d.geom_xpos[g] + m.mesh_vert[start : start + n] @ d.geom_xmat[g].reshape(3, 3).T
  )
  return verts[verts[:, 2].argmin()]


def settled_stance(m: mujoco.MjModel) -> mujoco.MjData:
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, m.key("init_state").id)
  d.ctrl[:] = d.qpos[7:]
  for _ in range(round(1.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
  return d


def quat_error(target: np.ndarray, quat: np.ndarray) -> np.ndarray:
  """Rotation vector (rad, world frame) taking ``quat`` to ``target``."""
  inv, err = np.zeros(4), np.zeros(4)
  mujoco.mju_negQuat(inv, quat)
  mujoco.mju_mulQuat(err, target, inv)
  if err[0] < 0:
    err = -err
  vec = np.zeros(3)
  mujoco.mju_quat2Vel(vec, err, 1.0)
  return vec


def yaw_quat(quat: np.ndarray, yaw: float) -> np.ndarray:
  """``quat`` turned by ``yaw`` (rad) about the world's vertical."""
  turn, out = np.zeros(4), np.zeros(4)
  mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), yaw)
  mujoco.mju_mulQuat(out, turn, quat)
  return out


def inside(triangles, nominal, margin):
  """The point closest to ``nominal`` (xy) inside every triangle, ``margin`` from
  their edges."""

  rows, rhs = [], []
  for tri in triangles:
    for i in range(3):
      a, b, c = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]
      normal = np.array([-(b - a)[1], (b - a)[0]])
      normal /= np.linalg.norm(normal)
      if (c - a) @ normal < 0:
        normal = -normal
      rows.append(normal)
      rhs.append(normal @ a + margin)
  a_ub, b_ub = np.array(rows), np.array(rhs)
  fit = minimize(
    lambda x: float((x - nominal) @ (x - nominal)),
    nominal,
    jac=lambda x: 2 * (x - nominal),
    constraints=[
      {"type": "ineq", "fun": lambda x: a_ub @ x - b_ub, "jac": lambda x: a_ub}
    ],
    method="SLSQP",
  )
  return fit.x


class TipIk:
  """Inverse kinematics and statics for the hand standing on its fingertips (the
  lowest point of each tip mesh)."""

  def __init__(self, m: mujoco.MjModel, smooth: float = 0.0) -> None:
    self.m = m
    self.d = mujoco.MjData(m)
    self.smooth = smooth  # see SMOOTH_IK
    self.palm = m.body("palm").id
    self.weight = float(m.body_subtreemass[self.palm]) * G
    self.floor = m.geom("floor").id
    self.tip_body = {k: m.body(f"{k}_ds").id for k in DIGITS}
    self.tip_geom = {k: m.geom(f"{k}_tip").id for k in DIGITS}
    self.others = [g for g in range(m.ngeom) if m.geom_contype[g] and g != self.floor]
    self.jadr = np.array([m.jnt_qposadr[j] for j in range(1, m.njnt)])
    self.dadr = np.array([m.jnt_dofadr[j] for j in range(1, m.njnt)])
    self.lo = m.jnt_range[1:, 0] + JOINT_MARGIN
    self.hi = m.jnt_range[1:, 1] - JOINT_MARGIN
    self.stiffness = robot.MEASURED_STIFFNESS

  def qpos(self, x: np.ndarray, quat_ref: np.ndarray) -> np.ndarray:
    """x = [palm position (3), rotation vector from quat_ref (3), 16 joints]."""
    q = np.zeros(self.m.nq)
    q[:3] = x[:3]
    dq, quat = np.zeros(4), np.zeros(4)
    angle = float(np.linalg.norm(x[3:6]))
    axis = x[3:6] / angle if angle > 1e-9 else np.array([1.0, 0.0, 0.0])
    mujoco.mju_axisAngle2Quat(dq, axis, angle)
    mujoco.mju_mulQuat(quat, quat_ref, dq)
    q[3:7] = quat
    q[self.jadr] = x[6:]
    return q

  def kin(self, q: np.ndarray) -> mujoco.MjData:
    """Kinematics and centres of mass of pose ``q`` (scratch data)."""
    self.d.qpos[:] = q
    mujoco.mj_kinematics(self.m, self.d)
    mujoco.mj_comPos(self.m, self.d)
    return self.d

  def diff_ik(self, q, quat_ref, planted, swing, com_xy, com_z, iters=12):
    """Keyframe by differential inverse kinematics from ``q`` (the previous one):
    ``planted`` tips stay, the ``swing`` tip (digit, point) follows its path, the CoM
    goes to (com_xy, com_z), the palm keeps ``quat_ref``. Smooth and fast; no torque
    or clearance terms (the four-digit stance does not need them). ``self.smooth``
    weighs staying near ``q`` (see SMOOTH_IK)."""
    m, d = self.m, self.d
    q_prev, q = q.copy(), q.copy()
    keep = np.zeros((len(self.dadr), m.nv))
    keep[np.arange(len(self.dadr)), self.dadr] = 1.0
    tasks = [(k, p, 30.0) for k, p in planted.items()]
    if swing:
      tasks.append((swing[0], swing[1], 10.0))
    target_com = np.r_[com_xy, com_z]
    for _ in range(iters):
      d.qpos[:] = q
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      rows, errs = [], []
      for k, p, w in tasks:
        point = lowest_point(m, d, k)
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, point, self.tip_body[k])
        rows.append(w * jacp)
        errs.append(w * (p - point))
      jcom = np.zeros((3, m.nv))
      mujoco.mj_jacSubtreeCom(m, d, jcom, self.palm)
      rows.append(20.0 * jcom)
      errs.append(20.0 * (target_com - d.subtree_com[self.palm]))
      jrot = np.zeros((3, m.nv))
      mujoco.mj_jacBody(m, d, None, jrot, self.palm)
      rows.append(0.3 * jrot)
      errs.append(0.3 * quat_error(quat_ref, d.xquat[self.palm]))
      if self.smooth:
        rows.append(self.smooth * keep)
        errs.append(self.smooth * (q_prev[self.jadr] - q[self.jadr]))
      a, b = np.vstack(rows), np.concatenate(errs)
      step = np.linalg.solve(a.T @ a + 1e-4 * np.eye(m.nv), a.T @ b)
      mujoco.mj_integratePos(m, q, step, 1.0)
      q[self.jadr] = np.clip(q[self.jadr], self.lo, self.hi)
    return q

  def loads(self, com_xy, points: dict) -> dict:
    """Vertical load (N) on each support for the CoM at ``com_xy`` (statics)."""
    keys = list(points)
    p = np.array([points[k][:2] for k in keys])
    if len(keys) == 1:
      return {keys[0]: self.weight}
    if len(keys) == 2:
      seg = p[1] - p[0]
      s = float(np.clip((com_xy - p[0]) @ seg / (seg @ seg), 0.0, 1.0))
      return {keys[0]: (1 - s) * self.weight, keys[1]: s * self.weight}
    # Three or more supports (possibly in a line): non-negative shares that put the
    # resultant at the CoM, as even as possible.
    n = len(keys)
    a = np.vstack([1e3 * np.vstack([p.T, np.ones(n)]), np.eye(n)])
    b = np.r_[1e3 * np.r_[com_xy, 1.0], np.full(n, 1.0 / n)]
    lam = lsq_linear(a, b, bounds=(0.0, 1.0)).x
    return {k: self.weight * v / lam.sum() for k, v in zip(keys, lam, strict=True)}

  def targets(self, q: np.ndarray, loads: dict) -> np.ndarray:
    """Servo targets for pose ``q``: the joint angles plus the servos' static sag."""
    m, d = self.m, self.d
    d.qpos[:] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    gen = d.qfrc_bias.copy()
    for k, f in loads.items():
      jacp = np.zeros((3, m.nv))
      mujoco.mj_jac(m, d, jacp, None, lowest_point(m, d, k), self.tip_body[k])
      gen -= jacp.T @ np.array([0.0, 0.0, f])
    return q[self.jadr] + gen[self.dadr] / self.stiffness


class Crawler:
  """The fast crawl, one cycle at a time, steerable: each cycle moves the body by a
  command (forward, left in m; turn in rad) given in the palm's heading frame.

  Going ahead the thumb steps like the fingers: the middle finger walks
  ``mf_ahead`` in front of index and ring, so the three make a triangle; all four
  down, the CoM moves into it (``thumb_margin`` from its edges), which leaves the
  thumb unloaded; the thumb lifts and steps; then ring, middle and index step.
  Backwards, sideways or turning (or with ``thumb_drag``) the thumb slides along
  instead, as in the real crawl: stepping, the hand stands on three fingertips with
  the CoM about 1 cm from an edge, and turning it fell on 5 of 6 commands."""

  ORDERS = (("rf", "mf", "if"), ("if", "mf", "rf"))

  def __init__(
    self,
    m,
    swing_s=0.25,
    drag_s=0.3,
    shift_s=0.5,
    thumb_s=0.2,
    lift=0.010,
    reach=0.041,
    thumb_open=0.0,
    thumb_back=0.07,
    mf_ahead=0.04,
    thumb_margin=0.01,
    thumb_lean=THUMB_LEAN,
    thumb_lift=THUMB_LIFT,
    thumb_drag=False,
    smooth=0.0,
  ):
    self.m, self.swing_s, self.drag_s, self.lift = m, swing_s, drag_s, lift
    self.shift_s, self.thumb_s, self.thumb_drag = shift_s, thumb_s, thumb_drag
    self.thumb_lean, self.thumb_lift = thumb_lean, thumb_lift
    self.thumb_margin = thumb_margin
    self.ik = TipIk(m, smooth)
    self.palm = self.ik.palm
    stance = settled_stance(m)
    self.start = stance.qpos.copy()
    d = self.ik.kin(self.start)
    self.com_z = float(d.subtree_com[self.palm][2])
    self.yaw0 = self.heading(d)
    self.quat0 = self.start[3:7].copy()
    rot = self.rot(-self.yaw0)
    tips = {k: lowest_point(m, d, k)[:2] for k in DIGITS}
    # Thumb stepping: the stance's layout around the index-ring midpoint, the middle
    # finger ``mf_ahead`` further forward so that index, middle and ring make a
    # triangle the CoM can stand in while the thumb is in the air.
    front = 0.5 * (tips["if"] + tips["rf"])
    self.offset = {k: rot @ (tips[k] - front) for k in DIGITS}
    self.offset["mf"] = self.offset["mf"] + np.array([mf_ahead, 0.0])
    self.mf_ahead = mf_ahead
    # Thumb dragging: the stance's layout, the front row ``reach`` ahead of the CoM,
    # anchored on the tips' centroid (the CoM's place in it: ``com_offset``).
    com = d.subtree_com[self.palm][:2]
    self.offset_com = {k: rot @ (tips[k] - com) for k in DIGITS}
    shift = reach - np.mean([self.offset_com[k][0] for k in FRONT])
    for k in FRONT:
      self.offset_com[k] = self.offset_com[k] + np.array([shift, 0.0])
    centre = np.mean(list(self.offset_com.values()), axis=0)
    self.offset_com = {k: v - centre for k, v in self.offset_com.items()}
    self.com_offset = -centre
    # The thumb moved ``thumb_open`` further out to its side (left). Tried (drag):
    # 25 and 45 mm made it fall to the ring side on 3 of 6 commands (the ring-thumb
    # edge comes closer to the CoM), -15 mm on going back; the stance's spot on none.
    for offset in (self.offset, self.offset_com):
      offset["th"] = offset["th"] + np.array([0.0, thumb_open])
    # Stepping, the thumb lands ``thumb_back`` closer to the fingers than in the
    # stance: with the CoM at the front row it must still reach the floor.
    self.offset["th"] = self.offset["th"] + np.array([thumb_back, 0.0])
    self.count = 0

  @staticmethod
  def rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])

  def heading(self, d) -> float:
    x = d.xmat[self.palm].reshape(3, 3)[:2, 0]
    return float(np.arctan2(x[1], x[0]))

  @staticmethod
  def slow(duration, thumb0, thumb1, com0, com1):
    """``duration`` (s), longer when the thumb slides or the CoM moves far (a change
    of layout): at most SLIDE_SPEED and COM_SPEED."""
    thumb = float(np.linalg.norm(np.asarray(thumb1)[:2] - np.asarray(thumb0)[:2]))
    com = float(np.linalg.norm(com1 - com0))
    return max(duration, thumb / SLIDE_SPEED, com / COM_SPEED)

  def targets_hold(self, q):
    d = self.ik.kin(q)
    planted = {k: lowest_point(self.m, d, k) for k in DIGITS}
    return self.ik.targets(q, self.ik.loads(d.subtree_com[self.palm][:2], planted))

  def mode(self, command) -> str:
    """'step' (the thumb steps) going ahead, else 'drag'."""
    forward, left, turn = command
    if not self.thumb_drag and forward > 0 and not left and abs(turn) <= STEP_TURN_MAX:
      return "step"
    return "drag"

  def cycle(self, q, command, drag_s=None, mode=None):
    """Servo targets (every KEYFRAME_S) for one cycle from pose ``q`` (measured or
    planned) and the planned pose it ends in. ``drag_s`` overrides the first body
    move's duration (longer from standing still)."""
    m, ik = self.m, self.ik
    d = ik.kin(q)
    planted = {k: lowest_point(m, d, k).copy() for k in DIGITS}
    for p in planted.values():
      p[2] = 0.0
    com = d.subtree_com[self.palm][:2].copy()
    yaw = self.heading(d)
    forward, left, turn = command
    move = self.rot(yaw) @ np.array([forward, left])
    yaw_new = yaw + turn
    step = (mode or self.mode(command)) == "step"
    if not step:  # the real crawl's layout, around the tips' centroid
      base = np.mean([planted[k][:2] for k in DIGITS], axis=0) + move
      offset = self.offset_com
    else:  # around the index-ring midpoint, the middle finger ahead
      base = 0.5 * (planted["if"][:2] + planted["rf"][:2]) + move
      offset = self.offset
    goal_tip = {k: base + self.rot(yaw_new) @ offset[k] for k in DIGITS}
    order = self.ORDERS[self.count % 2]
    if left:  # sideways: lift the digit on the side the CoM moves away from first
      order = self.ORDERS[0] if left > 0 else self.ORDERS[1]
    self.count += 1
    plan, keys = [q.copy()], []

    def tri(without, where):
      return [where[k][:2] for k in DIGITS if k != without]

    def phase(duration, digit, c0, c1, yaw0, yaw1, height, to=None):
      """``digit`` (or None: all stay) moves to ``to`` (default: its goal) on an arc
      ``height`` high (0: it slides), the CoM from c0 to c1, the heading from yaw0
      to yaw1."""
      start = planted[digit].copy() if digit else None
      end = goal_tip[digit] if to is None and digit else to
      n = max(1, round(duration / KEYFRAME_S))
      for i in range(1, n + 1):
        u = i / n
        s = 0.5 - 0.5 * np.cos(np.pi * u)
        c = c0 + s * (c1 - c0)
        stay = {k: planted[k] for k in DIGITS if k != digit}
        swing, support = None, stay
        if digit:
          tip = start + np.r_[s * (end - start[:2]), 0.0]
          tip[2] = height * np.sin(np.pi * u) - (0.002 * u if height else 0.0)
          swing = (digit, tip)
          if not height:
            support = {**stay, digit: tip}
        quat = yaw_quat(self.quat0, yaw0 + s * (yaw1 - yaw0) - self.yaw0)
        q_new = ik.diff_ik(plan[-1], quat, stay, swing, c, self.com_z)
        plan.append(q_new)
        keys.append(ik.targets(q_new, ik.loads(c, support)))
      if digit:
        planted[digit] = np.r_[end, 0.0]

    if not step:  # the body moves and turns, the thumb slides along
      after = dict(planted)
      after["th"] = np.r_[goal_tip["th"], 0.0]
      nominal = base + self.rot(yaw_new) @ self.com_offset
      goal = inside([tri(order[0], after)], nominal, FAST_MARGIN)
      duration = self.slow(drag_s or self.drag_s, planted["th"], after["th"], com, goal)
      phase(duration, "th", com, goal, yaw, yaw_new, 0.0)
    else:  # the CoM into the index-middle-ring triangle, then the thumb steps
      head = self.rot(yaw) @ np.array([1.0, 0.0])
      mid = 0.5 * (planted["if"][:2] + planted["rf"][:2])
      if (planted["mf"][:2] - mid) @ head < 0.5 * self.mf_ahead:  # from the stance
        ahead = mid + self.rot(yaw) @ offset["mf"]
        goal = inside([tri("mf", planted)], com, FAST_MARGIN)
        phase(self.swing_s, "mf", com, goal, yaw, yaw, self.lift, to=ahead)
        com = goal
      ready = mid + self.rot(yaw) @ offset["th"]
      after = dict(planted)
      after["th"] = np.r_[goal_tip["th"], 0.0]
      nominal = np.mean([planted[k][:2] for k in FRONT], axis=0)
      goal = inside(
        [tri("th", planted), tri(order[0], after)], nominal, self.thumb_margin
      )
      if np.linalg.norm(planted["th"][:2] - ready) > 0.005:  # changing layout: slide
        duration = self.slow(drag_s or self.shift_s, planted["th"], ready, com, goal)
        phase(duration, "th", com, goal, yaw, yaw, 0.0, to=ready)
      else:
        phase(drag_s or self.shift_s, None, com, goal, yaw, yaw, 0.0)
      phase(self.thumb_s, "th", goal, goal, yaw, yaw_new, self.thumb_lift)
    com = goal
    for j, digit in enumerate(order):
      after = dict(planted)
      after[digit] = np.r_[goal_tip[digit], 0.0]
      regions = [tri(digit, planted)]
      if j + 1 < len(order):
        regions.append(tri(order[j + 1], after))
      goal = inside(regions, com, FAST_MARGIN)
      phase(self.swing_s, digit, com, goal, yaw_new, yaw_new, self.lift)
      com = goal
    return np.array(keys), plan[-1]


def plan(crawler: Crawler, commands: list[str]) -> tuple[np.ndarray, np.ndarray]:
  """Servo targets (every KEYFRAME_S) from the folded-thumb stance: a first cycle in
  place that leaves the stance, then one cycle per command ("stand" holds still).
  Also, for each target, the command it serves as a planned velocity: forward and
  left (m/s, heading frame) and turn (rad/s), the cycle's step over its duration."""
  q = crawler.start.copy()
  keys = [crawler.targets_hold(q)]
  new, q = crawler.cycle(q, np.zeros(3), drag_s=FIRST_DRAG_S)
  keys.extend(new)
  velocity = [np.zeros(3)] * len(keys)
  for name in commands:
    if name == "stand":
      hold = round(STAND_S / KEYFRAME_S)
      keys.extend([keys[-1]] * hold)
      velocity.extend([np.zeros(3)] * hold)
      continue
    new, q = crawler.cycle(q, np.array(GAIT_COMMANDS[name]))
    keys.extend(new)
    velocity.extend(
      [np.array(GAIT_COMMANDS[name]) / (len(new) * KEYFRAME_S)] * len(new)
    )
  return np.array(keys), np.array(velocity)


def training_commands(cycles: int = 150, seed: int = 0) -> list[str]:
  """A random sequence of commands, each held 1 to 3 cycles, as if someone pressed
  keys: the reference clip of the gait task."""
  rng = np.random.default_rng(seed)
  names = list(GAIT_COMMANDS)
  commands: list[str] = []
  while len(commands) < cycles:
    commands += [names[int(rng.integers(len(names)))]] * int(rng.integers(1, 4))
  return commands[:cycles]


def training_targets(
  m: mujoco.MjModel, smooth: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
  """The gait task's servo targets (the thumb slides along), planned on ``m``, and
  each target's planned velocity (see ``plan``). ``smooth``: see SMOOTH_IK (0 for
  gait.npz, SMOOTH_IK for gait_smooth.npz)."""
  return plan(Crawler(m, thumb_drag=True, smooth=smooth), training_commands())
