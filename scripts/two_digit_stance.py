"""Can the real hand stand on two digits? A posture search, then a physics test.

  uv run python scripts/two_digit_stance.py                # middle finger + thumb
  uv run python scripts/two_digit_stance.py --pair rf th   # another pair
  uv run python scripts/two_digit_stance.py --view         # watch it in MuJoCo
  uv run python scripts/two_digit_stance.py --pair if mf --free --max-pitch 1.6
      # stand up on index + middle: the tips may move, the palm may tilt far

Model: the real hand as it runs now (IMU, stock fingertips, LEAP's thumb limits,
350 mA cap, stiffness 2.6 N*m/rad, torsional friction), from the folded-thumb stance.

1. Posture (kinematics, least squares over the palm pose and the 16 joints): the two
   chosen tips stay planted where they stand in the four-digit stance, the other two
   are lifted by at least LIFT, the centre of mass lies on the line between the two
   planted tips (and away from its ends), the palm stays near level, every joint
   0.05 rad inside its limits, as close to the stance as the rest allows.
2. Physics: from the settled four-digit stance the servo targets move to the posture
   over RAMP_S, are held for HOLD_S, and the run reports falls, which digits touch the
   ground, the lifted digits' clearance and the peak joint torque against the cap.
With ``--free`` the two tips may move to new spots (only their height is held), no
other part of the hand may come within CLEARANCE of the floor, and the physics test
starts in the posture instead of ramping from the four-digit stance.

Walking (work in progress, it still falls after one step):
  uv run python scripts/two_digit_stance.py --walk [--view]
index + middle as legs, the palm upright, a whole-body torque controller (``Wbc``)
and capture-point steps. TODO: see the commit that added ``--walk``.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from mjlab.entity import Entity
from scipy.optimize import least_squares, lsq_linear

from walking_leap import robot
from walking_leap.env_cfg import walking_leap_flat_env_cfg

DIGITS = ("if", "mf", "rf", "th")
NAMES = {"if": "índice", "mf": "medio", "rf": "anular", "th": "pulgar"}
LIFT = 0.010  # m, clearance of the lifted digits' tips
END_MARGIN = 0.020  # m, centre of mass at least this far from either planted tip
JOINT_MARGIN = 0.05  # rad
CLEARANCE = 0.005  # m, every other part of the hand above the floor
RAMP_S, HOLD_S = 2.0, 5.0


def build() -> mujoco.MjModel:
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
  m = spec.compile()
  walking_leap_flat_env_cfg().sim.mujoco.apply(m)
  return m


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


def solve_posture(
  m,
  stance: mujoco.MjData,
  planted: tuple[str, str],
  free: bool = False,
  max_pitch: float = 0.6,
  home_weight: float = 0.05,
  end_margin: float = END_MARGIN,
):
  """Least-squares posture; returns qpos and a summary."""
  lifted = [k for k in DIGITS if k not in planted]
  d = mujoco.MjData(m)
  d.qpos[:] = stance.qpos
  mujoco.mj_kinematics(m, d)
  spots = {k: lowest_point(m, d, k)[:2].copy() for k in planted}
  floor = m.geom("floor").id
  planted_tips = {m.geom(f"{k}_tip").id for k in planted}
  others = [
    g
    for g in range(m.ngeom)
    if m.geom_contype[g] and g != floor and g not in planted_tips
  ]

  def support():
    a, b = (lowest_point(m, d, k)[:2] for k in planted)
    if not free:
      a, b = spots[planted[0]], spots[planted[1]]
    length = float(np.linalg.norm(b - a))
    return a, (b - a) / length, length

  q_home = stance.qpos[7:].copy()
  lo = m.jnt_range[1:, 0] + JOINT_MARGIN
  hi = m.jnt_range[1:, 1] - JOINT_MARGIN
  euler0 = np.zeros(3)

  def unpack(x):
    qpos = np.zeros(m.nq)
    qpos[:3] = x[:3]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, x[3:6], "xyz")
    qpos[3:7] = quat
    qpos[7:] = x[6:]
    return qpos

  def residual(x):
    d.qpos[:] = unpack(x)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    r = []
    for k in planted:
      p = lowest_point(m, d, k)
      r += [1e3 * p[2]] if free else [1e3 * p[2], *(1e3 * (p[:2] - spots[k]))]
    for g in others:
      gap = mujoco.mj_geomDistance(m, d, floor, g, 0.05, None)
      r.append(1e3 * max(0.0, CLEARANCE - gap))
    for k in lifted:
      r.append(1e3 * max(0.0, LIFT - lowest_point(m, d, k)[2]))
    a, axis, length = support()
    com = d.subtree_com[m.body("palm").id][:2]
    rel = com - a
    along = float(rel @ axis)
    lateral = float(rel @ np.array([-axis[1], axis[0]]))
    r += [3e3 * lateral]
    r += [
      1e3 * max(0.0, end_margin - along),
      1e3 * max(0.0, end_margin - (length - along)),
    ]
    r += [0.3 * x[3], 0.3 * x[4]]  # palm roll, pitch: near level if possible
    r += list(home_weight * (x[6:] - q_home))  # stay close to the stance
    return np.array(r)

  x0 = np.r_[stance.qpos[:3], euler0, q_home]
  reach = 0.12 if free else 0.08
  lower = np.r_[x0[:2] - reach, 0.05, -0.6, -max_pitch, -np.pi, lo]
  upper = np.r_[x0[:2] + reach, 0.25, 0.6, max_pitch, np.pi, hi]
  fit = least_squares(
    residual, np.clip(x0, lower, upper), bounds=(lower, upper), max_nfev=400
  )
  qpos = unpack(fit.x)
  d.qpos[:] = qpos
  mujoco.mj_kinematics(m, d)
  mujoco.mj_comPos(m, d)
  a, axis, length = support()
  com = d.subtree_com[m.body("palm").id][:2] - a
  clearance = min(mujoco.mj_geomDistance(m, d, floor, g, 0.05, None) for g in others)
  summary = dict(
    planted_height_mm=round(max(1e3 * lowest_point(m, d, k)[2] for k in planted), 2),
    planted_moved_mm={
      NAMES[k]: round(
        1e3 * float(np.linalg.norm(lowest_point(m, d, k)[:2] - spots[k])), 1
      )
      for k in planted
    },
    support_length_mm=round(1e3 * length, 1),
    other_parts_clearance_mm=round(1e3 * clearance, 1),
    lift_mm={NAMES[k]: round(1e3 * lowest_point(m, d, k)[2], 1) for k in lifted},
    com_lateral_mm=round(1e3 * float(com @ np.array([-axis[1], axis[0]])), 1),
    com_along_fraction=round(float(com @ axis) / length, 2),
    palm_roll_pitch_deg=tuple(np.degrees(fit.x[3:5]).round(1)),
    palm_shift_mm=tuple((1e3 * (qpos[:3] - stance.qpos[:3])).round(1)),
    joint_margin_rad=round(
      float(
        np.minimum(qpos[7:] - m.jnt_range[1:, 0], m.jnt_range[1:, 1] - qpos[7:]).min()
      ),
      3,
    ),
  )
  return qpos, summary


def touching(m, d) -> set[str]:
  floor = m.geom("floor").id
  hit = set()
  for c in d.contact[: d.ncon]:
    if floor not in (c.geom1, c.geom2):
      continue
    body = m.geom_bodyid[c.geom2 if c.geom1 == floor else c.geom1]
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body) or ""
    hit.add(name[:2] if name[:2] in DIGITS else "palma/otro")
  return hit


def run_physics(m, start_qpos, target_q, ramp_s=RAMP_S, viewer=None, d=None):
  """From ``start_qpos`` (targets = its joints), ramp the targets to ``target_q``
  over ``ramp_s``, hold, report. With ``viewer``, steps its data ``d`` in real time."""
  d = mujoco.MjData(m) if d is None else d
  mujoco.mj_resetData(m, d)
  d.qpos[:], d.qvel[:] = start_qpos, 0.0
  d.ctrl[:] = start_qpos[7:]
  mujoco.mj_forward(m, d)
  start_ctrl = start_qpos[7:].copy()
  palm = m.body("palm").id
  n_ramp = max(1, round(ramp_s / m.opt.timestep))
  n_hold = round(HOLD_S / m.opt.timestep)
  counts = {k: 0 for k in [*DIGITS, "palma/otro"]}
  peak, worst_tilt, min_clear = 0.0, 0.0, {k: 1.0 for k in DIGITS}
  xy0 = None
  wall0 = time.perf_counter()
  for k in range(n_ramp + n_hold):
    s = min(1.0, k / n_ramp)
    s = 0.5 - 0.5 * np.cos(np.pi * s)
    d.ctrl[:] = (1 - s) * start_ctrl + s * target_q
    mujoco.mj_step(m, d)
    if k >= n_ramp:
      if xy0 is None:
        xy0 = d.qpos[:2].copy()
      for name in touching(m, d):
        counts[name] += 1
      peak = max(peak, float(np.abs(d.actuator_force).max()))
      tilt = np.degrees(np.arccos(np.clip(d.xmat[palm].reshape(3, 3)[2, 2], -1, 1)))
      worst_tilt = max(worst_tilt, float(tilt))
      for dg in DIGITS:
        min_clear[dg] = min(min_clear[dg], float(lowest_point(m, d, dg)[2]))
    if viewer is not None and k % 10 == 0:
      if not viewer.is_running():
        return None
      viewer.sync()
      time.sleep(max(0.0, wall0 + k * m.opt.timestep - time.perf_counter()))
  return dict(
    fell=bool(worst_tilt > 40.0),
    worst_tilt_deg=round(worst_tilt, 1),
    drift_mm=round(1e3 * float(np.linalg.norm(d.qpos[:2] - xy0)), 1),
    ground_contact={
      NAMES.get(k, k): f"{100 * v / n_hold:.0f} %" for k, v in counts.items()
    },
    min_clearance_mm={NAMES[k]: round(1e3 * v, 1) for k, v in min_clear.items()},
    peak_torque_Nm=round(peak, 3),
    cap_Nm=round(robot.API_EFFORT_LIMIT, 3),
    warnings=int(sum(w.number for w in d.warning)),
  )


##
# Walking on two digits: index and middle finger as legs, the palm standing upright
# above them like a torso, ring and thumb tucked away. On two point feet the hand
# cannot stand still (it tips over within ~0.3 s), so it walks like a biped on stilts:
# a linear inverted pendulum, each step placed at the divergent component of motion
# (capture point) predicted for touchdown, with an offset that keeps a steady gait.
# Every control tick an IK solve (stance foot fixed, swing foot on its arc, palm
# upright, CoM at its measured place and the nominal height) gives the servo targets.
##

LEGS = ("if", "mf")
TUCK = {  # ring and thumb folded clear of the floor, the palm and the legs (no contact)
  "rf_mcp": 0.69, "rf_rot": 0.32, "rf_pip": 1.34, "rf_dip": 1.91,
  "th_cmc": 1.56, "th_axl": 2.05, "th_mcp": 0.58, "th_ipl": 1.29,
}  # fmt: skip
G = 9.81


class Biped:
  """The real hand's model walking on index and middle finger."""

  def __init__(self, m: mujoco.MjModel) -> None:
    self.m = m
    self.palm = m.body("palm").id
    self.weight = float(m.body_subtreemass[self.palm]) * G
    self.leg_q = {
      k: [m.joint(f"{k}_{j}").qposadr[0] for j in ("mcp", "rot", "pip", "dip")]
      for k in LEGS
    }
    self.leg_v = {
      k: [m.joint(f"{k}_{j}").dofadr[0] for j in ("mcp", "rot", "pip", "dip")]
      for k in LEGS
    }
    self.leg_u = {
      k: [m.actuator(f"{k}_{j}").id for j in ("mcp", "rot", "pip", "dip")] for k in LEGS
    }
    self.tip_body = {k: m.body(f"{k}_ds").id for k in LEGS}
    self.legs_adr = self.leg_q["if"] + self.leg_q["mf"]
    lo = m.jnt_range[1:, 0] + JOINT_MARGIN
    hi = m.jnt_range[1:, 1] - JOINT_MARGIN
    self.lo = np.array([lo[a - 7] for a in self.legs_adr])
    self.hi = np.array([hi[a - 7] for a in self.legs_adr])
    self.tuck = {m.joint(n).qposadr[0]: v for n, v in TUCK.items()}
    self.kd = mujoco.MjData(m)  # scratch data for kinematics
    self.stiffness = robot.MEASURED_STIFFNESS

  def posture_qpos(self, base_pos, base_quat, legs) -> np.ndarray:
    q = np.zeros(self.m.nq)
    q[:3], q[3:7] = base_pos, base_quat
    q[self.legs_adr] = legs
    for adr, v in self.tuck.items():
      q[adr] = v
    return q

  def kin(self, qpos):
    d = self.kd
    d.qpos[:] = qpos
    mujoco.mj_kinematics(self.m, d)
    mujoco.mj_comPos(self.m, d)
    return d

  def upright(self) -> tuple[np.ndarray, np.ndarray]:
    """Standing posture: both tips on the floor, CoM between them, low torque."""
    m, floor = self.m, self.m.geom("floor").id
    tips = {m.geom(f"{k}_tip").id for k in LEGS}
    others = [
      g for g in range(m.ngeom) if m.geom_contype[g] and g != floor and g not in tips
    ]
    names = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in others}

    def unpack(x):
      quat = np.zeros(4)
      mujoco.mju_euler2Quat(quat, x[3:6], "xyz")
      return self.posture_qpos(x[:3], quat, x[6:])

    def residual(x):
      d = self.kin(unpack(x))
      a, b = (lowest_point(m, d, k) for k in LEGS)
      r = [3e3 * a[2], 3e3 * b[2]]
      com = d.subtree_com[self.palm]
      r += list(3e3 * (com[:2] - 0.5 * (a[:2] + b[:2])))
      for g in others:
        gap = mujoco.mj_geomDistance(m, d, floor, g, 0.08, None)
        need = 0.025 if names[g].startswith(("rf", "th")) else 0.006
        r.append(1e3 * max(0.0, need - gap))
      for k, foot in zip(LEGS, (a, b), strict=True):
        tau = self.support_torque(d, k, foot, self.weight / 2)
        r += list(30 * np.maximum(0.0, np.abs(tau) - 0.06))
      r.append(30 * max(0.0, 0.035 - float(np.linalg.norm(a[:2] - b[:2]))))
      return np.array(r)

    x0 = np.r_[
      0, 0, 0.15, 0, 0.9, 0, np.clip(np.tile([0.0, 0.0, 0.2, 0.2], 2), self.lo, self.hi)
    ]
    lower = np.r_[-0.3, -0.3, 0.03, -0.9, -1.65, -np.pi, self.lo]
    upper = np.r_[0.3, 0.3, 0.3, 0.9, 1.65, np.pi, self.hi]
    fit = least_squares(
      residual, np.clip(x0, lower, upper), bounds=(lower, upper), max_nfev=400
    )
    q = unpack(fit.x)
    d = self.kin(q)
    q[2] += 0.0005 - min(lowest_point(m, d, k)[2] for k in LEGS)
    return q, fit.x[6:]

  def support_torque(self, d, leg: str, point: np.ndarray, load: float) -> np.ndarray:
    """Joint torques (N*m) a leg's servos make to carry ``load`` (N) at ``point``."""
    jacp = np.zeros((3, self.m.nv))
    mujoco.mj_jac(self.m, d, jacp, None, point, self.tip_body[leg])
    return -jacp[2, self.leg_v[leg]] * load

  def ik(
    self, guess, base_quat0, stance, swing, stance_xy, swing_target, com_xy, com_z
  ):
    """Leg joints (8) for the targets; the base pose is a free variable."""
    m = self.m

    def unpack(x):
      dq = np.zeros(4)
      angle = float(np.linalg.norm(x[3:6]))
      axis = x[3:6] / angle if angle > 1e-9 else np.array([1.0, 0.0, 0.0])
      mujoco.mju_axisAngle2Quat(dq, axis, angle)
      quat = np.zeros(4)
      mujoco.mju_mulQuat(quat, base_quat0, dq)
      return self.posture_qpos(x[:3], quat, x[6:])

    def residual(x):
      d = self.kin(unpack(x))
      s = lowest_point(m, d, stance)
      w = lowest_point(m, d, swing)
      com = d.subtree_com[self.palm]
      r = [*(2e3 * (s[:2] - stance_xy)), 2e3 * s[2]]
      r += list(1e3 * (w - swing_target))
      r += list(2e3 * (com[:2] - com_xy)) + [5e2 * (com[2] - com_z)]
      r += list(40.0 * x[3:6])  # palm stays as upright as the stance
      r += list(0.5 * (x[6:] - guess[6:]))
      return np.array(r)

    lower = np.r_[guess[:3] - 0.1, -1.0, -1.0, -1.0, self.lo]
    upper = np.r_[guess[:3] + 0.1, 1.0, 1.0, 1.0, self.hi]
    fit = least_squares(
      residual, np.clip(guess, lower, upper), bounds=(lower, upper), max_nfev=60
    )
    return fit.x, unpack(fit.x)

  def leg_limits(self, leg: str) -> tuple[np.ndarray, np.ndarray]:
    ids = [self.m.joint(f"{leg}_{j}").id for j in ("mcp", "rot", "pip", "dip")]
    rng = self.m.jnt_range[ids]
    return rng[:, 0] + JOINT_MARGIN, rng[:, 1] - JOINT_MARGIN

  def ik_swing(self, qpos_now, swing, target, guess_legs):
    """Swing-leg joints (4) putting its foot at ``target``, from the measured pose."""
    adr = self.leg_q[swing]
    lo, hi = self.leg_limits(swing)
    q = qpos_now.copy()

    def residual(x):
      q[adr] = x
      w = lowest_point(self.m, self.kin(q), swing)
      return np.r_[1e3 * (w - target), 0.3 * (x - guess_legs)]

    fit = least_squares(
      residual, np.clip(guess_legs, lo, hi), bounds=(lo, hi), max_nfev=40
    )
    return fit.x

  def ik_stance(self, qpos_now, base_quat0, stance, stance_xy, com_xy, com_z, guess):
    """Stance-leg joints (4): palm upright, CoM at its height over its measured place,
    stance foot where it stands. Unknowns: the base pose (6) and the four joints."""
    adr = self.leg_q[stance]
    lo, hi = self.leg_limits(stance)
    q = qpos_now.copy()

    def residual(x):
      dq = np.zeros(4)
      angle = float(np.linalg.norm(x[3:6]))
      axis = x[3:6] / angle if angle > 1e-9 else np.array([1.0, 0.0, 0.0])
      mujoco.mju_axisAngle2Quat(dq, axis, angle)
      quat = np.zeros(4)
      mujoco.mju_mulQuat(quat, base_quat0, dq)
      q[:3], q[3:7], q[adr] = x[:3], quat, x[6:]
      d = self.kin(q)
      s = lowest_point(self.m, d, stance)
      com = d.subtree_com[self.palm]
      return np.r_[
        2e3 * (s[:2] - stance_xy),
        2e3 * s[2],
        2e3 * (com[:2] - com_xy),
        1e3 * (com[2] - com_z),
        40.0 * x[3:6],
        0.3 * (x[6:] - guess[6:]),
      ]

    lower = np.r_[guess[:3] - 0.1, -1.0, -1.0, -1.0, lo]
    upper = np.r_[guess[:3] + 0.1, 1.0, 1.0, 1.0, hi]
    fit = least_squares(
      residual, np.clip(guess, lower, upper), bounds=(lower, upper), max_nfev=40
    )
    return fit.x


##
# Whole-body control. The servos are position servos, but their model is known exactly
# (MuJoCo's DC motor: tau = Kt/R * (kp*(target - q) - K*qdot), capped at 0.181 N*m), so
# a target can be chosen every physics step that makes each servo give a chosen torque
# (``Wbc.ctrl_for``). The torques come from a weighted least-squares problem over the
# joint accelerations, the 16 torques and the stance foot's ground force: the rigid-body
# equations of motion and a non-slipping stance foot are (near-)hard rows, the tasks
# (centre-of-mass height, palm attitude, swing foot, tucked ring and thumb) soft rows,
# torques bounded by the cap and the force inside a friction pyramid.
##

MU = 0.4  # friction used by the controller (the tips' is 0.5)
TAU_MAX = 0.17  # N*m, the controller's torque bound (cap 0.181)


class Wbc:
  """Torque controller over the full floating-base model."""

  def __init__(self, m: mujoco.MjModel) -> None:
    self.m = m
    self.qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
    self.dadr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
    self.S = np.zeros((m.nv, m.nu))
    self.S[self.dadr, np.arange(m.nu)] = 1.0
    self.kp_volt = robot.MEASURED_STIFFNESS / robot.TORQUE_PER_VOLT  # V/rad
    self.k_emf = robot.SUPPLY_VOLTAGE / robot.NO_LOAD_SPEED  # V*s/rad
    edges = [(MU, 0, 1), (-MU, 0, 1), (0, MU, 1), (0, -MU, 1)]
    self.cone = np.array(edges, dtype=float).T  # 3 x 4, force = cone @ lambda

  def ctrl_for(self, d: mujoco.MjData, tau: np.ndarray) -> np.ndarray:
    """Servo targets (rad) that make the servos produce ``tau`` (N*m) now."""
    volt = tau / robot.TORQUE_PER_VOLT + self.k_emf * d.qvel[self.dadr]
    volt = np.clip(volt, -robot.SUPPLY_VOLTAGE, robot.SUPPLY_VOLTAGE)
    return d.qpos[self.qadr] + volt / self.kp_volt

  def solve(self, d, contacts, tasks, tau_max=TAU_MAX) -> np.ndarray:
    """contacts: [(point, body)]; tasks: [(J, desired J*qacc, weight)]. Returns tau."""
    m, nv, nu = self.m, self.m.nv, self.m.nu
    nl = 4 * len(contacts)
    n = nv + nu + nl
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(m, d, M)
    rows, rhs = [], []
    jc_force = np.zeros((nv, nl))
    for i, (point, body) in enumerate(contacts):
      jp, jd = np.zeros((3, nv)), np.zeros((3, nv))
      mujoco.mj_jac(m, d, jp, None, point, body)
      mujoco.mj_jacDot(m, d, jd, None, point, body)
      jc_force[:, 4 * i : 4 * i + 4] = jp.T @ self.cone
      row = np.zeros((3, n))
      row[:, :nv] = jp
      rows.append(300.0 * row)  # the stance foot does not accelerate (or slide)
      rhs.append(300.0 * (-jd @ d.qvel - 20.0 * jp @ d.qvel))
    dyn = np.hstack([M, -self.S, -jc_force])
    rows.append(3e3 * dyn)
    rhs.append(3e3 * -(d.qfrc_bias - d.qfrc_passive))
    for jac, acc, weight in tasks:
      row = np.zeros((jac.shape[0], n))
      row[:, :nv] = jac
      rows.append(weight * row)
      rhs.append(weight * np.asarray(acc, dtype=float))
    reg = np.zeros((n, n))
    reg[np.arange(nv), np.arange(nv)] = 1e-3
    reg[nv + np.arange(nu), nv + np.arange(nu)] = 0.3
    reg[nv + nu + np.arange(nl), nv + nu + np.arange(nl)] = 1e-3
    rows.append(reg)
    rhs.append(np.zeros(n))
    lower = np.r_[np.full(nv, -np.inf), np.full(nu, -tau_max), np.zeros(nl)]
    upper = np.r_[np.full(nv, np.inf), np.full(nu, tau_max), np.full(nl, np.inf)]
    sol = lsq_linear(np.vstack(rows), np.concatenate(rhs), bounds=(lower, upper))
    return sol.x[nv : nv + nu]


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


def walk(
  m,
  seconds=8.0,
  step_time=0.2,
  lift=0.010,
  step_length=0.0,
  shift_time=0.3,
  record=False,
  verbose=False,
  trace=False,
):
  """Walk on index + middle from the upright posture; returns metrics (and frames).

  Gait: a linear inverted pendulum (CoM height h, omega = sqrt(g / h)). Before the
  first step the CoM moves (both feet down) to where a steady gait starts; then each
  swing foot lands at the divergent component of motion predicted for touchdown,
  offset sideways by b_lat = width / (exp(omega T) + 1) and backwards by
  b_fwd = step_length / (exp(omega T) - 1), which keeps the motion periodic.
  """
  bp, wbc = Biped(m), Wbc(m)
  q0, _ = bp.upright()
  d = mujoco.MjData(m)
  d.qpos[:] = q0
  d.ctrl[:] = q0[7:]
  mujoco.mj_forward(m, d)
  quat0 = q0[3:7].copy()
  com_z = float(d.subtree_com[bp.palm][2])
  feet = {k: lowest_point(m, d, k).copy() for k in LEGS}
  width = float(np.linalg.norm(feet["if"][:2] - feet["mf"][:2]))
  lateral = (feet["mf"][:2] - feet["if"][:2]) / width  # from index to middle
  forward = np.array([-lateral[1], lateral[0]])
  omega = np.sqrt(G / com_z)
  b_lat = width / (np.exp(omega * step_time) + 1.0)
  b_fwd = step_length / (np.exp(omega * step_time) - 1.0)
  tuck_dofs = [m.jnt_dofadr[m.joint(n).id] for n in TUCK]
  tuck_q = np.array(list(TUCK.values()))
  leg_home = {k: q0[bp.leg_q[k]].copy() for k in LEGS}
  eye = np.eye(m.nv)

  stance, swing = "if", "mf"
  phase, t_phase = "shift", 0.0
  plant = {k: feet[k].copy() for k in LEGS}  # contact points of the planted feet
  swing_start, land = feet[swing].copy(), feet[swing][:2].copy()
  foot_ref = swing_start.copy()
  start_xy = d.subtree_com[bp.palm][:2].copy()
  dt = m.opt.timestep
  steps, counts, frames, fell, saturated = 0, {}, [], False, 0
  n_steps = round(seconds / dt)
  for i in range(n_steps):
    mujoco.mj_forward(m, d)
    mujoco.mj_subtreeVel(m, d)
    com = d.subtree_com[bp.palm].copy()
    vel = d.subtree_linvel[bp.palm].copy()
    if com[2] < 0.6 * com_z:
      fell = True
      break
    jcom = np.zeros((3, m.nv))
    mujoco.mj_jacSubtreeCom(m, d, jcom, bp.palm)
    jrot = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, None, jrot, bp.palm)
    att = quat_error(quat0, d.xquat[bp.palm])
    w_palm = jrot @ d.qvel
    tasks = [
      (jrot[:2], 300.0 * att[:2] - 35.0 * w_palm[:2], 3.0),  # palm roll, pitch
      (jrot[2:], 100.0 * att[2:] - 20.0 * w_palm[2:], 0.5),  # palm yaw
      (jcom[2:], [400.0 * (com_z - com[2]) - 40.0 * vel[2]], 10.0),  # height
      (eye[tuck_dofs], 400.0 * (tuck_q - d.qpos[[a for a in bp.tuck]])
       - 40.0 * d.qvel[tuck_dofs], 1.0),
    ]  # fmt: skip
    if phase == "shift":  # both feet down: CoM to where the first step starts
      s = min(1.0, t_phase / shift_time)
      s = s * s * (3 - 2 * s)
      mid = 0.5 * (plant["if"][:2] + plant["mf"][:2])
      goal = (1 - s) * mid + s * (plant[stance][:2] + b_lat * lateral)
      acc = 200.0 * (goal - com[:2]) - 30.0 * vel[:2]
      tasks.append((jcom[:2], acc, 3.0))
      contacts = [(plant[k], bp.tip_body[k]) for k in LEGS]
      for k in LEGS:
        tasks.append((eye[bp.leg_v[k]], 50.0 * (leg_home[k] - d.qpos[bp.leg_q[k]])
                      - 10.0 * d.qvel[bp.leg_v[k]], 0.05))  # fmt: skip
      if t_phase >= shift_time:
        phase, t_phase = "single", 0.0
    else:
      p_st = plant[stance][:2]
      xi = com[:2] + vel[:2] / omega
      u = min(t_phase / step_time, 1.0)
      if u < 0.75:  # re-plan the landing spot, frozen near the end of the swing
        remaining = step_time - t_phase
        xi_end = p_st + (xi - p_st) * np.exp(omega * remaining)
        side = 1.0 if swing == "mf" else -1.0  # middle is on the +lateral side
        rel = xi_end + side * b_lat * lateral - b_fwd * forward - p_st
        gap = float(np.clip(side * (rel @ lateral), 0.025, 0.080))
        ahead = float(np.clip(rel @ forward, -0.05, 0.05))
        land = p_st + side * gap * lateral + ahead * forward
      ease = u * u * (3 - 2 * u)
      dease = 6 * u * (1 - u) / step_time
      foot_ref = np.r_[
        swing_start[:2] + ease * (land - swing_start[:2]),
        lift * np.sin(np.pi * u) ** 2
        - 0.004 * max(0.0, (u - 0.8) / 0.2)
        - 0.02 * max(0.0, t_phase - step_time),
      ]
      foot_vel = np.r_[dease * (land - swing_start[:2]),
                       lift * np.pi / step_time * np.sin(2 * np.pi * u)]  # fmt: skip
      foot = lowest_point(m, d, swing)
      jfoot = np.zeros((3, m.nv))
      mujoco.mj_jac(m, d, jfoot, None, foot, bp.tip_body[swing])
      acc = 900.0 * (foot_ref - foot) + 60.0 * (foot_vel - jfoot @ d.qvel)
      tasks.append((jfoot, acc, 10.0))
      # The CoM follows the pendulum about the stance foot (no angular momentum).
      tasks.append((jcom[:2], omega**2 * (com[:2] - p_st), 1.0))
      for k in LEGS:
        home = leg_home[k]
        tasks.append((eye[bp.leg_v[k]], 20.0 * (home - d.qpos[bp.leg_q[k]])
                      - 5.0 * d.qvel[bp.leg_v[k]], 0.02))  # fmt: skip
      contacts = [(plant[stance], bp.tip_body[stance])]
      if (u > 0.5 and swing in touching(m, d)) or t_phase > 1.6 * step_time:
        steps += 1
        plant[swing] = lowest_point(m, d, swing).copy()
        plant[swing][2] = 0.0
        stance, swing = swing, stance
        swing_start = lowest_point(m, d, swing).copy()
        t_phase = 0.0
        if verbose:
          rel = com[:2] - plant[stance][:2]
          print(
            f"  paso {steps:2d} t={i * dt:5.2f}s sobre {NAMES[stance]:7s} "
            f"CoM-pie {1e3 * rel @ lateral:+6.1f} lat {1e3 * rel @ forward:+6.1f} "
            f"adelante mm, aterrizó a {1e3 * np.linalg.norm(plant[stance][:2] - land):4.1f} "
            f"mm del plan"
          )
    tau = wbc.solve(d, contacts, tasks)
    if trace and i % 10 == 0:
      p_ref = plant[stance][:2]
      rel, xi_rel = com[:2] - p_ref, com[:2] + vel[:2] / omega - p_ref
      sw_err = (
        1e3 * (lowest_point(m, d, swing) - foot_ref)
        if phase != "shift"
        else np.zeros(3)
      )
      print(
        f"{i * dt:5.2f} {phase[:2]} {stance} CoM {1e3 * rel @ lateral:+6.1f}/"
        f"{1e3 * rel @ forward:+6.1f} DCM {1e3 * xi_rel @ lateral:+6.1f}/"
        f"{1e3 * xi_rel @ forward:+6.1f} z {1e3 * (com[2] - com_z):+5.1f} "
        f"att {np.degrees(att[0]):+5.1f} {np.degrees(att[1]):+5.1f} {np.degrees(att[2]):+5.1f} "
        f"sw {sw_err[0]:+5.1f} {sw_err[1]:+5.1f} {sw_err[2]:+5.1f} "
        f"tau {np.abs(tau).max():.3f} {sorted(touching(m, d))}"
      )
    d.ctrl[:] = wbc.ctrl_for(d, tau)
    mujoco.mj_step(m, d)
    saturated += int(np.any(np.abs(d.actuator_force) > 0.99 * robot.API_EFFORT_LIMIT))
    for name in touching(m, d):
      counts[name] = counts.get(name, 0) + 1
    if record and i % 5 == 0:
      frames.append(d.qpos.copy())
    t_phase += dt
  done = i + 1
  mujoco.mj_forward(m, d)
  moved = d.subtree_com[bp.palm][:2] - start_xy
  return dict(
    fell=fell,
    time_s=round(done * dt, 2),
    steps=steps,
    moved_forward_mm=round(1e3 * float(moved @ forward), 1),
    moved_lateral_mm=round(1e3 * float(moved @ lateral), 1),
    ground_contact={
      NAMES.get(k, k): f"{100 * v / done:.0f} %" for k, v in counts.items()
    },
    torque_at_cap=f"{100 * saturated / done:.0f} % of steps",
  ), frames


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--pair", nargs=2, default=["mf", "th"], choices=DIGITS)
  parser.add_argument("--view", action="store_true")
  parser.add_argument("--free", action="store_true", help="tips may move")
  parser.add_argument("--max-pitch", type=float, default=0.6, help="rad")
  parser.add_argument("--home-weight", type=float, default=0.05)
  parser.add_argument("--end-margin", type=float, default=END_MARGIN, help="m")
  parser.add_argument("--walk", action="store_true", help="walk on index + middle")
  parser.add_argument("--seconds", type=float, default=6.0)
  parser.add_argument("--step-time", type=float, default=0.15)
  parser.add_argument("--step-length", type=float, default=0.0, help="m per step")
  parser.add_argument("--slow", type=float, default=4.0, help="viewer slow-down")
  args = parser.parse_args()
  if args.walk:
    m = build()
    result, frames = walk(
      m,
      seconds=args.seconds,
      step_time=args.step_time,
      step_length=args.step_length,
      record=True,
      verbose=True,
    )
    print("Caminata sobre índice + medio:")
    for key, value in result.items():
      print(f"  {key}: {value}")
    if args.view:  # replay the recorded run, slowed down, in a loop
      d = mujoco.MjData(m)
      with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.6, 90.0, -15.0
        viewer.cam.lookat[:] = (*frames[0][:2], 0.08)
        while viewer.is_running():
          for i in range(0, len(frames), 5):
            if not viewer.is_running():
              break
            d.qpos[:] = frames[i]
            mujoco.mj_forward(m, d)
            viewer.sync()
            time.sleep(5 * m.opt.timestep * args.slow)
          time.sleep(1.0)
    return
  planted = tuple(args.pair)
  m = build()
  stance = settled_stance(m)
  qpos, posture = solve_posture(
    m, stance, planted, args.free, args.max_pitch, args.home_weight, args.end_margin
  )
  print(f"Postura sobre {NAMES[planted[0]]} + {NAMES[planted[1]]}:")
  for key, value in posture.items():
    print(f"  {key}: {value}")
  if args.free:  # start in the posture, tips 1 mm above the floor
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    qpos[2] += 0.001 - min(lowest_point(m, d, k)[2] for k in planted)
    start, ramp = qpos, 0.0
  else:
    start, ramp = stance.qpos, RAMP_S
  result = run_physics(m, start, qpos[7:], ramp_s=ramp)
  print(f"Física (rampa {ramp:.0f} s, mantener {HOLD_S:.0f} s):")
  for key, value in result.items():
    print(f"  {key}: {value}")
  if args.view:
    d = mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m, d) as viewer:
      viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.5, 135.0, -15.0
      viewer.cam.lookat[:] = (*start[:2], 0.06)
      while viewer.is_running():  # stance -> two digits -> hold, then again
        if run_physics(m, start, qpos[7:], ramp, viewer=viewer, d=d) is None:
          break
        time.sleep(1.0)


if __name__ == "__main__":
  main()
