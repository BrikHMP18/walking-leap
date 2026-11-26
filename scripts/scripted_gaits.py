"""Scripted gaits on the real hand's model: simulation only, no learning.

Model: the real hand as it runs now (IMU, stock fingertips, LEAP's thumb limits,
350 mA cap, stiffness 2.6 N*m/rad, torsional friction), from the folded-thumb stance.
Joint targets come from inverse kinematics plus the servos' static sag and are played
by the position servos.

  uv run python scripts/scripted_gaits.py --teleop
      the fast four-digit crawl steered from the keyboard in the MuJoCo viewer
      (arrows move, ',' '.' turn, the same key again stops); the thumb slides along.
      --thumb-steps: going ahead the thumb steps (it falls when switching).
  uv run python scripts/scripted_gaits.py --stream --fast
      the crawl straight ahead, in a viewer that re-runs it whenever this file is saved

Trials on two digits, which still fall:
  --stream --cane   index + middle stepping, the thumb as a cane (falls after ~2 s)
  --walk [--view]   index + middle only, palm upright, whole-body torque control
                    (falls after one step: one finger cannot carry the hand at 350 mA)
  (no mode flag)    can the hand stand on two digits? A posture search, then physics:
    uv run python scripts/scripted_gaits.py                # middle finger + thumb
    uv run python scripts/scripted_gaits.py --pair rf th   # another pair
    uv run python scripts/scripted_gaits.py --pair if mf --free --max-pitch 1.6
  1. Posture (least squares over the palm pose and the 16 joints): the two chosen
     tips stay planted where they stand in the four-digit stance, the other two are
     lifted by at least LIFT, the CoM lies on the line between the planted tips (and
     away from its ends), the palm stays near level, every joint 0.05 rad inside its
     limits. With ``--free`` the tips may move (only their height is held) and no
     other part may come within CLEARANCE of the floor.
  2. Physics: the servo targets ramp to the posture over RAMP_S (from the posture
     itself with ``--free``), are held for HOLD_S; the run reports falls, contacts,
     the lifted digits' clearance and the peak torque against the cap.
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
NAMES = {"if": "index", "mf": "middle", "rf": "ring", "th": "thumb"}
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
  # Scene as in docs/render_media.py: sky, checkered floor, a light (visual only).
  spec.add_texture(
    name="sky",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=(0.85, 0.9, 0.95),
    rgb2=(0.55, 0.65, 0.75),
    width=256,
    height=256,
  )
  spec.add_texture(
    name="grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(0.82, 0.82, 0.8),
    rgb2=(0.72, 0.72, 0.7),
    mark=mujoco.mjtMark.mjMARK_EDGE,
    markrgb=(0.6, 0.6, 0.6),
    width=300,
    height=300,
  )
  floor_mat = spec.add_material(name="floor", texrepeat=(20, 20), texuniform=True)
  floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01), material="floor"
  )
  sun = spec.worldbody.add_light(pos=(0.3, -0.3, 1.0), dir=(-0.3, 0.3, -1.0))
  sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL  # lights the whole floor
  sun.castshadow = True
  m = spec.compile()
  walking_leap_flat_env_cfg().sim.mujoco.apply(m)
  m.vis.headlight.ambient[:] = 0.4  # brighter viewer (the hand's parts are black)
  m.vis.headlight.diffuse[:] = 0.6
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
    hit.add(name[:2] if name[:2] in DIGITS else "palm/other")
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
  counts = {k: 0 for k in [*DIGITS, "palm/other"]}
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
            f"  step {steps:2d} t={i * dt:5.2f}s on {NAMES[stance]:7s} "
            f"CoM-foot {1e3 * rel @ lateral:+6.1f} lat {1e3 * rel @ forward:+6.1f} "
            f"fwd mm, landed {1e3 * np.linalg.norm(plant[stance][:2] - land):4.1f} "
            f"mm from the plan"
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


##
# Walking with a cane: index and middle step like legs, the thumb stays behind on the
# floor like a cane (a third support) and the ring finger is tucked. Quasi-static, as a
# person's three-point cane gait: with the three digits down the centre of mass moves
# onto the line of the two that stay, CANE_LEAN toward where the lifted digit will
# land, so the hand tips gently onto it; then that digit lifts and moves forward. Order
# per cycle: middle, index, thumb. Joint targets come from inverse kinematics plus the
# servos' static sag, played open loop by the position servos (as the real hand's
# scripted crawl is).
##

CANE = ("if", "mf", "th")
CANE_ORDER = ("mf", "if", "th")
RING_TUCK = {n: v for n, v in TUCK.items() if n.startswith("rf")}
CANE_LEAN = 0.004  # m, CoM past the support line toward the landing digit
CANE_LIFT = 0.010  # m, swing height of a digit
TORQUE_SOFT = 0.10  # N*m, static torque the keyframes try to stay under (cap 0.181)
KEYFRAME_S = 0.02  # s between inverse-kinematics keyframes (50 Hz, as the real clips)


class Cane:
  """Inverse kinematics and statics for the hand on its digit tips (the ring finger
  tucked unless ``tuck_ring`` is False)."""

  def __init__(self, m: mujoco.MjModel, tuck_ring: bool = True) -> None:
    self.m = m
    self.tuck_ring = tuck_ring
    self.d = mujoco.MjData(m)
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
    adr = list(self.jadr)
    self.ring_idx = np.array([adr.index(m.joint(n).qposadr[0]) for n in RING_TUCK])
    self.ring_q = np.array(list(RING_TUCK.values()))
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
    self.d.qpos[:] = q
    mujoco.mj_kinematics(self.m, self.d)
    mujoco.mj_comPos(self.m, self.d)
    return self.d

  def static_torque(self, d, loads: dict) -> np.ndarray:
    """Joint torques (N*m) holding the pose in ``d`` still with ``loads`` (N) on the
    supports (gravity only; ``d`` must hold kinematics and comPos)."""
    m = self.m
    gen = np.zeros(m.nv)
    mujoco.mj_rne(m, d, 0, gen)
    for k, f in loads.items():
      jacp = np.zeros((3, m.nv))
      mujoco.mj_jac(m, d, jacp, None, lowest_point(m, d, k), self.tip_body[k])
      gen -= jacp.T @ np.array([0.0, 0.0, f])
    return gen[self.dadr]

  def solve(self, x0, quat_ref, planted, swing, com_xy, com_z, loads=None):
    """Keyframe: ``planted`` {digit: point} stay put, ``swing`` (digit, point) or None,
    the CoM at ``com_xy`` and near ``com_z``, the palm near ``quat_ref``, the ring
    tucked, the rest of the hand 4 mm above the floor, static torques (for ``loads``)
    kept under TORQUE_SOFT where possible. Returns (x, qpos)."""
    m = self.m
    free_tips = {self.tip_geom[k] for k in planted} | (
      {self.tip_geom[swing[0]]} if swing else set()
    )

    def residual(x):
      d = self.kin(self.qpos(x, quat_ref))
      r = []
      for k, p in planted.items():
        r.extend(3e3 * (lowest_point(m, d, k) - p))
      if swing:
        r.extend(1e3 * (lowest_point(m, d, swing[0]) - swing[1]))
      com = d.subtree_com[self.palm]
      r.extend(2e3 * (com[:2] - com_xy))
      r.append(50.0 * (com[2] - com_z))
      r.extend(5.0 * x[3:6])
      if loads:
        tau = self.static_torque(d, loads)
        r.extend(60.0 * np.maximum(0.0, np.abs(tau) - TORQUE_SOFT))
      if self.tuck_ring:
        r.extend(10.0 * (x[6:][self.ring_idx] - self.ring_q))
      for g in self.others:
        if g not in free_tips:
          gap = mujoco.mj_geomDistance(m, d, self.floor, g, 0.03, None)
          r.append(1e3 * max(0.0, 0.004 - gap))
      r.extend(0.3 * (x[6:] - x0[6:]))
      return np.array(r)

    lower = np.r_[x0[:3] - 0.1, -1.0, -1.0, -1.0, self.lo]
    upper = np.r_[x0[:3] + 0.1, 1.0, 1.0, 1.0, self.hi]
    fit = least_squares(
      residual, np.clip(x0, lower, upper), bounds=(lower, upper), max_nfev=100
    )
    return fit.x, self.qpos(fit.x, quat_ref)

  def diff_ik(self, q, quat_ref, planted, swing, com_xy, com_z, iters=12):
    """Keyframe by differential inverse kinematics from ``q`` (the previous one):
    ``planted`` tips stay, the ``swing`` tip (digit, point) follows its path, the CoM
    goes to (com_xy, com_z), the palm keeps ``quat_ref``. Smooth and fast; no torque
    or clearance terms (the four-digit stance does not need them)."""
    m, d = self.m, self.d
    q = q.copy()
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


def on_line(point, a, b, toward, lean):
  """``point`` projected onto segment a-b (xy), moved ``lean`` toward ``toward``."""
  seg = b - a
  s = float(np.clip((point - a) @ seg / (seg @ seg), 0.1, 0.9))
  foot = a + s * seg
  normal = np.array([-seg[1], seg[0]]) / np.linalg.norm(seg)
  if (toward - foot) @ normal < 0:
    normal = -normal
  return foot + lean * normal


def cane_plan(
  m, cycles=3, step=0.015, shift_s=0.5, swing_s=0.25, lean=CANE_LEAN, verbose=False
):
  """Keyframes (servo targets every KEYFRAME_S) of the cane gait, and the start qpos."""
  cane = Cane(m)
  stance = settled_stance(m)
  quat_ref = stance.qpos[3:7].copy()
  d = cane.kin(stance.qpos.copy())
  planted = {k: lowest_point(m, d, k).copy() for k in CANE}
  for p in planted.values():
    p[2] = 0.0
  com_z = float(d.subtree_com[cane.palm][2])
  ahead = d.xmat[cane.palm].reshape(3, 3)[:2, 0]  # palm +x: wrist to fingertips
  ahead /= np.linalg.norm(ahead)
  com = np.mean([p[:2] for p in planted.values()], axis=0)
  x = np.r_[stance.qpos[:3], 0.0, 0.0, 0.0, stance.qpos[cane.jadr]]
  x, q = cane.solve(x, quat_ref, planted, None, com, com_z, cane.loads(com, planted))
  start = q.copy()
  keys = [cane.targets(q, cane.loads(com, planted))]

  def frames(duration, make):  # make(u) -> (planted, swing, com)
    nonlocal x, q
    n = max(1, round(duration / KEYFRAME_S))
    for i in range(1, n + 1):
      u = i / n
      pl, sw, c = make(0.5 - 0.5 * np.cos(np.pi * u))
      load = cane.loads(c, dict(pl))
      x, q = cane.solve(x, quat_ref, pl, sw, c, com_z, load)
      keys.append(cane.targets(q, load))

  for cycle in range(cycles):
    for digit in CANE_ORDER:
      stay = [k for k in CANE if k != digit]
      land = planted[digit].copy()
      land[:2] += step * ahead
      goal = on_line(com, planted[stay[0]][:2], planted[stay[1]][:2], land[:2], lean)
      com0 = com.copy()
      frames(shift_s, lambda s, c0=com0, g=goal: (planted, None, (1 - s) * c0 + s * g))
      com = goal
      lift0 = planted[digit].copy()
      rest = {k: planted[k] for k in stay}

      def swing(s, a=lift0, b=land, rest=rest, c=goal, digit=digit):
        p = a + s * (b - a)
        p[2] = CANE_LIFT * np.sin(np.pi * s) - 0.002 * s
        return rest, (digit, p), c

      frames(swing_s, swing)
      planted[digit] = land
      if verbose:
        print(f"  cycle {cycle + 1}: {NAMES[digit]} moves {1e3 * step:.0f} mm ahead")
  return start, np.array(keys)


def cane_walk(m, cycles=3, step=0.015, record=False, verbose=False, **plan):
  """Plan the cane gait, play it in physics; returns metrics (and frames)."""
  start, keys = cane_plan(m, cycles=cycles, step=step, verbose=verbose, **plan)
  d = mujoco.MjData(m)
  d.qpos[:] = start
  mujoco.mj_kinematics(m, d)
  d.qpos[2] += 0.0005 - min(lowest_point(m, d, k)[2] for k in CANE)
  d.ctrl[:] = keys[0]
  mujoco.mj_forward(m, d)
  palm = m.body("palm").id
  start_xy = d.xpos[palm][:2].copy()
  ahead = d.xmat[palm].reshape(3, 3)[:2, 0].copy()
  ahead /= np.linalg.norm(ahead)
  per_key = round(KEYFRAME_S / m.opt.timestep)
  counts, frames, fell, at_cap, n = {}, [], False, 0, 0
  for i in range(len(keys) - 1):
    for j in range(per_key):
      d.ctrl[:] = keys[i] + (keys[i + 1] - keys[i]) * (j / per_key)
      mujoco.mj_step(m, d)
      n += 1
      for name in touching(m, d):
        counts[name] = counts.get(name, 0) + 1
      at_cap += int(np.any(np.abs(d.actuator_force) > 0.99 * robot.API_EFFORT_LIMIT))
      if record and n % 5 == 0:
        frames.append(d.qpos.copy())
    if d.xmat[palm].reshape(3, 3)[2, 2] < 0.5 or d.xpos[palm][2] < 0.06:
      fell = True
      break
  moved = d.xpos[palm][:2] - start_xy
  return dict(
    fell=fell,
    time_s=round(n * m.opt.timestep, 2),
    steps=sum(1 for _ in range(len(CANE_ORDER) * cycles)) if not fell else None,
    moved_forward_mm=round(1e3 * float(moved @ ahead), 1),
    ground_contact={NAMES.get(k, k): f"{100 * v / n:.0f} %" for k, v in counts.items()},
    torque_at_cap=f"{100 * at_cap / n:.0f} % of steps",
  ), frames


##
# Fast crawl on the four digits, the real crawl's idea sped up. Each cycle: a drag (the
# palm moves forward over the three planted fingers, the thumb, lightly loaded, slides
# along with it until the CoM is FAST_BACK behind the front row), then ring, middle and
# index each step STEP forward (B cycles: index, middle, ring, so the CoM never has to
# cross a support edge). At every switch the CoM lies inside both the support before
# and the support after, FAST_MARGIN from their edges; it moves while the digits
# swing. Targets get the servos' static sag and a velocity lead (back-EMF damping /
# stiffness) so the fast swings keep up.
##

FAST_MARGIN = 0.004  # m
FAST_BACK = 0.006  # m, CoM behind the front row after a drag
THUMB_LEAN = 0.003  # m, CoM behind the front row while the thumb steps
THUMB_LIFT = 0.008  # m
SLIDE_SPEED = 0.1  # m/s, fastest thumb slide
COM_SPEED = 0.08  # m/s, fastest CoM move with all four down
STEP_TURN_MAX = np.radians(
  3.0
)  # the thumb steps only going ahead, turning at most this
FRONT = ("if", "mf", "rf")
SERVO_LAG = (robot.STALL_TORQUE / robot.NO_LOAD_SPEED) / robot.MEASURED_STIFFNESS  # s


def inside(triangles, nominal, margin):
  """The point closest to ``nominal`` (xy) inside every triangle, ``margin`` from
  their edges."""
  from scipy.optimize import minimize

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


def yaw_quat(quat: np.ndarray, yaw: float) -> np.ndarray:
  """``quat`` turned by ``yaw`` (rad) about the world's vertical."""
  turn, out = np.zeros(4), np.zeros(4)
  mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), yaw)
  mujoco.mju_mulQuat(out, turn, quat)
  return out


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
  ):
    self.m, self.swing_s, self.drag_s, self.lift = m, swing_s, drag_s, lift
    self.shift_s, self.thumb_s, self.thumb_drag = shift_s, thumb_s, thumb_drag
    self.thumb_lean, self.thumb_lift = thumb_lean, thumb_lift
    self.thumb_margin = thumb_margin
    self.cane = Cane(m, tuck_ring=False)
    self.palm = self.cane.palm
    stance = settled_stance(m)
    self.start = stance.qpos.copy()
    d = self.cane.kin(self.start)
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
    # Thumb dragging: around the CoM, the front row ``reach`` ahead of it.
    com = d.subtree_com[self.palm][:2]
    self.offset_com = {k: rot @ (tips[k] - com) for k in DIGITS}
    shift = reach - np.mean([self.offset_com[k][0] for k in FRONT])
    for k in FRONT:
      self.offset_com[k] = self.offset_com[k] + np.array([shift, 0.0])
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
    d = self.cane.kin(q)
    planted = {k: lowest_point(self.m, d, k) for k in DIGITS}
    return self.cane.targets(q, self.cane.loads(d.subtree_com[self.palm][:2], planted))

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
    m, cane = self.m, self.cane
    d = cane.kin(q)
    planted = {k: lowest_point(m, d, k).copy() for k in DIGITS}
    for p in planted.values():
      p[2] = 0.0
    com = d.subtree_com[self.palm][:2].copy()
    yaw = self.heading(d)
    forward, left, turn = command
    move = self.rot(yaw) @ np.array([forward, left])
    yaw_new = yaw + turn
    step = (mode or self.mode(command)) == "step"
    if not step:  # the real crawl's layout, around the CoM
      base, offset = com + move, self.offset_com
    else:  # around the index-ring midpoint, the middle finger ahead
      base = 0.5 * (planted["if"][:2] + planted["rf"][:2]) + move
      offset = self.offset
    goal_tip = {k: base + self.rot(yaw_new) @ offset[k] for k in DIGITS}
    order = self.ORDERS[self.count % 2]
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
        q_new = cane.diff_ik(plan[-1], quat, stay, swing, c, self.com_z)
        plan.append(q_new)
        keys.append(cane.targets(q_new, cane.loads(c, support)))
      if digit:
        planted[digit] = np.r_[end, 0.0]

    if not step:  # the body moves and turns, the thumb slides along
      after = dict(planted)
      after["th"] = np.r_[goal_tip["th"], 0.0]
      goal = inside([tri(order[0], after)], com + move, FAST_MARGIN)
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


def fast_plan(m, cycles=4, step=0.035, verbose=False, **crawl):
  """Keyframes (servo targets every KEYFRAME_S) of the fast crawl straight ahead, and
  the start qpos."""
  crawler = Crawler(m, **{"thumb_drag": True, **crawl})  # the robust gait by default
  q = crawler.start.copy()
  keys = [crawler.targets_hold(q)]
  for cycle in range(cycles):
    k, q = crawler.cycle(q, (step, 0.0, 0.0), drag_s=1.0 if cycle == 0 else None)
    keys.extend(k)
    if verbose:
      print(f"  cycle {cycle + 1}")
  return crawler.start, np.array(keys)


def fast_walk(m, cycles=4, record=False, verbose=False, **plan):
  """Plan the fast crawl and play it in physics; returns metrics (and frames)."""
  start, keys = fast_plan(m, cycles=cycles, verbose=verbose, **plan)
  lead = np.gradient(keys, KEYFRAME_S, axis=0) * SERVO_LAG
  keys = keys + lead
  d = mujoco.MjData(m)
  d.qpos[:] = start
  mujoco.mj_kinematics(m, d)
  d.qpos[2] += 0.0005 - min(lowest_point(m, d, k)[2] for k in DIGITS)
  d.ctrl[:] = keys[0]
  mujoco.mj_forward(m, d)
  palm = m.body("palm").id
  start_xy = d.xpos[palm][:2].copy()
  ahead = d.xmat[palm].reshape(3, 3)[:2, 0].copy()
  ahead /= np.linalg.norm(ahead)
  per_key = round(KEYFRAME_S / m.opt.timestep)
  counts, frames, fell, at_cap, n = {}, [], False, 0, 0
  for i in range(len(keys) - 1):
    for j in range(per_key):
      d.ctrl[:] = keys[i] + (keys[i + 1] - keys[i]) * (j / per_key)
      mujoco.mj_step(m, d)
      n += 1
      for name in touching(m, d):
        counts[name] = counts.get(name, 0) + 1
      at_cap += int(np.any(np.abs(d.actuator_force) > 0.99 * robot.API_EFFORT_LIMIT))
      if record and n % 5 == 0:
        frames.append(d.qpos.copy())
    if d.xmat[palm].reshape(3, 3)[2, 2] < 0.5 or d.xpos[palm][2] < 0.06:
      fell = True
      break
  seconds = n * m.opt.timestep
  moved = float((d.xpos[palm][:2] - start_xy) @ ahead)
  return dict(
    fell=fell,
    time_s=round(seconds, 2),
    moved_forward_mm=round(1e3 * moved, 1),
    speed_cm_s=round(100 * moved / seconds, 2),
    ground_contact={NAMES.get(k, k): f"{100 * v / n:.0f} %" for k, v in counts.items()},
    torque_at_cap=f"{100 * at_cap / n:.0f} % of steps",
  ), frames


# Keys (GLFW codes): arrows move, ',' and '.' turn; the same key again stops.
TELEOP_KEYS = {
  265: ("forward", (1.0, 0.0, 0.0)),
  264: ("back", (-1.0, 0.0, 0.0)),
  263: ("left", (0.0, 1.0, 0.0)),
  262: ("right", (0.0, -1.0, 0.0)),
  44: ("turn left", (0.0, 0.0, 1.0)),
  46: ("turn right", (0.0, 0.0, -1.0)),
}
TELEOP_STEP = (0.030, 0.015, np.radians(20.0))  # per cycle: forward m, sideways m, turn


def steer(command, yaw_ref, yaw):
  """Command for the next cycle: a translation also turns back to the heading held
  since the last turn (the digits' slips make the hand drift in yaw)."""
  command = np.array(command, dtype=float)
  if command[:2].any() and not command[2]:
    error = (yaw_ref - yaw + np.pi) % (2 * np.pi) - np.pi
    limit = STEP_TURN_MAX if command[0] > 0 and not command[1] else TELEOP_STEP[2]
    command[2] = np.clip(error, -limit, limit)  # going ahead: small, the thumb steps
  return command


def teleop(thumb_open: float = 0.0, thumb_steps: bool = False) -> None:
  """Drive the fast crawl from the keyboard in the MuJoCo viewer, live physics.
  The thumb slides (robust: three test sequences of 5-6 commands, 15-17 s, no
  fall); ``thumb_steps`` makes it step going ahead (6 cycles straight ahead without
  falling, but changing between stepping and sliding it fell in all three)."""
  from collections import deque

  m = build()
  crawler = Crawler(m, thumb_open=thumb_open, thumb_drag=not thumb_steps)
  d = mujoco.MjData(m)
  d.qpos[:] = crawler.start
  mujoco.mj_kinematics(m, d)
  d.qpos[2] += 0.0005 - min(lowest_point(m, d, k)[2] for k in DIGITS)
  hold = crawler.targets_hold(d.qpos.copy())
  d.ctrl[:] = hold
  mujoco.mj_forward(m, d)
  state = {"name": "stopped", "command": np.zeros(3), "yaw": crawler.heading(d)}

  def key(code: int) -> None:
    if code not in TELEOP_KEYS:
      return
    name, unit = TELEOP_KEYS[code]
    if state["name"] == name:
      state["name"], state["command"] = "stopped", np.zeros(3)
    else:
      state["name"], state["command"] = name, np.array(unit) * TELEOP_STEP

  queue, last = deque(), hold
  per_key = round(KEYFRAME_S / m.opt.timestep)
  print("Arrows: forward/back/left/right; ',' and '.': turn; the same key again: stop")
  with mujoco.viewer.launch_passive(m, d, key_callback=key) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = m.body("palm").id
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.6, 135.0, -25.0
    while viewer.is_running():
      if not queue and state["command"].any():
        from_rest = not state.get("moving")
        yaw = crawler.heading(d)
        if state["command"][2]:  # turning: the heading to hold is the new one
          state["yaw"] = yaw + state["command"][2]
        command = steer(state["command"], state["yaw"], yaw)
        keys, _ = crawler.cycle(
          d.qpos.copy(), command, drag_s=1.0 if from_rest else None
        )
        queue.extend(keys)
        state["moving"] = True
      elif not queue:
        state["moving"] = False
      target = queue.popleft() if queue else last
      t0 = time.perf_counter()
      with viewer.lock():
        for j in range(per_key):
          d.ctrl[:] = last + (target - last) * ((j + 1) / per_key)
          mujoco.mj_step(m, d)
      last = target
      caption(
        viewer, f"{state['name']}  (arrows move, ',' '.' turn, same key again stops)"
      )
      viewer.sync()
      time.sleep(max(0.0, KEYFRAME_S - (time.perf_counter() - t0)))


def caption(viewer, text: str) -> None:
  import unicodedata  # MuJoCo's font is ASCII only

  text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
  try:
    viewer.set_texts(
      (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, "")
    )
  except Exception:  # viewers without overlay text
    pass


def stream(args) -> None:
  """Live view: replay the walk in a loop and re-run it whenever this file is saved."""
  import importlib
  import pathlib

  path = pathlib.Path(__file__).resolve()
  m = build()
  d = mujoco.MjData(m)
  frames, stamp = [], None
  with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = m.body("palm").id
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.5, 135.0, -15.0
    while viewer.is_running():
      if path.stat().st_mtime != stamp:
        stamp = path.stat().st_mtime
        caption(viewer, "computing the new version...")
        try:
          mod = importlib.reload(importlib.import_module(path.stem))
          if args.fast:
            result, new = mod.fast_walk(mod.build(), cycles=args.cycles, record=True)
          elif args.cane:
            result, new = mod.cane_walk(mod.build(), cycles=args.cycles, record=True)
          else:
            result, new = mod.walk(
              mod.build(),
              seconds=args.seconds,
              step_time=args.step_time,
              step_length=args.step_length,
              record=True,
            )
          frames = new
          end = f"falls at {result['time_s']} s" if result["fell"] else "does not fall"
          steps = f"{result['steps']} steps, " if result.get("steps") else ""
          if "speed_cm_s" in result:
            steps = f"{result['speed_cm_s']} cm/s, "
          label = (
            f"version {time.strftime('%H:%M:%S', time.localtime(stamp))}: "
            f"{steps}{end}, {result['moved_forward_mm']} mm ahead, "
            f"contacts {result['ground_contact']}"
          )
        except Exception as e:  # a half-written edit: keep the last run
          label = (
            f"error in the new version ({type(e).__name__}: {e}); showing the last one"
          )
        print(label, flush=True)
        caption(viewer, label)
      for q in frames:
        if not viewer.is_running() or path.stat().st_mtime != stamp:
          break
        with viewer.lock():
          d.qpos[:] = q
          mujoco.mj_forward(m, d)
        viewer.sync()
        time.sleep(5 * m.opt.timestep * args.slow)
      time.sleep(0.7)


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
  parser.add_argument("--step-time", type=float, default=0.2)
  parser.add_argument("--step-length", type=float, default=0.0, help="m per step")
  parser.add_argument("--slow", type=float, default=2.0, help="viewer slow-down")
  parser.add_argument("--stream", action="store_true", help="live view of the walk")
  parser.add_argument("--cane", action="store_true", help="thumb as a cane")
  parser.add_argument("--fast", action="store_true", help="fast four-digit crawl")
  parser.add_argument("--teleop", action="store_true", help="steer it with the keys")
  parser.add_argument("--thumb-open", type=float, default=0.0, help="m, thumb outward")
  parser.add_argument("--thumb-steps", action="store_true", help="thumb steps ahead")
  parser.add_argument("--cycles", type=int, default=3, help="cane-gait cycles")
  args = parser.parse_args()
  if args.teleop:
    teleop(args.thumb_open, args.thumb_steps)
    return
  if args.stream:
    stream(args)
    return
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
    print("Walk on index + middle:")
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
  print(f"Posture on {NAMES[planted[0]]} + {NAMES[planted[1]]}:")
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
  print(f"Physics (ramp {ramp:.0f} s, hold {HOLD_S:.0f} s):")
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
