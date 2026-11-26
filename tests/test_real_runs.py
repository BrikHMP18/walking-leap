"""The real hand's recordings (docs/data/real_runs/) against the simulator.

Each recording is a scripted run on the real hand: the motor targets that were sent, what
the motors and the IMU read back. Replaying the targets in plain MuJoCo must keep the palm's
"down" (the IMU's gravity direction) close to the real one: a model change that breaks this
breaks sim-to-real. Hand as it ran: stock fingertips, LEAP's own thumb limits, the IMU on
its holder, gains 1200 / 0 / 600 (about 2.6 N*m/rad, docs/real_hand.md), 350 mA.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from walking_leap import robot

RUNS = Path(__file__).parents[1] / "docs" / "data" / "real_runs"
PUSHUPS = RUNS / "20261125_174531_stand_up_pushups_lie_down.npz"

DT = 0.002
# N*m/rad at 1200 / 0 / 600: the best fit over five runs (docs/real_hand.md).
STIFFNESS = 2.6
TIP_FRICTION = 0.8
BACK_EMF = 0.11  # N*m*s/rad, as robot.py's motor model
ABDUCTION_SCALE = 0.75  # the abduction motors run at 0.75 of the gains (LEAP's API)


def load(path: Path) -> dict:
  d = np.load(path)
  run = {k: d[k] for k in d.files if k != "meta"}
  run["meta"] = json.loads(str(d["meta"]))
  return run


def stock_hand() -> tuple[mujoco.MjModel, np.ndarray, np.ndarray]:
  """The hand as it ran: stock tips, LEAP's thumb limits, the IMU, on a floor."""
  spec = robot.get_spec(as_built=False)
  for joint, limits in robot.THUMB_LIMITS.items():
    spec.joint(joint).range = limits
  robot.add_imu(spec)
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1, 1, 0.1)
  )
  spec.option.timestep = DT
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = 10
  for joint in spec.joints:
    if joint.name in robot.JOINT_NAMES:
      joint.armature = robot.ARMATURE
      joint.damping[:] = 0.0
  m = spec.compile()
  for g in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    collides = name == "floor" or "_collision" in name or name.endswith("_tip")
    m.geom_contype[g] = m.geom_conaffinity[g] = int(collides)
    if collides and name != "floor":
      m.geom_priority[g], m.geom_condim[g] = 1, 3
      m.geom_solref[g, 0] = max(m.geom_solref[g, 0], 0.004)
      if name.endswith("_tip"):
        m.geom_friction[g, 0] = TIP_FRICTION
  qadr = np.array([m.joint(n).qposadr[0] for n in robot.JOINT_NAMES])
  vadr = np.array([m.joint(n).dofadr[0] for n in robot.JOINT_NAMES])
  return m, qadr, vadr


def replay(run: dict) -> tuple[np.ndarray, np.ndarray]:
  """Sent targets replayed from the real starting pose: the sim's gravity at each one,
  and MuJoCo's warning counters."""
  m, qadr, vadr = stock_hand()
  d = mujoco.MjData(m)
  lo, hi = m.jnt_range[m.jnt_qposadr.searchsorted(qadr)].T
  kp = np.full(16, STIFFNESS)
  kp[[1, 5, 9]] *= ABDUCTION_SCALE  # if_rot, mf_rot, rf_rot
  cap = robot.API_EFFORT_LIMIT
  palm = m.body(robot.ROOT_BODY).id
  floor = m.geom("floor").id
  # Lying as it started: joints as measured, palm just touching the floor.
  d.qpos[qadr] = np.clip(run["q"][0], lo + 1e-3, hi - 1e-3)
  d.qpos[:7] = (0, 0, 0.2, 1, 0, 0, 0)
  mujoco.mj_forward(m, d)
  gap = min(
    mujoco.mj_geomDistance(m, d, g, floor, 1.0, None)
    for g in range(m.ngeom)
    if g != floor and m.geom_contype[g]
  )
  d.qpos[2] -= gap - 0.002

  def step(target: np.ndarray) -> None:
    tau = kp * (target - d.qpos[qadr]) - BACK_EMF * d.qvel[vadr]
    d.qfrc_applied[vadr] = np.clip(tau, -cap, cap)
    mujoco.mj_step(m, d)

  for _ in range(int(0.5 / DT)):
    step(d.qpos[qadr].copy())
  sent = np.clip(run["sent"], lo, hi)
  previous, gravity = sent[0], []
  for target in sent:  # 50 Hz; the bus applies each target one 10 ms tick late
    for t in (previous, target):
      for _ in range(5):
        step(t)
    previous = target
    gravity.append(d.xmat[palm].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0]))
  warnings = np.array([w.number for w in d.warning])
  return np.array(gravity), warnings


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  a = a / np.linalg.norm(a, axis=1, keepdims=True)
  b = b / np.linalg.norm(b, axis=1, keepdims=True)
  return np.degrees(np.arccos(np.clip((a * b).sum(axis=1), -1, 1)))


@pytest.mark.parametrize("path", sorted(RUNS.glob("*.npz")), ids=lambda p: p.stem)
def test_recording_is_complete(path):
  run = load(path)
  meta = run["meta"]
  assert tuple(meta["joint_names"]) == robot.JOINT_NAMES
  assert (
    meta["finished"]
    and meta["clips"][0] == "stand_up"
    and meta["clips"][-1] == "lie_down"
  )
  for t in ("motor_t", "imu_t", "cmd_t"):
    assert np.all(np.diff(run[t]) > 0), f"{t} not increasing"
  assert run["q"].shape == run["current"].shape == (len(run["motor_t"]), 16)
  assert run["sent"].shape == (len(run["cmd_t"]), 16)
  # About 100 Hz of motors, 100 Hz of IMU, 50 Hz of commands, one run.
  span = run["cmd_t"][-1] - run["cmd_t"][0]
  assert len(run["cmd_t"]) == pytest.approx(50 * span, rel=0.02)
  assert len(run["motor_t"]) == pytest.approx(100 * span, rel=0.1)
  assert np.abs(np.linalg.norm(run["gravity"], axis=1) - 1).max() < 1e-3
  assert np.abs(run["current"]).max() < 500


def test_sim_follows_the_real_pushups():
  """Stand up, push-ups, lie down: the sim's "down" within 3 deg of the real IMU on
  average (1.5 measured when this test was written; a softer, more slippery hand,
  1.3 N*m/rad and grip 0.3, was 4), and no simulation warning."""
  run = load(PUSHUPS)
  gravity, warnings = replay(run)
  real = np.stack(
    [np.interp(run["cmd_t"], run["imu_t"], run["gravity"][:, k]) for k in range(3)], 1
  )
  off = angle_deg(gravity, real)
  assert not warnings.any(), "MuJoCo warned during the replay"
  assert off.mean() < 3.0, f"{off.mean():.2f} deg on average"
  assert np.percentile(off, 95) < 6.0
  # The stand itself: palm level within a few degrees, like the real one.
  standing = run["clip"] == 1  # push-ups
  assert np.degrees(np.arccos(-gravity[standing, 2])).max() < 8.0
