"""The hand as it is being built: feet, IMU, thumb limits, folded-thumb stance.

Plain MuJoCo on the CPU, like test_robot.py.
"""

import mujoco
import numpy as np
import pytest
from mjlab.entity import Entity

from walking_leap import robot
from walking_leap.env_cfg import walking_leap_flat_env_cfg

FEET = robot.FOOT_GEOMS
STANCES = pytest.mark.parametrize("folded_thumb", [False, True])


def build(folded_thumb: bool = False) -> mujoco.MjModel:
  cfg = robot.get_leap_hand_cfg(
    as_built=True, folded_thumb=folded_thumb, effort_limit=robot.API_EFFORT_LIMIT
  )
  spec = Entity(cfg).spec
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01)
  )
  m = spec.compile()
  walking_leap_flat_env_cfg().sim.mujoco.apply(m)
  return m


def nominal_data(m: mujoco.MjModel, lift: float = 0.0) -> mujoco.MjData:
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, m.key("init_state").id)
  d.qpos[2] += lift
  mujoco.mj_forward(m, d)
  return d


def mesh_vertices(m, d, name: str) -> np.ndarray:
  """World positions of a mesh geom's vertices."""
  g = m.geom(name).id
  mesh = m.geom_dataid[g]
  start, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
  vertices = m.mesh_vert[start : start + n]
  return d.geom_xpos[g] + vertices @ d.geom_xmat[g].reshape(3, 3).T


def lowest_point(m, d, name: str) -> float:
  return float(mesh_vertices(m, d, name)[:, 2].min())


def test_parts_add_their_mass_and_nothing_else():
  stock = Entity(robot.get_leap_hand_cfg()).spec.compile()
  m = build()
  for b in range(1, stock.nbody):
    name = stock.body(b).name
    assert m.body(name).mass[0] == pytest.approx(stock.body(name).mass[0])
    assert np.allclose(m.body(name).ipos, stock.body(name).ipos)
  mount = robot.IMU_MASS + robot.IMU_COVER_MASS + robot.IMU_PAD_MASS
  mount += robot.IMU_PLUG_MASS + 3 * robot.IMU_BOLT_MASS
  total = 0.746 + 4 * robot.FOOT_MASS + mount
  assert m.body_subtreemass[m.body("palm").id] == pytest.approx(total, abs=1e-6)
  assert m.body("imu_mount").mass[0] == pytest.approx(mount)
  # No joint and no actuator came with the parts.
  assert m.njnt == stock.njnt and m.nu == stock.nu


def test_mount_balance():
  """Centre of mass of IMU, cover, pad, bolts and plug: on the lid, 7 mm above it."""
  m = build()
  d = nominal_data(m)
  offset = d.xipos[m.body("imu_mount").id] - d.xpos[m.body("palm").id]
  assert np.allclose(offset, (-0.056, -0.035, 0.018), atol=0.002)


def test_cover_holds_the_imu_down():
  """The parts stack on the lid: pad, board, case, and the cover round them."""
  m = build()
  d = nominal_data(m)
  lid = d.xpos[m.body("imu_mount").id][2]
  top = {}
  for name in ("imu_pad", "imu_cover"):
    top[name] = mesh_vertices(m, d, name)[:, 2].max() - lid
  for name in ("imu_board", "imu_case"):
    g = m.geom(name).id
    top[name] = d.geom_xpos[g][2] + m.geom_size[g][2] - lid
  assert top["imu_pad"] == pytest.approx(0.0024, abs=1e-4)
  assert top["imu_board"] == pytest.approx(0.0040, abs=1e-4)
  assert top["imu_case"] == pytest.approx(0.0150, abs=1e-4)
  assert top["imu_cover"] == pytest.approx(0.0174, abs=1e-4)
  # The contact box has the cover's height and ends where its roof does.
  box = m.geom("imu_collision").id
  assert d.geom_xpos[box][2] + m.geom_size[box][2] - lid == pytest.approx(0.0174)
  cover = mesh_vertices(m, d, "imu_cover")
  roof = cover[cover[:, 2] - lid > 0.0173][:, :2]
  half = m.geom_size[box][:2]
  assert np.allclose(roof.min(axis=0), d.geom_xpos[box][:2] - half, atol=2e-4)
  assert np.allclose(roof.max(axis=0), d.geom_xpos[box][:2] + half, atol=2e-4)


def test_feet_are_longer_and_grippier_than_the_tips():
  m = build()
  d = nominal_data(m, lift=0.1)
  for foot in FEET:
    tip = f"{foot[:2]}_tip"
    assert 0.001 < lowest_point(m, d, tip) - lowest_point(m, d, foot) < 0.004
    g = m.geom(foot)
    assert g.friction[0] == pytest.approx(robot.FOOT_FRICTION)
    assert tuple(g.solref) == pytest.approx(robot.FOOT_SOLREF)
    assert g.contype[0] == 1 and g.priority[0] == 1
    # The sock is centred on its body: that point is what clips plant.
    assert m.body(foot).parentid[0] == m.body(f"{foot[:2]}_ds").id


def test_thumb_limits_are_leaps_own():
  m = build()
  limits = robot.joint_limits(as_built=True)
  stock = robot.joint_limits()
  for name in robot.JOINT_NAMES:
    assert tuple(m.joint(name).range) == pytest.approx(limits[name])
    if name not in robot.THUMB_LIMITS:
      assert limits[name] == stock[name]
  # Shifted by one joint in Menagerie's file: each of these two has the range of the
  # joint before it.
  assert limits["th_axl"] == (-0.47, 2.443) == stock["th_mcp"]
  assert stock["th_axl"] == stock["th_cmc"]
  assert limits["th_mcp"] == (-1.20, 1.90)


def test_imu_axes():
  """The IMU is turned a quarter left: palm x = -imu y, palm y = imu x, palm z = imu z."""
  m = build()
  d = nominal_data(m, lift=1.0)
  rng = np.random.default_rng(0)
  for _ in range(5):
    quat = rng.normal(size=4)
    d.qpos[3:7] = quat / np.linalg.norm(quat)
    d.qvel[:] = 0.0
    d.qvel[3:6] = rng.normal(size=3)  # Free joint: angular velocity in the palm frame.
    mujoco.mj_forward(m, d)
    palm = d.xmat[m.body("palm").id].reshape(3, 3)
    gyro, up = d.sensor("imu_ang_vel").data, d.sensor("imu_upvector").data
    for imu, expected in ((gyro, d.qvel[3:6]), (up, palm.T @ (0.0, 0.0, 1.0))):
      assert np.allclose((-imu[1], imu[0], imu[2]), expected, atol=1e-6)


@STANCES
def test_stands_on_its_feet(folded_thumb):
  """Holds its stance for 4 s on the four socks, far below the 350 mA torque."""
  m = build(folded_thumb)
  d = nominal_data(m)
  palm = m.body("palm").id
  assert d.ncon == 0, "no penetration and no self-contact at reset"
  lows = [lowest_point(m, d, foot) for foot in FEET]
  assert 0.0 < min(lows) and max(lows) < 0.003
  z0, xy0 = d.qpos[2], d.qpos[:2].copy()
  peak = 0.0
  for k in range(round(4.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
    if k * m.opt.timestep > 0.5:
      peak = max(peak, float(np.abs(d.actuator_force).max()))
  assert all(w.number == 0 for w in d.warning)
  assert d.xmat[palm].reshape(3, 3)[2, 2] > np.cos(np.radians(2.0))
  assert abs(d.qpos[2] - z0) < 0.005
  assert np.linalg.norm(d.qpos[:2] - xy0) < 0.005
  floor = m.geom("floor").id
  touching = {
    m.geom(c.geom2 if c.geom1 == floor else c.geom1).name
    for c in d.contact[: d.ncon]
    if floor in (c.geom1, c.geom2)
  }
  assert touching == set(FEET)
  # Measured: 0.062 N*m (first stance) and 0.054 (folded) of 0.181.
  assert peak < 0.45 * robot.API_EFFORT_LIMIT


def test_api_effort_limit():
  """350 mA of the datasheet's 1.8 A stall point."""
  assert robot.API_EFFORT_LIMIT == pytest.approx(0.181, abs=1e-3)
  m = build()
  assert np.allclose(np.abs(m.actuator_forcerange), robot.API_EFFORT_LIMIT)
