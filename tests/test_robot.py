"""Robot model and physics validation.

Plain MuJoCo on the CPU (no GPU needed), using the exact spec mjlab builds for
the entity plus a ground plane, with the environment's MujocoCfg applied.
"""

import hashlib

import mujoco
import numpy as np
import pytest
from mjlab.entity import Entity

from leap_walking import robot
from leap_walking.env_cfg import leap_walking_flat_env_cfg

# sha256 of Menagerie's leap_hand/right_hand.xml at robot.MENAGERIE_COMMIT.
UPSTREAM_XML_SHA256 = "9d8f25c38bdb60617f378c07b7e71f97306471353c81bf281eb9943f3f2340d8"
TIPS = robot.FINGERTIP_GEOMS
G = 9.81


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  spec = Entity(robot.get_leap_hand_cfg()).spec
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01)
  )
  m = spec.compile()
  leap_walking_flat_env_cfg().sim.mujoco.apply(m)
  return m


def nominal_data(m: mujoco.MjModel, lift: float = 0.0) -> mujoco.MjData:
  """Data at the init_state keyframe (actuator targets = nominal pose)."""
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, m.key("init_state").id)
  d.qpos[2] += lift
  mujoco.mj_forward(m, d)
  return d


def assert_no_sim_warnings(d: mujoco.MjData) -> None:
  """MuJoCo silently resets the state after a divergence, so check its counters."""
  for w in (
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_CONTACTFULL,
    mujoco.mjtWarning.mjWARN_CNSTRFULL,
  ):
    assert d.warning[w].number == 0, f"MuJoCo warning {w.name}"


def joint_qpos(m, d, name):
  return d.qpos[m.joint(name).qposadr[0]]


def joint_qvel(m, d, name):
  return d.qvel[m.joint(name).dofadr[0]]


def hand_geoms(m) -> list[int]:
  floor = m.geom("floor").id
  return [g for g in range(m.ngeom) if m.geom_contype[g] and g != floor]


def lowest_point(m, d, g) -> float:
  """Lowest world z over the vertices of a box or mesh collision geom."""
  if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
    corners = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
    local = corners * m.geom_size[g]
  else:
    assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    mesh = m.geom_dataid[g]
    start, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
    local = m.mesh_vert[start : start + n]
  world = d.geom_xpos[g] + local @ d.geom_xmat[g].reshape(3, 3).T
  return float(world[:, 2].min())


def tilt_deg(d, body_id: int) -> float:
  up_z = d.xmat[body_id].reshape(3, 3)[2, 2]
  return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


def floor_contact_geoms(m, d) -> set[str]:
  floor = m.geom("floor").id
  names = set()
  for c in d.contact[: d.ncon]:
    if floor in (c.geom1, c.geom2):
      names.add(m.geom(c.geom2 if c.geom1 == floor else c.geom1).name)
  return names


##
# Model structure.
##


def test_asset_matches_upstream():
  digest = hashlib.sha256(robot.LEAP_HAND_XML.read_bytes()).hexdigest()
  assert digest == UPSTREAM_XML_SHA256


def test_floating_base(model):
  m = model
  free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
  assert free == [0]
  palm = m.body(robot.ROOT_BODY).id
  assert m.jnt_bodyid[0] == palm and m.body_parentid[palm] == 0
  # Nothing else ties the hand to the world or pushes it.
  assert m.neq == 0 and m.ntendon == 0 and m.nmocap == 0
  assert np.all(m.body_gravcomp == 0)
  assert m.body_weldid[palm] == palm  # palm is not welded to the world
  # No actuator drives the free joint.
  assert np.all(m.actuator_trnid[:, 0] != 0)


def test_sixteen_actuated_finger_joints(model):
  m = model
  hinges = [m.joint(j).name for j in range(1, m.njnt)]
  assert tuple(hinges) == robot.JOINT_NAMES
  assert all(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in range(1, m.njnt))
  assert m.nu == 16
  for i, name in enumerate(robot.JOINT_NAMES):
    assert m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
    assert m.actuator_trnid[i, 0] == m.joint(name).id, "actuator/joint order"
  # Exactly one actuator per joint (the XML ones were removed).
  assert len(set(m.actuator_trnid[:, 0])) == 16


def test_masses_and_inertias(model):
  m = model
  for b in range(1, m.nbody):
    assert m.body_mass[b] > 0
    i1, i2, i3 = m.body_inertia[b]
    assert min(i1, i2, i3) > 0
    assert i1 + i2 >= i3 and i1 + i3 >= i2 and i2 + i3 >= i1  # triangle ineq.
  # Menagerie totals: palm 0.237 kg + 3 x 0.129 kg fingers + 0.122 kg thumb.
  assert m.body_subtreemass[m.body(robot.ROOT_BODY).id] == pytest.approx(0.746)


def test_joint_limits_and_nominal_pose(model):
  m = model
  limits = robot.joint_limits()
  for name in robot.JOINT_NAMES:
    j = m.joint(name).id
    assert m.jnt_limited[j]
    lo, hi = m.jnt_range[j]
    assert lo < hi and (lo, hi) == pytest.approx(limits[name])
  d = nominal_data(model)
  for name in robot.JOINT_NAMES:
    lo, hi = limits[name]
    q = joint_qpos(m, d, name)
    assert lo + 0.25 <= q <= hi - 0.25, f"{name} too close to a limit"


def test_actuator_model(model):
  """DC-motor position servo: 3 N*m/rad small-signal, back-EMF, torque cap."""
  m = model
  i = robot.JOINT_NAMES.index("if_pip")
  dof = m.joint("if_pip").dofadr[0]
  assert m.dof_armature[dof] == pytest.approx(robot.ARMATURE)
  assert m.dof_damping[dof] == 0.0
  assert tuple(m.actuator_forcerange[i]) == (-robot.EFFORT_LIMIT, robot.EFFORT_LIMIT)

  d = nominal_data(m, lift=1.0)
  q = joint_qpos(m, d, "if_pip")

  def torque(error: float, qdot: float = 0.0) -> float:
    d.ctrl[i] = q + error
    d.qvel[:] = 0.0
    d.qvel[dof] = qdot
    mujoco.mj_forward(m, d)
    return float(d.actuator_force[i])

  assert torque(0.01) == pytest.approx(0.01 * robot.STIFFNESS, rel=1e-6)
  # Back-EMF damping K^2/R = stall_torque / no_load_speed.
  b = robot.STALL_TORQUE / robot.NO_LOAD_SPEED
  assert torque(0.0, qdot=1.0) == pytest.approx(-b, rel=1e-3)
  # Saturation: large errors are capped at the effort limit.
  assert torque(0.1) == pytest.approx(0.3, rel=1e-6)
  assert torque(1.0) == pytest.approx(robot.EFFORT_LIMIT)
  assert torque(-1.0) == pytest.approx(-robot.EFFORT_LIMIT)


##
# Initial configuration.
##


def test_initial_pose_rests_on_four_digit_tips(model):
  m = model
  d = nominal_data(m)
  lows = {m.geom(g).name: lowest_point(m, d, g) for g in hand_geoms(m)}
  zmin = min(lows.values())
  assert 0.0 < zmin < 0.002, "nominal pose must start just above the ground"
  for tip in TIPS:
    assert lows[tip] - zmin < 0.001, f"{tip} is not a support point"
  palm_low = min(v for k, v in lows.items() if k.startswith("palm"))
  assert palm_low > 0.10
  assert d.ncon == 0, "no penetration and no self-collision in the nominal pose"


def test_palm_frame_conventions(model):
  """Identity root orientation = palm down; +x forward, +y index side, +z dorsal."""
  m = model
  d = nominal_data(m)
  palm = m.body(robot.ROOT_BODY).id
  assert np.allclose(d.xmat[palm].reshape(3, 3), np.eye(3))
  tip = {name: d.geom(name).xpos - d.xpos[palm] for name in TIPS}
  # Fingers support the front, the thumb the rear (along +x).
  for name in ("if_tip", "mf_tip", "rf_tip"):
    assert tip[name][0] > 0.0 > tip["th_tip"][0]
  # Index (next to the thumb) on +y, ring on -y.
  assert tip["if_tip"][1] > tip["mf_tip"][1] > tip["rf_tip"][1]
  # Digits hang below the palm: the palm faces -z (the ground).
  assert all(t[2] < -0.1 for t in tip.values())
  # Straight fingers point along +x.
  d.qpos[7:] = 0.0
  mujoco.mj_forward(m, d)
  direction = d.geom("mf_tip").xpos - d.body("mf_bs").xpos
  assert direction[0] > 0.9 * np.linalg.norm(direction)


##
# Physics.
##


def test_gravity_acts_on_floating_base(model):
  m = model
  d = nominal_data(m, lift=1.0)
  palm = m.body(robot.ROOT_BODY).id
  com0 = d.subtree_com[palm].copy()
  t = 0.3
  for _ in range(round(t / m.opt.timestep)):
    mujoco.mj_step(m, d)
  assert_no_sim_warnings(d)
  mujoco.mj_forward(m, d)
  drop = com0[2] - d.subtree_com[palm][2]
  assert drop == pytest.approx(0.5 * G * t**2, rel=0.02)
  assert np.linalg.norm(d.subtree_com[palm][:2] - com0[:2]) < 1e-3


def test_zero_action_static_support(model):
  """Holding the nominal pose on the ground: the hand supports its own weight."""
  m = model
  d = nominal_data(m)
  palm = m.body(robot.ROOT_BODY).id
  z0, xy0 = d.qpos[2], d.qpos[:2].copy()
  max_pen, max_tau = 0.0, 0.0
  for k in range(round(4.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
    if d.ncon:
      max_pen = max(max_pen, -float(d.contact.dist[: d.ncon].min()))
    if k * m.opt.timestep > 0.5:  # after the initial ~1 mm drop
      max_tau = max(max_tau, float(np.abs(d.actuator_force).max()))
  assert_no_sim_warnings(d)
  mujoco.mj_forward(m, d)
  assert tilt_deg(d, palm) < 2.0
  assert abs(d.qpos[2] - z0) < 0.005
  assert np.linalg.norm(d.qpos[:2] - xy0) < 0.005
  assert np.abs(d.qvel).max() < 1e-2
  assert floor_contact_geoms(m, d) == set(TIPS)
  assert max_pen < 0.001
  # Static load uses < 25% of the torque limit (measured: ~0.04 N*m).
  assert max_tau < 0.25 * robot.EFFORT_LIMIT


def test_individual_joint_actuation(model):
  """Each actuator moves its own joint (hand floating, gravity off)."""
  m = model
  gravity = m.opt.gravity.copy()
  m.opt.gravity[:] = 0.0
  try:
    limits = robot.joint_limits()
    for i, name in enumerate(robot.JOINT_NAMES):
      d = nominal_data(m, lift=1.0)
      q0 = d.qpos[7:].copy()
      lo, hi = limits[name]
      delta = 0.2 if hi - q0[i] > q0[i] - lo else -0.2
      d.ctrl[i] = q0[i] + delta
      for _ in range(round(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
      assert_no_sim_warnings(d)
      moved = d.qpos[7:] - q0
      assert moved[i] == pytest.approx(delta, abs=0.02), name
      others = np.delete(np.abs(moved), i)
      assert others.max() < 0.02, f"{name} disturbed other joints"
  finally:
    m.opt.gravity[:] = gravity


def test_joint_speed_within_datasheet(model):
  """Full-range moves saturate near, never above, the 8.48 rad/s no-load speed."""
  m = model
  gravity = m.opt.gravity.copy()
  m.opt.gravity[:] = 0.0
  try:
    for name, target in (("if_pip", 1.885), ("th_cmc", -0.349), ("th_ipl", 1.88)):
      d = nominal_data(m, lift=1.0)
      d.ctrl[robot.JOINT_NAMES.index(name)] = target
      peak = 0.0
      for _ in range(round(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        peak = max(peak, abs(joint_qvel(m, d, name)))
      assert_no_sim_warnings(d)
      assert 0.9 * robot.NO_LOAD_SPEED < peak <= 1.001 * robot.NO_LOAD_SPEED, name
  finally:
    m.opt.gravity[:] = gravity


def test_self_collision_between_fingers(model):
  """Index and middle fingers collide when abducted toward each other."""
  m = model
  gravity = m.opt.gravity.copy()
  m.opt.gravity[:] = 0.0
  try:
    d = nominal_data(m, lift=1.0)
    d.ctrl[:] = 0.0  # straight fingers
    d.ctrl[robot.JOINT_NAMES.index("if_rot")] = 1.0  # index toward middle
    d.ctrl[robot.JOINT_NAMES.index("mf_rot")] = -1.0  # middle toward index
    hit, max_pen = False, 0.0
    for _ in range(round(1.0 / m.opt.timestep)):
      mujoco.mj_step(m, d)
      for c in d.contact[: d.ncon]:
        b1, b2 = (m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
        if {b1[:2], b2[:2]} == {"if", "mf"}:
          hit = True
          max_pen = max(max_pen, -float(c.dist))
    assert_no_sim_warnings(d)
    assert hit
    assert max_pen < 0.002
    assert joint_qpos(m, d, "if_rot") < 0.9  # blocked before its target
  finally:
    m.opt.gravity[:] = gravity


def test_random_targets_numerically_stable(model):
  """Random joint targets within limits on the ground for 5 s: no blow-up."""
  m = model
  rng = np.random.default_rng(0)
  lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
  d = nominal_data(m)
  max_pen = 0.0
  for k in range(round(5.0 / m.opt.timestep)):
    if k % 10 == 0:  # new targets at the 50 Hz policy rate
      d.ctrl[:] = rng.uniform(lo, hi)
    mujoco.mj_step(m, d)
    if d.ncon:
      max_pen = max(max_pen, -float(d.contact.dist[: d.ncon].min()))
  assert_no_sim_warnings(d)
  assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
  assert np.abs(d.qvel[6:]).max() < 2 * robot.NO_LOAD_SPEED
  # Transient impacts under these violent targets; ~2 mm measured (0.35 mm when
  # standing).
  assert max_pen < 0.003
