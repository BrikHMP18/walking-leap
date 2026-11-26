"""LEAP Hand V1 (right) as a floating-base mjlab entity.

The MJCF in ``assets/leap_hand`` is an unmodified copy of MuJoCo Menagerie's
``leap_hand/right_hand.xml``. Every change needed for locomotion is applied here,
in code, so the upstream file stays byte-identical:

1. A free joint on ``palm`` (the only floating-base joint; nothing else attaches
   the hand to the world).
2. The 16 XML ``<position>`` actuators (kp=3, no torque limit) are deleted and
   replaced by mjlab-built DC-motor position servos (``LEAP_ACTUATOR_CFG``), so
   gains, torque and speed limits are explicit and exist exactly once. The
   actuator config also sets joint armature (0 -> 0.001) and passive damping
   (0.03 -> 0); see the parameter table below.
3. Collision geoms get ``priority=1`` so the hand's own friction and contact
   parameters (not the ground's) govern hand-ground contacts.
4. The palm starts palm-down at the nominal height, and the XML ``<option>``
   is reset (simulation options live in ``env_cfg.py``).
5. Optionally (``as_built=True``) the hand as it is being built: four TPU foot
   socks (``feet=False`` leaves them off) and the IMU on its mount (see "Add-on
   parts"), and the thumb limits of the LEAP authors' own URDF, which Menagerie's
   file has shifted by one joint.
6. Optionally (``torsion``) torsional friction on the hand's contacts, fitted on
   the real hand; it needs condim 4 (``collision_cfg``).

Frames. With the palm at identity orientation the palm frame is a conventional
base frame: +x points from the wrist toward the fingertips, +y toward the thumb
side, +z out of the back of the hand. Palm-down walking therefore uses identity
root orientation, the robot's forward direction is +x of the palm, and mjlab's
body-frame quantities (``root_link_lin_vel_b``, ``projected_gravity_b``) need no
extra rotation.
"""

import math
from dataclasses import replace
from functools import partial
from pathlib import Path

import mujoco
from mjlab.actuator import (
  BuiltinDcMotorActuatorCfg,
  DcMotorDatasheetParams,
  DcMotorInputMode,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

LEAP_HAND_XML: Path = Path(__file__).parent / "assets" / "leap_hand" / "right_hand.xml"

# Menagerie revision the asset was copied from (last commit touching leap_hand/).
MENAGERIE_COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"

ROOT_BODY = "palm"

# Joint order as defined in the MJCF (and therefore in mjlab's action vector).
# NOTE: the LEAP hardware API orders each finger (side, forward, pip, dip), i.e.
# rot before mcp. That remapping belongs to hardware deployment, not here.
JOINT_NAMES: tuple[str, ...] = (
  "if_mcp", "if_rot", "if_pip", "if_dip",
  "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
  "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
  "th_cmc", "th_axl", "th_mcp", "th_ipl",
)  # fmt: skip

FINGER_ROOT_BODIES: tuple[str, ...] = ("if_bs", "mf_bs", "rf_bs", "th_mp")
FINGERTIP_BODIES: tuple[str, ...] = ("if_ds", "mf_ds", "rf_ds", "th_ds")
FINGERTIP_GEOMS: tuple[str, ...] = ("if_tip", "mf_tip", "rf_tip", "th_tip")


def get_spec(
  as_built: bool = False,
  feet: bool = True,
  torsion: tuple[float, float] | None = None,
) -> mujoco.MjSpec:
  """The hand's spec. ``as_built`` adds the IMU, LEAP's thumb limits and, with
  ``feet``, the TPU socks; ``torsion`` = (fingertips, other contacts) torsional
  friction in m (only acts with condim >= 4, see ``collision_cfg``)."""
  spec = mujoco.MjSpec.from_file(str(LEAP_HAND_XML))
  palm = spec.body(ROOT_BODY)
  socks = as_built and feet
  # The menagerie places the palm upside down at z=0.1. Resets use INIT_STATE,
  # but qpos0 (palm pose here, joints at 0) is what mjlab uses to size contact
  # buffers, so keep it palm-down and clear of the ground.
  palm.pos = (0.0, 0.0, INIT_ROOT_HEIGHT_AS_BUILT if socks else INIT_ROOT_HEIGHT)
  palm.quat = (1.0, 0.0, 0.0, 0.0)
  palm.add_freejoint(name="floating_base")
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  # Physics options are owned by env_cfg's MujocoCfg; mjlab drops entity
  # <option>s, and leaving the XML's (cone, impratio, integrator) in place only
  # triggers attach-conflict warnings.
  default = mujoco.MjSpec().option
  spec.option.cone = default.cone
  spec.option.impratio = default.impratio
  spec.option.integrator = default.integrator
  if as_built:
    for joint, limits in THUMB_LIMITS.items():
      spec.joint(joint).range = limits
    if feet:
      add_feet(spec)
    add_imu(spec)
  if torsion is not None:
    set_torsional_friction(spec, *torsion)
  return spec


# Torsional friction (m): how much a contact resists spinning about its normal.
# Fitted on the real hand by replaying its scripted runs in MuJoCo with condim=4
# (docs/next_training.md, "Torsional friction: the test"): 0.003 m at the
# fingertips and 0.03 m on every other contact halve the heading error per clip
# (10 to 5 degrees) and make the crawl turn the way the real one does (M: fit
# against the real IMU, five runs). Without it a fingertip spins for free.
TORSION: tuple[float, float] = (0.003, 0.03)


def set_torsional_friction(spec: mujoco.MjSpec, tip: float, other: float) -> None:
  """Set friction[1] (m) of the fingertips/feet and of every other collision geom.

  Sliding (friction[0]) and rolling friction stay as the XML and the parts set them.
  """
  for geom in spec.geoms:
    if geom.name.endswith(("_tip", "_foot")):
      value = tip
    elif "_collision" in geom.name:
      value = other
    else:
      continue
    friction = list(geom.friction)
    friction[1] = value
    geom.friction = friction


# Thumb limits (rad) of the LEAP authors' URDF and API (`LEAPsim_limits`). Menagerie
# gives th_axl the range of th_cmc and th_mcp the range of th_axl; the URDF's joint
# placements match Menagerie's body for body, so only the ranges are shifted.
THUMB_LIMITS: dict[str, tuple[float, float]] = {
  "th_axl": (-0.47, 2.443),
  "th_mcp": (-1.20, 1.90),
}


##
# Add-on parts: what goes on the real hand on top of the stock LEAP Hand. None of
# these numbers is measured yet (C = from the parts' CAD, D = datasheet, A = assumed).
#
# Each part is a child body without a joint: the hand's bodies carry explicit
# inertials in the XML, so a mass put on one of their geoms would be ignored.
##

PARTS_DIR: Path = Path(__file__).parent / "assets" / "parts"

# Feet: a thin TPU 95A sock over each stock fingertip, 26.5 x 23.5 mm on the tip and
# 2.2 mm longer than it. One 64-point convex mesh per foot, already in the fingertip
# body's frame and within 0.12 mm (mean) of the printed shape.
FOOT_GEOMS: tuple[str, ...] = ("if_foot", "mf_foot", "rf_foot", "th_foot")
# Origin of each foot's body, in its fingertip body's frame (m): the middle of the
# sock's rounded end, so a foot's position is the point it stands and rolls on.
FOOT_CENTER: dict[str, tuple[float, float, float]] = {
  "if_foot": (-0.0017, -0.0410, 0.0144),
  "mf_foot": (-0.0017, -0.0410, 0.0144),
  "rf_foot": (-0.0017, -0.0410, 0.0144),
  "th_foot": (-0.0017, -0.0537, -0.0144),
}
FOOT_MASS = 0.0048  # kg each, C: solid TPU.
FOOT_FRICTION = 0.6  # A: no measured value for printed TPU 95A; expect 0.4 to 0.9.
# A: 1.3-2.5 mm of 95A over a hard tip is close to rigid (0.03 mm under 1.9 N). The
# XML's default solref (1e-4) would make the sock a steel-hard contact.
FOOT_SOLREF = (0.006, 1.0)
FOOT_SOLIMP = (0.95, 0.95, 0.001, 0.5, 2.0)
FOOT_RGBA = (0.95, 0.45, 0.10, 1.0)


def add_feet(spec: mujoco.MjSpec) -> None:
  """Slip one sock on each fingertip (bodies and geoms named ``<digit>_foot``)."""
  for kind in ("finger", "thumb"):
    spec.add_mesh(
      name=f"foot_sock_{kind}", file=str(PARTS_DIR / f"foot_sock_{kind}.obj")
    )
  for name in FOOT_GEOMS:
    digit = name[:2]
    kind = "thumb" if digit == "th" else "finger"
    center = FOOT_CENTER[name]
    foot = spec.body(f"{digit}_ds").add_body(name=name, pos=center)
    foot.add_site(name=name, group=5)
    foot.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=f"foot_sock_{kind}",
      pos=[-c for c in center],
      mass=FOOT_MASS,
      friction=(FOOT_FRICTION, 0.005, 0.0001),
      solref=FOOT_SOLREF,
      solimp=FOOT_SOLIMP,
      rgba=FOOT_RGBA,
    )


# IMU: a SYD TransducerM TM171 on a thin printed pad, held down by a printed cover on
# three M6 bolts of the back lid. Positions in the palm frame, from the LEAP lid's 35 mm
# bolt square: its centre on the lid's outer face (C).
LID_CENTER = (-0.05760, -0.03374, 0.01128)
IMU_PAD_T = 0.0024  # C, thickness of the pad between lid and board.
IMU_BOARD_CENTER = (0.0, -0.0012)  # C, from LID_CENTER (forward, thumb side).
IMU_BOARD_SIZE = (0.034, 0.040, 0.0016)  # D, with the board turned as mounted.
IMU_CASE_SIZE = (0.030, 0.026, 0.011)  # D, the aluminium case on the board.
IMU_COVER_SIZE = (0.0348, 0.0409, 0.0174)  # C, cover over board and pins, from the lid.
IMU_COVER_CENTER = (0.0, -0.00075)  # C, from LID_CENTER: the pin end is longer.
IMU_MASS = 0.019  # D.
IMU_COVER_MASS = 0.013  # C: PETG, 11 g (2 walls) to 14 g (solid).
IMU_PAD_MASS = 0.004  # C: PETG.
IMU_BOLTS = ((0.0, 0.025), (-0.025, 0.0), (0.0315, 0.0))  # C, from LID_CENTER.
IMU_BOLT_MASS = 0.0053  # A: one M6 x 12 flat-head bolt and its nut.
# The USB-C plug in the board: it leaves on the ring side. Its cable is not modelled;
# its pull is a random force (motion_cfg.py).
IMU_PLUG_POS = (0.0062, -0.0342, 0.0057)  # C, from LID_CENTER.
IMU_PLUG_SIZE = (0.013, 0.026, 0.007)  # A: the plug's body.
IMU_PLUG_MASS = 0.006  # A.
# The board is turned a quarter left (seen from above): its pins (its +x) point to the
# thumb side (palm +y) and its USB-C leaves on the ring side. Columns of R(quat) are
# the IMU's axes in the palm frame: palm x = -imu y, palm y = +imu x, palm z = imu z.
IMU_QUAT = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))


def add_imu(spec: mujoco.MjSpec) -> None:
  """Hold the IMU under its cover on the back of the palm, with its two sensors.

  ``imu_ang_vel`` (rad/s) and ``imu_upvector`` (world z) are in the IMU's own axes,
  like the real board's gyro and gravity outputs.
  """
  for part in ("cover", "pad"):
    spec.add_mesh(
      name=f"imu_{part}",
      file=str(PARTS_DIR / f"imu_{part}.stl"),  # mm, origin at LID_CENTER.
      scale=(1e-3, 1e-3, 1e-3),
      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_EXACT,
    )
  mount = spec.body(ROOT_BODY).add_body(name="imu_mount", pos=LID_CENTER)
  bx, by = IMU_BOARD_CENTER
  hx, hy, hz = (0.5 * size for size in IMU_BOARD_SIZE)
  cx, cy, cz = (0.5 * size for size in IMU_CASE_SIZE)
  t = IMU_PAD_T
  box, cylinder = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
  board_z = t + hz
  case_z = t + 2 * hz + cz
  # Mass carriers, also what is drawn. The case holds nearly all of the IMU's mass.
  visual = dict(contype=0, conaffinity=0, group=1)
  for part, mass, rgba in (
    ("cover", IMU_COVER_MASS, (0.20, 0.35, 0.65, 1.0)),
    ("pad", IMU_PAD_MASS, (0.16, 0.16, 0.18, 1.0)),
  ):
    mount.add_geom(
      name=f"imu_{part}",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=f"imu_{part}",
      mass=mass,
      rgba=rgba,
      **visual,
    )
  mount.add_geom(
    name="imu_board",
    type=box,
    pos=(bx, by, board_z),
    size=(hx, hy, hz),
    mass=0.2 * IMU_MASS,
    rgba=(0.05, 0.35, 0.15, 1.0),
    **visual,
  )
  mount.add_geom(
    name="imu_case",
    type=box,
    pos=(bx, by + 0.0006, case_z),
    size=(cx, cy, cz),
    mass=0.8 * IMU_MASS,
    rgba=(0.75, 0.76, 0.78, 1.0),
    **visual,
  )
  mount.add_geom(
    name="imu_plug",
    type=box,
    pos=IMU_PLUG_POS,
    size=[0.5 * size for size in IMU_PLUG_SIZE],
    mass=IMU_PLUG_MASS,
    rgba=(0.10, 0.10, 0.10, 1.0),
    **visual,
  )
  for i, (x, y) in enumerate(IMU_BOLTS):
    mount.add_geom(
      name=f"imu_bolt_{i}",
      type=cylinder,
      pos=(x, y, 0.0066),
      size=(0.0050, 0.0026, 0.0),
      mass=IMU_BOLT_MASS,
      rgba=(0.55, 0.56, 0.58, 1.0),
      **visual,
    )
  # One box for contacts (the thumb can reach behind the palm; falls on the back). It
  # leaves out the three low bosses and the plug.
  mount.add_geom(
    name="imu_collision",
    type=box,
    pos=(*IMU_COVER_CENTER, 0.5 * IMU_COVER_SIZE[2]),
    size=[0.5 * size for size in IMU_COVER_SIZE],
    mass=0.0,
    group=3,
  )
  mount.add_site(name="imu", pos=(bx, by, case_z), quat=IMU_QUAT, group=5)
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  spec.add_sensor(
    name="imu_upvector",
    type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="world",
    reftype=mujoco.mjtObj.mjOBJ_SITE,
    refname="imu",
  )


##
# Actuators: 16 Dynamixel XC330-M288-T servos in (current-limited) position mode,
# modelled with MuJoCo's native DC motor in position mode:
#   V   = clamp(kp * (q_target - q) - kd * qdot, +-SUPPLY_VOLTAGE)
#   tau = clamp(K * (V - K * qdot) / R, +-EFFORT_LIMIT)
# K and R follow from the datasheet triple (V, stall torque, no-load speed), so
# the torque-speed line is the real one: an unloaded joint tops out at 8.48 rad/s
# (a plain PD with the same torque cap reached 13-16 rad/s), and back-EMF adds
# K^2/R = 0.11 N*m*s/rad of damping.
#
# Sources (D = datasheet/documented, S = LEAP authors' simulator, A = assumed):
#   5.0 V, 0.93 N*m stall, 81 rpm   D  ROBOTIS XC330-M288-T e-manual, at 5 V.
#   stiffness 3.0 N*m/rad (small   S  LEAP_Hand_Sim pgain=3 and Menagerie kp=3.
#     signal, at zero velocity)       Close to the API's Dynamixel KPP=600
#                                     converted through the motor model
#                                     (~3.2 N*m/rad; README).
#   kd = 0 V*s/rad                  A  Damping comes from back-EMF (0.11), which
#                                     already matches LEAP_Hand_Sim dgain=0.1.
#   effort limit 0.5 N*m            S  LEAP_Hand_Sim effort=0.5. The real cap is
#                                     the servo current limit (API default
#                                     350 mA, up to 550 mA on the full hand, D);
#                                     its torque equivalent is unmeasured.
#   armature 0.001 kg*m^2           S  LEAP_Hand_Sim (reflected rotor inertia).
#   joint damping 0 (XML: 0.03)     D  The datasheet torque-speed line is measured
#                                     at the output shaft, so gearbox losses are
#                                     already in it; the XML's passive damping
#                                     would cap speed at 6.6 instead of 8.5 rad/s.
#   joint frictionloss 0.001: Menagerie XML value, kept (A).
##

SUPPLY_VOLTAGE = 5.0
STALL_TORQUE = 0.93
NO_LOAD_SPEED = 81.0 * 2.0 * math.pi / 60.0  # 8.48 rad/s
STIFFNESS = 3.0
EFFORT_LIMIT = 0.5
ARMATURE = 0.001
# Torque at the LEAP Hand API's default current limit, 350 mA: the datasheet's stall
# point (0.93 N*m at 1.8 A) scaled by current. Estimated, not measured (D/A).
STALL_CURRENT = 1.8
API_CURRENT_LIMIT = 0.350
API_EFFORT_LIMIT = STALL_TORQUE * API_CURRENT_LIMIT / STALL_CURRENT  # 0.181 N*m
# Torque per volt at zero speed, K/R = stall_torque / voltage (N*m/V).
TORQUE_PER_VOLT = STALL_TORQUE / SUPPLY_VOLTAGE
# Stiffness the real joints follow their commands with, at the API's gains 1200 / 0 /
# 600: replaying the real runs, 2.2-3.0 N*m/rad fit and 2.6 fits best; 1.6 is ruled
# out (M: fit against the real encoders, docs/real_hand.md, docs/next_training.md).
MEASURED_STIFFNESS = 2.6


def actuator_cfg(
  effort_limit: float = EFFORT_LIMIT, stiffness: float = STIFFNESS
) -> BuiltinDcMotorActuatorCfg:
  """One XC330 position servo per joint: torque capped at ``effort_limit`` (N*m),
  small-signal ``stiffness`` in N*m/rad."""
  return BuiltinDcMotorActuatorCfg(
    # Joints only: ".*" would also match the IMU's site.
    target_names_expr=(r".*_(mcp|rot|pip|dip|cmc|axl|ipl)",),
    motor_params=DcMotorDatasheetParams(
      nominal_voltage=SUPPLY_VOLTAGE,
      stall_torque=STALL_TORQUE,
      no_load_speed=NO_LOAD_SPEED,
    ),
    mode=DcMotorInputMode.POSITION,
    stiffness=stiffness / TORQUE_PER_VOLT,  # V/rad
    damping=0.0,
    voltage_limit=SUPPLY_VOLTAGE,
    effort_limit=effort_limit,
    armature=ARMATURE,
    viscous_damping=0.0,
  )


LEAP_ACTUATOR_CFG = actuator_cfg()

##
# Nominal pose (see README "Nominal pose"): palm down and level, the index,
# middle and ring fingers flexed down in front (tips at palm-frame x=+0.017 m)
# and the thumb swung down behind (tip at x=-0.129 m), so the hand stands on four
# digit tips. Chosen by a kinematic search maximizing the support-polygon margin
# of the CoM (29 mm) with every joint >= 0.25 rad from its limits, then checked
# in simulation: it stands with 0.04 N*m peak joint torque and 110 mm of palm
# clearance. Fingers are not treated as symmetric legs; the thumb is the only
# rear support. With these angles the lowest tip is 0.1458 m below the palm
# origin; INIT_ROOT_HEIGHT leaves ~1 mm of clearance.
##

INIT_JOINT_POS: dict[str, float] = {
  r"(if|mf|rf)_mcp": 0.9,
  r"(if|mf|rf)_rot": 0.0,
  r"(if|mf|rf)_pip": 0.6,
  r"(if|mf|rf)_dip": 0.3,
  "th_cmc": 1.75,
  "th_axl": 0.0,
  "th_mcp": -0.15,
  "th_ipl": -0.7,
}
INIT_ROOT_HEIGHT = 0.147
INIT_ROOT_HEIGHT_AS_BUILT = 0.149  # With the socks: the lowest foot is 0.1480 m below.

# Folded-thumb stance, as-built hand only: the thumb folds under the palm the other way
# round (base turned ~110 deg), feet at x=+0.020 m (fingers) and x=-0.115 m (thumb).
# With the feet planted the palm has -60/+27 mm of height and +-41 deg of yaw here,
# against -18/+6 mm and -17/+34 deg in the pose above, at the same joint torque
# (0.06 N*m peak). Solved by clips.py's planted-feet kinematics.
FOLDED_JOINT_POS: dict[str, float] = {
  r"(if|mf|rf)_mcp": 0.563,
  r"(if|mf|rf)_rot": 0.0,
  r"(if|mf|rf)_pip": 0.889,
  r"(if|mf|rf)_dip": 0.635,
  "th_cmc": 0.243,
  "th_axl": 1.912,
  "th_mcp": 1.460,
  "th_ipl": 0.307,
}
FOLDED_ROOT_HEIGHT = 0.131
# The same stance on the stock fingertips (no socks): the lowest point is 0.1280 m
# below the palm (index tip), so ~1 mm of clearance. The real hand stands at about
# 128 mm in it (docs/real_hand.md).
FOLDED_ROOT_HEIGHT_STOCK = 0.129
# Where the four feet (FOOT_GEOMS order) stand in that stance: x, y in the palm frame, m.
FOLDED_FEET: tuple[tuple[float, float], ...] = (
  (0.020, 0.0082),
  (0.020, -0.0372),
  (0.020, -0.0826),
  (-0.115, 0.0),
)

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, INIT_ROOT_HEIGHT),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos=INIT_JOINT_POS,
  joint_vel={".*": 0.0},
)

##
# Collisions: every collision primitive of the palm and the four digits can touch
# the ground and each other (Menagerie's <contact><exclude> filters are kept).
# friction / solref / solimp are inherited from the XML (links 0.2, tips 0.5).
##


def collision_cfg(condim: int = 3) -> CollisionCfg:
  """The hand's contacts; ``condim=4`` adds torsional friction (see ``TORSION``)."""
  return CollisionCfg(
    geom_names_expr=(r".*_collision.*", r".*_tip$", r".*_foot$"),
    contype=1,
    conaffinity=1,
    condim=condim,
    priority=1,
  )


COLLISION_CFG = collision_cfg()


def joint_limits(as_built: bool = False) -> dict[str, tuple[float, float]]:
  """Joint position limits (rad): the MJCF's, or with LEAP's own thumb limits."""
  spec = mujoco.MjSpec.from_file(str(LEAP_HAND_XML))
  limits = {j.name: (float(j.range[0]), float(j.range[1])) for j in spec.joints}
  return {**limits, **THUMB_LIMITS} if as_built else limits


def get_leap_hand_cfg(
  as_built: bool = False,
  folded_thumb: bool = False,
  effort_limit: float = EFFORT_LIMIT,
  feet: bool = True,
  stiffness: float = STIFFNESS,
  torsion: tuple[float, float] | None = None,
) -> EntityCfg:
  """Return a fresh LEAP Hand entity config (fresh to avoid shared mutation).

  ``as_built`` adds the IMU, LEAP's own thumb limits and, with ``feet``, the TPU
  socks (``feet=False``: the stock fingertips the real hand runs on now);
  ``folded_thumb`` (as-built only) starts in that stance; ``effort_limit`` is the
  torque cap (N*m), ``stiffness`` the servos' (N*m/rad); ``torsion`` (m, fingertips
  and other contacts) turns on torsional friction with condim 4.
  """
  socks = as_built and feet
  init_state = INIT_STATE
  if folded_thumb:
    assert as_built, "the folded-thumb stance needs LEAP's own thumb limits"
    height = FOLDED_ROOT_HEIGHT if socks else FOLDED_ROOT_HEIGHT_STOCK
    init_state = replace(INIT_STATE, pos=(0.0, 0.0, height), joint_pos=FOLDED_JOINT_POS)
  elif socks:
    init_state = replace(INIT_STATE, pos=(0.0, 0.0, INIT_ROOT_HEIGHT_AS_BUILT))
  return EntityCfg(
    init_state=init_state,
    collisions=(collision_cfg(condim=3 if torsion is None else 4),),
    spec_fn=partial(get_spec, as_built=as_built, feet=feet, torsion=torsion),
    articulation=EntityArticulationInfoCfg(
      actuators=(actuator_cfg(effort_limit, stiffness),)
    ),
  )
