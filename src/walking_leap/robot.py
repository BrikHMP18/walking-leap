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

Frames. With the palm at identity orientation the palm frame is a conventional
base frame: +x points from the wrist toward the fingertips, +y toward the thumb
side, +z out of the back of the hand. Palm-down walking therefore uses identity
root orientation, the robot's forward direction is +x of the palm, and mjlab's
body-frame quantities (``root_link_lin_vel_b``, ``projected_gravity_b``) need no
extra rotation.
"""

import math
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
FINGERTIP_GEOMS: tuple[str, ...] = ("if_tip", "mf_tip", "rf_tip", "th_tip")


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(LEAP_HAND_XML))
  palm = spec.body(ROOT_BODY)
  # The menagerie places the palm upside down at z=0.1. Resets use INIT_STATE,
  # but qpos0 (palm pose here, joints at 0) is what mjlab uses to size contact
  # buffers, so keep it palm-down and clear of the ground.
  palm.pos = (0.0, 0.0, INIT_ROOT_HEIGHT)
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
  return spec


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
# Torque per volt at zero speed, K/R = stall_torque / voltage (N*m/V).
TORQUE_PER_VOLT = STALL_TORQUE / SUPPLY_VOLTAGE

LEAP_ACTUATOR_CFG = BuiltinDcMotorActuatorCfg(
  target_names_expr=(".*",),
  motor_params=DcMotorDatasheetParams(
    nominal_voltage=SUPPLY_VOLTAGE,
    stall_torque=STALL_TORQUE,
    no_load_speed=NO_LOAD_SPEED,
  ),
  mode=DcMotorInputMode.POSITION,
  stiffness=STIFFNESS / TORQUE_PER_VOLT,  # V/rad
  damping=0.0,
  voltage_limit=SUPPLY_VOLTAGE,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE,
  viscous_damping=0.0,
)

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

COLLISION_CFG = CollisionCfg(
  geom_names_expr=(r".*_collision.*", r".*_tip$"),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
)


def joint_limits() -> dict[str, tuple[float, float]]:
  """Joint position limits (rad) as defined in the MJCF."""
  spec = mujoco.MjSpec.from_file(str(LEAP_HAND_XML))
  return {j.name: (float(j.range[0]), float(j.range[1])) for j in spec.joints}


def get_leap_hand_cfg() -> EntityCfg:
  """Return a fresh LEAP Hand entity config (fresh to avoid shared mutation)."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(COLLISION_CFG,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(actuators=(LEAP_ACTUATOR_CFG,)),
  )
