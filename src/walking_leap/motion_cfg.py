"""Motion imitation on flat ground: the as-built hand plays a reference clip.

mjlab's motion-tracking terms (a re-implementation of BeyondMimic), with two changes
for this hand: every scale is the hand's (mm and cm, not a humanoid's), and the actor
only sees what the hand can measure, the encoders and the IMU on its back. It never
sees where it is in the world, nor its heading. Clips are written in ``clips.py``.
"""

from dataclasses import fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from walking_leap import mdp
from walking_leap.clips import CLIPS_DIR
from walking_leap.env_cfg import ACTION_SCALE, IMU_OBSERVATIONS, PALM_CONTACT
from walking_leap.robot import (
  API_EFFORT_LIMIT,
  FINGERTIP_BODIES,
  FINGERTIP_GEOMS,
  FOOT_GEOMS,
  MEASURED_STIFFNESS,
  ROOT_BODY,
  STIFFNESS,
  TORSION,
  get_leap_hand_cfg,
  joint_limits,
)

# Bodies whose pose is imitated: the palm (first: it is the root), the two last links
# of every digit and the feet.
TRACKED_BODIES: tuple[str, ...] = (
  ROOT_BODY,
  "if_md", "if_ds", "if_foot",
  "mf_md", "mf_ds", "mf_foot",
  "rf_md", "rf_ds", "rf_foot",
  "th_px", "th_ds", "th_foot",
)  # fmt: skip
SELF_CONTACT = "self_contact"

# Kicks of the robust variant (instant change of the palm's velocity, m/s and rad/s).
PUSH_RANGE = {
  "x": (-0.10, 0.10),
  "y": (-0.10, 0.10),
  "z": (-0.05, 0.05),
  "roll": (-0.6, 0.6),
  "pitch": (-0.6, 0.6),
  "yaw": (-0.6, 0.6),
}


# Steady pull of the cables, per axis and per body: the IMU's USB on its mount, the
# hand's USB and power on the palm. Guessed: about 0.2 N in all, 2 % of the weight.
CABLE_BODIES = ("imu_mount", ROOT_BODY)
CABLE_FORCE = 0.15  # N
CABLE_TORQUE = 0.01  # N*m


def walking_leap_motion_env_cfg(
  play: bool = False,
  robust: bool = False,
  cables: bool = False,
  clip: str = "hello",
  feet: bool = True,
  imu: bool = True,
  stiffness: float = STIFFNESS,
  torsion: tuple[float, float] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Imitate ``clip``. ``robust`` adds kicks, sensor noise, delay and random numbers
  for what nobody has measured (the feet's grip, the balance, the motors); ``cables``
  adds a small steady pull that changes every few seconds.

  ``feet=False`` is the hand on its stock fingertips (no socks): the digits' last
  links stand in for the feet. ``imu=False`` removes the palm-IMU terms from the
  actor (the critic keeps them). ``stiffness`` (N*m/rad) and ``torsion`` (m) are the
  hand's, see ``robot.get_leap_hand_cfg``."""

  # Tracked bodies and the bodies whose height ends an episode: the feet, or without
  # socks the digits' last links (already tracked).
  tracked = (
    TRACKED_BODIES
    if feet
    else tuple(b for b in TRACKED_BODIES if not b.endswith("_foot"))
  )
  stand_on = FOOT_GEOMS if feet else FINGERTIP_BODIES

  ##
  # Sensors.
  ##

  palm_contact = ContactSensorCfg(
    name=PALM_CONTACT,
    primary=ContactMatch(mode="body", pattern=ROOT_BODY, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    track_air_time=True,
  )
  self_contact = ContactSensorCfg(
    name=SELF_CONTACT,
    primary=ContactMatch(mode="subtree", pattern=ROOT_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=ROOT_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=10,
  )

  ##
  # Observations. Actor: the clip's joints and tilt now, encoders, IMU, last action.
  ##

  motion = {"command_name": "motion"}

  def noisy(limit: float) -> Unoise | None:
    return Unoise(n_min=-limit, n_max=limit) if robust else None

  actor_terms = {
    "reference_joints": ObservationTermCfg(
      func=envs_mdp.generated_commands, params=motion
    ),
    "reference_gravity": ObservationTermCfg(func=mdp.motion_gravity, params=motion),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel, noise=noisy(0.05)),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity, noise=noisy(0.03)
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel, params={"biased": True}, noise=noisy(0.01)
    ),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, noise=noisy(0.3)),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  critic_terms = {
    **{
      k: ObservationTermCfg(func=t.func, params=t.params)
      for k, t in actor_terms.items()
    },
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "anchor_pos": ObservationTermCfg(func=track_mdp.motion_anchor_pos_b, params=motion),
    "anchor_ori": ObservationTermCfg(func=track_mdp.motion_anchor_ori_b, params=motion),
    "body_pos": ObservationTermCfg(func=track_mdp.robot_body_pos_b, params=motion),
    "body_ori": ObservationTermCfg(func=track_mdp.robot_body_ori_b, params=motion),
  }
  if not imu:
    for name in IMU_OBSERVATIONS:
      del actor_terms[name]
  observations = {
    "actor": ObservationGroupCfg(terms=actor_terms, enable_corruption=robust),
    "critic": ObservationGroupCfg(terms=critic_terms, enable_corruption=False),
  }

  ##
  # Actions: 16 joint targets around the folded-thumb stance, inside the joint limits.
  ##

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=ACTION_SCALE,
      use_default_offset=True,
      clip=joint_limits(as_built=True),
    )
  }

  ##
  # Command: the clip, from a random frame, near its state there.
  ##

  commands = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      # Off: mjlab draws the clip as a translucent green "ghost" hand over the real
      # one. Drawing only (no physics, training unchanged); set True to see it.
      debug_vis=False,
      motion_file=str(CLIPS_DIR / f"{clip}.npz"),
      anchor_body_name=ROOT_BODY,
      body_names=tracked,
      pose_range={
        "x": (-0.005, 0.005),
        "y": (-0.005, 0.005),
        "z": (0.0, 0.004),
        "roll": (-0.04, 0.04),
        "pitch": (-0.04, 0.04),
        "yaw": (-0.1, 0.1),
      },
      velocity_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.02, 0.02),
        "roll": (-0.2, 0.2),
        "pitch": (-0.2, 0.2),
        "yaw": (-0.2, 0.2),
      },
      joint_position_range=(-0.03, 0.03),
    )
  }

  ##
  # Events: none in the nominal task (resets come from the motion command).
  ##

  events: dict[str, EventTermCfg] = {}
  if robust:
    events = {
      "push": EventTermCfg(
        func=envs_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": PUSH_RANGE},
      ),
      # Not measured yet: the socks' grip, the mass on the back, the motors. Stock
      # tips: 0.5 to 1.0, from the real stand-up (docs/next_training.md).
      "foot_friction": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", geom_names=FOOT_GEOMS if feet else FINGERTIP_GEOMS
          ),
          "operation": "abs",
          "ranges": (0.4, 0.9) if feet else (0.5, 1.0),
          "shared_random": True,
        },
      ),
      "palm_com": EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names=(ROOT_BODY,)),
          "operation": "add",
          "ranges": {0: (-0.008, 0.008), 1: (-0.008, 0.008), 2: (-0.008, 0.008)},
        },
      ),
      "encoder_bias": EventTermCfg(
        mode="startup",
        func=dr.encoder_bias,
        params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.02, 0.02)},
      ),
      "motor_stiffness": EventTermCfg(
        mode="startup",
        func=dr.pd_gains,
        params={"kp_range": (0.6, 1.2), "kd_range": (1.0, 1.0), "operation": "scale"},
      ),
      "motor_torque": EventTermCfg(
        mode="startup",
        func=dr.effort_limits,
        params={"effort_limit_range": (0.8, 1.2), "operation": "scale"},
      ),
    }

  if cables:
    # A wrench stays until the next one: set at reset, changed every 1 to 4 s.
    pull = {
      "asset_cfg": SceneEntityCfg("robot", body_names=CABLE_BODIES),
      "force_range": (-CABLE_FORCE, CABLE_FORCE),
      "torque_range": (-CABLE_TORQUE, CABLE_TORQUE),
    }
    events["cables_at_reset"] = EventTermCfg(
      func=envs_mdp.apply_external_force_torque, mode="reset", params=pull
    )
    events["cables"] = EventTermCfg(
      func=envs_mdp.apply_external_force_torque,
      mode="interval",
      interval_range_s=(1.0, 4.0),
      params=pull,
    )

  ##
  # Rewards (weights are per second). exp(-error^2 / std^2) of each imitated quantity.
  ##

  def imitate(func, weight: float, std: float) -> RewardTermCfg:
    return RewardTermCfg(func=func, weight=weight, params={**motion, "std": std})

  rewards = {
    "palm_position": imitate(
      track_mdp.motion_global_anchor_position_error_exp, 0.5, 0.04
    ),
    "palm_orientation": imitate(
      track_mdp.motion_global_anchor_orientation_error_exp, 0.5, 0.3
    ),
    "body_position": imitate(
      track_mdp.motion_relative_body_position_error_exp, 1.5, 0.02
    ),
    "body_orientation": imitate(
      track_mdp.motion_relative_body_orientation_error_exp, 1.0, 0.3
    ),
    "body_lin_vel": imitate(
      track_mdp.motion_global_body_linear_velocity_error_exp, 1.0, 0.15
    ),
    "body_ang_vel": imitate(
      track_mdp.motion_global_body_angular_velocity_error_exp, 1.0, 2.0
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
    "joint_torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-0.2),
    "joint_limit": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_contact": RewardTermCfg(
      func=track_mdp.self_collision_cost,
      weight=-0.5,
      params={"sensor_name": SELF_CONTACT, "force_threshold": 0.5},
    ),
  }

  ##
  # Terminations: too far from the clip (palm height, tilt, a foot's height).
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "nan_state": TerminationTermCfg(func=envs_mdp.nan_detection),
    "palm_height": TerminationTermCfg(
      func=track_mdp.bad_anchor_pos_z_only, params={**motion, "threshold": 0.03}
    ),
    "palm_tilt": TerminationTermCfg(
      func=track_mdp.bad_anchor_ori,
      params={**motion, "asset_cfg": SceneEntityCfg("robot"), "threshold": 0.15},
    ),
    "foot_height": TerminationTermCfg(
      func=track_mdp.bad_motion_body_pos_z_only,
      params={**motion, "threshold": 0.03, "body_names": stand_on},
    ),
  }

  metrics = {
    "palm_ground_contact": MetricsTermCfg(
      func=mdp.contact_active, params={"sensor_name": PALM_CONTACT}
    ),
    "actuator_saturation": MetricsTermCfg(
      func=mdp.actuator_saturation, params={"effort_limit": API_EFFORT_LIMIT}
    ),
    "mechanical_power": MetricsTermCfg(func=mdp.mechanical_power),
  }

  hand = get_leap_hand_cfg(
    as_built=True,
    folded_thumb=True,
    effort_limit=API_EFFORT_LIMIT,
    feet=feet,
    stiffness=stiffness,
    torsion=torsion,
  )
  if robust:  # Targets reach the motors up to 20 ms late.
    assert hand.articulation is not None
    motors = tuple(replace(a, delay_max_lag=10) for a in hand.articulation.actuators)
    hand.articulation = replace(hand.articulation, actuators=motors)

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": hand},
      sensors=(palm_contact, self_contact),
      num_envs=1,
      env_spacing=1.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=ROOT_BODY,
      distance=0.7,
      elevation=-20.0,
      azimuth=135.0,
    ),
    # Same physics as the walking task (env_cfg.py); a few more contacts per world,
    # as the clip bends the digits closer together.
    sim=SimulationCfg(
      nconmax=64,
      njmax=256,
      mujoco=MujocoCfg(
        timestep=0.002, integrator="implicitfast", cone="elliptic", impratio=1.0
      ),
    ),
    decimation=10,
    episode_length_s=10.0,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    for disturbance in ("push", "cables_at_reset", "cables"):
      cfg.events.pop(disturbance, None)
    command = cfg.commands["motion"]
    assert isinstance(command, MotionCommandCfg)
    command.pose_range, command.velocity_range = {}, {}
    command.joint_position_range = (0.0, 0.0)
    command.sampling_mode = "start"

  return cfg


# Episodes of the gait task that start with the hand dropped on its side or back.
FALLEN_START_PROB = 0.2


def walking_leap_gait_env_cfg(
  play: bool = False,
  clip: str = "gait",
  fallen_start_prob: float = FALLEN_START_PROB,
  push: dict[str, tuple[float, float]] | None = None,
  slew: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Imitate the scripted crawl (assets/clips/gait.npz) and survive: the real hand
  with the IMU, everything random of ``robust`` and the cables, and falls. A fall
  no longer ends an episode: the clip waits while the hand is down and resumes where
  it gets up (``mdp.GaitMotionCommand``); a fifth of the episodes start fallen;
  rewards for an upright palm at the clip's height, and for moving as the plan
  intends (``mdp.planned_velocity_exp``), which the slipping replay undershoots.
  Observations are those of ``Walking-Leap-Motion-Gait-Flat-v0``, so training can
  continue from its checkpoints. ``clip``, ``fallen_start_prob``, ``push`` (the pushes'
  velocity range) and ``slew`` (``mdp.SlewJointPositionActionCfg``, the real driver's
  setpoint limit) are for Gait-v2 and Getup; the defaults are Gait-v1."""
  cfg = walking_leap_motion_env_cfg(
    play=play,
    robust=True,
    cables=True,
    clip=clip,
    feet=False,
    stiffness=MEASURED_STIFFNESS,
    torsion=TORSION,
  )
  motion = {"command_name": "motion"}
  old = cfg.commands["motion"]
  cfg.commands["motion"] = mdp.GaitMotionCommandCfg(
    **{f.name: getattr(old, f.name) for f in fields(old) if f.init},
    fallen_start_prob=0.0 if play else fallen_start_prob,
  )
  if push is not None and "push" in cfg.events:
    cfg.events["push"].params["velocity_range"] = push
  if slew:
    old_action = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = mdp.SlewJointPositionActionCfg(
      **{f.name: getattr(old_action, f.name) for f in fields(old_action) if f.init}
    )
  for name in ("palm_height", "palm_tilt", "foot_height"):
    del cfg.terminations[name]
  cfg.terminations["fallen_too_long"] = TerminationTermCfg(
    func=mdp.fallen_too_long, params={**motion, "limit_s": 5.0}
  )
  cfg.rewards["upright"] = RewardTermCfg(func=mdp.palm_upright, weight=1.0)
  cfg.rewards["stand_height"] = RewardTermCfg(
    func=mdp.palm_height_exp, weight=0.5, params={**motion, "std": 0.04}
  )
  cfg.rewards["planned_velocity"] = RewardTermCfg(
    func=mdp.planned_velocity_exp,
    weight=1.0,
    params={**motion, "std_lin": 0.02, "std_yaw": 0.3},
  )
  if not play:
    cfg.episode_length_s = 15.0
  return cfg


# Gait-v2's pushes: a hand holding its stance did not fall from 0.3 m/s; 0.5 m/s felled
# 22 % of them (scripts/eval_survival.py). Training's 0.1 m/s (PUSH_RANGE) did nothing.
STRONG_PUSH_RANGE = {**PUSH_RANGE, "x": (-0.4, 0.4), "y": (-0.4, 0.4)}


def walking_leap_gait_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Gait-v1 on gait_smooth.npz (targets under the real loop's 6 rad/s), with the real
  driver's setpoint limit, pushes up to 0.4 m/s per axis and no fallen starts (getting
  up is Getup's). Same observations as v0/v1: it continues from Gait-v1."""
  return walking_leap_gait_env_cfg(
    play=play,
    clip="gait_smooth",
    fallen_start_prob=0.0,
    push=STRONG_PUSH_RANGE,
    slew=True,
  )


GETUP_EPISODE_S = 6.0


def walking_leap_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Getting up: Gait-v2's hand (randomized, cables, strong pushes, the driver's
  setpoint limit), every episode started fallen (``GaitMotionCommandCfg``: 57-115 deg
  about a random horizontal axis, from 0.16 m). The actor sees the IMU, the encoders and
  its last action, never the clip. Rewarded for being up (the command's own test:
  palm within 17 deg of upright and above 9 cm for 0.3 s), an upright palm and the
  clip's palm height; the clip's tracking rewards are gone (once up, the hand is to
  stand, and the walking teacher takes over in the student)."""
  cfg = walking_leap_gait_env_cfg(
    play=play,
    clip="gait_smooth",
    fallen_start_prob=1.0,
    push=STRONG_PUSH_RANGE,
    slew=True,
  )
  for name in ("reference_joints", "reference_gravity"):
    del cfg.observations["actor"].terms[name]
  for name in (
    "palm_position",
    "palm_orientation",
    "body_position",
    "body_orientation",
    "body_lin_vel",
    "body_ang_vel",
    "planned_velocity",
  ):
    del cfg.rewards[name]
  motion = {"command_name": "motion"}
  cfg.rewards["up"] = RewardTermCfg(func=mdp.not_fallen, weight=2.0, params=motion)
  cfg.rewards["stand_height"].weight = 1.0
  if play:
    cfg.commands["motion"].fallen_start_prob = 1.0
  else:
    cfg.episode_length_s = GETUP_EPISODE_S
  return cfg


STUDENT_HISTORY = 5  # steps (0.1 s): without the clip, the student tells the phase


def walking_leap_gait_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """``Walking-Leap-Motion-Gait-Flat-v1`` for a teacher-student distillation. The
  teacher (a v1 checkpoint) keeps its inputs, the clip included. The student sees
  only what the real hand has with a keyboard: the plan's velocity as its command
  (``mdp.planned_command``), the IMU, the encoders and its last action, over the last
  ``STUDENT_HISTORY`` steps, with the teacher's sensor noise. The rewards are only
  logged (distillation does not use them)."""
  cfg = walking_leap_gait_env_cfg(play=play)
  teacher = cfg.observations.pop("actor")
  del cfg.observations["critic"]
  student_terms = {
    "command": ObservationTermCfg(
      func=mdp.planned_command, params={"command_name": "motion"}
    ),
    **{
      name: replace(term)
      for name, term in teacher.terms.items()
      if not name.startswith("reference_")
    },
  }
  cfg.observations = {
    "student": ObservationGroupCfg(
      terms=student_terms,
      enable_corruption=teacher.enable_corruption,
      history_length=STUDENT_HISTORY,
    ),
    "teacher": teacher,
  }
  return cfg


TWO_TEACHER_FALLEN_PROB = 0.3


def walking_leap_two_teacher_student_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """The keyboard-velocity student of two teachers (docs/velocity_policy.md, stage 2):
  Gait-v2's hand (smooth clip, the driver's setpoint limit, strong pushes) with
  ``TWO_TEACHER_FALLEN_PROB`` of the episodes started fallen. Groups: ``student`` (as
  Student-v0: the plan's velocity, IMU, encoders, last action, 5 steps), ``teacher``
  (Gait-v2's actor), ``teacher_getup`` (Getup's actor) and ``fallen`` (which teacher
  answers: runner.SwitchTeacher)."""
  cfg = walking_leap_gait_env_cfg(
    play=play,
    clip="gait_smooth",
    fallen_start_prob=TWO_TEACHER_FALLEN_PROB,
    push=STRONG_PUSH_RANGE,
    slew=True,
  )
  teacher = cfg.observations.pop("actor")
  del cfg.observations["critic"]
  motion = {"command_name": "motion"}
  proprio = {
    name: replace(term)
    for name, term in teacher.terms.items()
    if not name.startswith("reference_")
  }
  cfg.observations = {
    "student": ObservationGroupCfg(
      terms={
        "command": ObservationTermCfg(func=mdp.planned_command, params=motion),
        **{name: replace(term) for name, term in proprio.items()},
      },
      enable_corruption=teacher.enable_corruption,
      history_length=STUDENT_HISTORY,
    ),
    "teacher": teacher,
    "teacher_getup": ObservationGroupCfg(
      terms=proprio, enable_corruption=teacher.enable_corruption
    ),
    "fallen": ObservationGroupCfg(
      terms={"fallen": ObservationTermCfg(func=mdp.fallen_flag, params=motion)},
      enable_corruption=False,
    ),
  }
  return cfg


# Stage 3: keyboard commands as the student will get them. The gait clip's commands
# per cycle (gait.npz): forward 28, back 20, left 15, right 8 mm/s, turns 19 deg/s.
KEYS_LIN_VEL_X = (-0.02, 0.03)  # m/s
KEYS_LIN_VEL_Y = (-0.01, 0.015)  # m/s
KEYS_ANG_VEL_Z = (-0.33, 0.33)  # rad/s
KEYS_STANDING = 0.15  # share of commands that stand still
KEYS_RESAMPLING_S = (0.5, 3.0)  # a key pressed or released at any phase of a cycle


def walking_leap_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The student's own task (docs/velocity_policy.md, stage 3), for PPO continuing from
  the two-teacher student: random velocity commands over the keys' range, combined,
  changed every 0.5-3 s, standing still 15 % of the time; the actor sees exactly the
  student's inputs (the command zero while fallen). Rewards for the commanded velocity,
  being up, an upright palm and the stance's height; the clip only times falls and
  fallen starts (``TWO_TEACHER_FALLEN_PROB``), no tracking reward."""
  cfg = walking_leap_two_teacher_student_env_cfg(play=play)
  student = cfg.observations["student"]
  critic_terms = {
    name: replace(term) for name, term in cfg.observations["teacher"].terms.items()
  }
  student.terms["command"] = ObservationTermCfg(
    func=mdp.velocity_command_unless_fallen,
    params={"command_name": "twist", "motion_command": "motion"},
  )
  critic_terms["command"] = replace(student.terms["command"])
  critic_terms["base_lin_vel"] = ObservationTermCfg(func=envs_mdp.base_lin_vel)
  critic_terms["fallen"] = replace(cfg.observations["fallen"].terms["fallen"])
  cfg.observations = {
    "actor": student,
    "critic": ObservationGroupCfg(terms=critic_terms, enable_corruption=False),
  }
  cfg.commands["twist"] = mdp.VelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=KEYS_RESAMPLING_S,
    heading_command=False,
    rel_standing_envs=KEYS_STANDING,
    rel_heading_envs=0.0,
    debug_vis=True,
    ranges=mdp.VelocityCommandCfg.Ranges(
      lin_vel_x=KEYS_LIN_VEL_X, lin_vel_y=KEYS_LIN_VEL_Y, ang_vel_z=KEYS_ANG_VEL_Z
    ),
  )
  for name in (
    "palm_position",
    "palm_orientation",
    "body_position",
    "body_orientation",
    "body_lin_vel",
    "body_ang_vel",
    "planned_velocity",
  ):
    del cfg.rewards[name]
  motion = {"command_name": "motion"}
  cfg.rewards["commanded_velocity"] = RewardTermCfg(
    func=mdp.commanded_velocity_exp,
    weight=2.0,
    params={
      "command_name": "twist",
      "motion_command": "motion",
      "std_lin": 0.01,
      "std_yaw": 0.15,
    },
  )
  cfg.rewards["up"] = RewardTermCfg(func=mdp.not_fallen, weight=1.0, params=motion)
  return cfg


# Velocity-v1's tracking: the scales Walking-Leap-Flat-v3 learned to walk with
# (env_cfg.LIN_VEL_TRACKING_STD, YAW_TRACKING_STD). Velocity-v0's single term (std
# 0.01 m/s, half its value from the turn) barely moved: forward 1.7 of 30 mm/s.
KEYS_LIN_STD, KEYS_YAW_STD = 0.02, 0.25  # m/s, rad/s


def walking_leap_velocity_v1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Velocity-v0 with the velocity reward split in two: the horizontal velocity (std
  KEYS_LIN_STD, weight 2) and the turn rate (std KEYS_YAW_STD, weight 1), so that
  standing still under a translation command pays nothing for it. Same observations:
  it continues from Velocity-v0."""
  cfg = walking_leap_velocity_env_cfg(play=play)
  del cfg.rewards["commanded_velocity"]
  keys = {"command_name": "twist", "motion_command": "motion"}
  cfg.rewards["track_lin_vel"] = RewardTermCfg(
    func=mdp.commanded_lin_vel_exp, weight=2.0, params={**keys, "std": KEYS_LIN_STD}
  )
  cfg.rewards["track_yaw_vel"] = RewardTermCfg(
    func=mdp.commanded_yaw_vel_exp, weight=1.0, params={**keys, "std": KEYS_YAW_STD}
  )
  return cfg


def walking_leap_velocity_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Velocity-v1 with the translation reward weighing 3 (2 in v1): with ``up``,
  ``upright`` and the stance's height the hand earned 2.5 a step for standing safely."""
  cfg = walking_leap_velocity_v1_env_cfg(play=play)
  cfg.rewards["track_lin_vel"].weight = 3.0
  return cfg


def walking_leap_velocity_v3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Velocity-v2 with Gait-v1's light pushes (``PUSH_RANGE``, 0.1 m/s per axis) instead
  of 0.4 m/s every 1-3 s: under those the hand walked at a seventh of what it is asked
  (v2: forward 4.6 of 30 mm/s) and Gait-v2 slowed down too. A curriculum's first step:
  walk fast, then push harder."""
  cfg = walking_leap_velocity_v2_env_cfg(play=play)
  if "push" in cfg.events:
    cfg.events["push"].params["velocity_range"] = PUSH_RANGE
  return cfg
