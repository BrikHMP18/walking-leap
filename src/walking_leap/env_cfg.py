"""Flat-ground walking environments for the LEAP Hand.

  Walking-Leap-Flat-v0        constant forward command (baseline).
  Walking-Leap-Flat-v1        omnidirectional velocity commands.
  Walking-Leap-Flat-NoImu-v1  as v1, actor without the palm-IMU observations.

Built directly from mjlab managers instead of mjlab's legged velocity template,
which carries quadruped assumptions (feet, air time, swing height, terrain
scans). Reused as-is: generic mjlab MDP terms (``mjlab.envs.mdp``) and the
velocity-task tracking/upright/contact terms (``mjlab.tasks.velocity.mdp``).
Only what mjlab lacks lives in ``mdp.py`` (including a GUI-less subclass of
mjlab's velocity command).

Reference frames:
  world  z up, gravity (0, 0, -9.81); flat ground plane at z = 0.
  palm   +x wrist->fingertips (robot forward), +y thumb side, +z back of hand.
         Identity orientation = palm down and level (see robot.py).
  command (vx, vy, wz) is expressed in the palm (body) frame, like mjlab's
         velocity tasks.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
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
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from walking_leap import mdp
from walking_leap.robot import (
  EFFORT_LIMIT,
  FINGER_ROOT_BODIES,
  FINGERTIP_BODIES,
  ROOT_BODY,
  get_leap_hand_cfg,
  joint_limits,
)

# Commanded forward speed (m/s). The stance is ~0.15 m long (thumb tip to finger
# tips); 0.05 m/s is a third of that per second, a deliberately modest start.
FORWARD_VELOCITY = 0.05
# Joint target offset (rad) per unit action: target = nominal + scale * action.
ACTION_SCALE = 0.25
EPISODE_LENGTH_S = 10.0

PALM_CONTACT = "palm_ground_contact"
DIGIT_CONTACT = "digit_ground_contact"
TIP_CONTACT = "tip_ground_contact"

# Omnidirectional commands (v1), palm frame. Forward 0.05 m/s is reachable
# (v0 baseline: 0.049 m/s); the backward, lateral and yaw limits are first
# guesses, not yet shown to be reachable.
OMNI_LIN_VEL_X = (-0.05, 0.05)  # m/s
OMNI_LIN_VEL_Y = (-0.03, 0.03)  # m/s
OMNI_ANG_VEL_Z = (-0.3, 0.3)  # rad/s
OMNI_STANDING_FRACTION = 0.15  # share of commands that are exactly zero
OMNI_RESAMPLING_TIME_S = (3.0, 5.0)  # 2-3 commands per 10 s episode
# Tracking kernels exp(-err^2 / std^2), with stand-still payoffs computed by
# Monte Carlo over the command distribution above:
# - Linear: with v0's std of 0.05 m/s a hand that stands still would earn 72 %
#   of the term, since most v1 commands are slower than v0's. 0.025 m/s (half
#   the largest speed, mjlab's ratio) brings that to 40 %, close to v0's 37 %.
# - Yaw: v0 used std 0.5 rad/s, weight 0.5 (stabilization only): standing
#   still against a 0.3 rad/s command would still earn 67 % of the term. With
#   0.25 it earns 20 %, while the v0 policy's angular-rate error (~0.1 rad/s
#   RMS, from its track_ang_vel of 0.48 / 0.5) keeps 85 %.
LIN_VEL_TRACKING_STD = 0.025  # m/s
YAW_TRACKING_STD = 0.25  # rad/s
YAW_TRACKING_WEIGHT = 1.0
# Fingertip sliding penalty: sum over tips in contact of (slip / 1 cm/s)^2. The
# v0 gait slid at 1.3-1.9 cm/s on ~2.4 tips, which would cost 0.19-0.43 per
# second, less than the 1.2 per second that tracking adds over standing still;
# the standing hand's vibration (< 1 mm/s) costs < 0.002 per second.
SLIP_REFERENCE_SPEED = 0.01  # m/s
SLIP_PENALTY_WEIGHT = -0.05
# Actor terms that need an IMU on the palm (not on the stock hand).
IMU_OBSERVATIONS = ("base_ang_vel", "projected_gravity")


def walking_leap_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Forward walking on flat ground. ``play=True`` gives an endless episode."""

  ##
  # Sensors: palm-ground and per-digit ground contact (whole digit subtree, so
  # contacts on any finger surface count, not only the tips).
  ##

  palm_contact = ContactSensorCfg(
    name=PALM_CONTACT,
    primary=ContactMatch(mode="body", pattern=ROOT_BODY, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    track_air_time=True,
  )
  digit_contact = ContactSensorCfg(
    name=DIGIT_CONTACT,
    primary=ContactMatch(mode="subtree", pattern=FINGER_ROOT_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    track_air_time=True,
  )
  # Distal links (fingertip pads) only, with the deepest contact's position:
  # used by the tip-slip metric, not observed by the actor or the critic.
  tip_contact = ContactSensorCfg(
    name=TIP_CONTACT,
    primary=ContactMatch(mode="body", pattern=FINGERTIP_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "pos"),
    reduce="mindist",
  )

  ##
  # Observations. Actor: proprioception that the real hand can provide.
  #   joint_pos / joint_vel / actions: Dynamixel encoders + policy history.
  #   base_ang_vel, projected_gravity: need an IMU on the palm (not part of the
  #     stock LEAP Hand) - here taken from the simulator state, noise-free.
  # Critic: actor terms plus privileged simulator state.
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "twist"}
    ),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "root_height": ObservationTermCfg(func=mdp.root_height),
    "digit_contact": ObservationTermCfg(
      func=vel_mdp.foot_contact, params={"sensor_name": DIGIT_CONTACT}
    ),
    "digit_contact_forces": ObservationTermCfg(
      func=vel_mdp.foot_contact_forces, params={"sensor_name": DIGIT_CONTACT}
    ),
    "palm_contact": ObservationTermCfg(
      func=vel_mdp.foot_contact, params={"sensor_name": PALM_CONTACT}
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(terms=actor_terms, enable_corruption=False),
    "critic": ObservationGroupCfg(terms=critic_terms, enable_corruption=False),
  }

  ##
  # Actions: 16 joint position targets around the nominal pose, clipped to the
  # MJCF joint limits before they reach the position actuators.
  ##

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=ACTION_SCALE,
      use_default_offset=True,
      clip=joint_limits(),
    )
  }

  ##
  # Command: constant forward velocity in the palm frame.
  ##

  commands = {
    "twist": mdp.VelocityCommandCfg(
      entity_name="robot",
      # The command is constant; this window only normalizes mjlab's
      # Metrics/twist/error_vel_* (mean tracking error over one episode).
      resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
      heading_command=False,
      rel_standing_envs=0.0,
      rel_heading_envs=0.0,
      debug_vis=True,
      ranges=mdp.VelocityCommandCfg.Ranges(
        lin_vel_x=(FORWARD_VELOCITY, FORWARD_VELOCITY),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
      ),
      viz=mdp.VelocityCommandCfg.VizCfg(z_offset=0.1, scale=2.0),
    )
  }

  ##
  # Events: reset to the nominal pose, dropped from <= 5 mm, random heading.
  # No domain randomization yet (nominal physics first).
  ##

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"z": (0.0, 0.005), "yaw": (-math.pi, math.pi)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  ##
  # Rewards (each term is multiplied by weight * step_dt, i.e. weights are per
  # second). No gait timing, contact sequence, reference motion or symmetry.
  ##

  rewards = {
    # exp(-(|v_cmd,xy - v_xy|^2 + v_z^2) / std^2), body frame. std must be on
    # the scale of the command: standing still scores exp(-1) = 0.37.
    "track_lin_vel": RewardTermCfg(
      func=vel_mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": FORWARD_VELOCITY},
    ),
    # Zero yaw-rate command; also penalizes roll/pitch rates.
    "track_ang_vel": RewardTermCfg(
      func=vel_mdp.track_angular_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": 0.5},
    ),
    "palm_height": RewardTermCfg(
      func=mdp.root_height_floor,
      weight=1.0,
      params={"min_height": 0.08, "std": 0.02},
    ),
    "upright": RewardTermCfg(
      func=vel_mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=(ROOT_BODY,)),
      },
    ),
    "joint_torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-0.2),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    "palm_contact": RewardTermCfg(
      func=mdp.contact_active, weight=-2.0, params={"sensor_name": PALM_CONTACT}
    ),
  }

  ##
  # Terminations. Only time_out is a truncation (time_out=True); the others are
  # physical failures.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "nan_state": TerminationTermCfg(func=envs_mdp.nan_detection),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
    ),
    "palm_on_ground": TerminationTermCfg(
      func=mdp.sustained_contact,
      params={"sensor_name": PALM_CONTACT, "max_duration": 1.0},
    ),
  }

  ##
  # Metrics: locomotion quality reported separately from the reward.
  ##

  metrics = {
    "forward_velocity": MetricsTermCfg(func=mdp.forward_velocity),
    "net_displacement": MetricsTermCfg(func=mdp.net_displacement, reduce="last"),
    "palm_ground_contact": MetricsTermCfg(
      func=mdp.contact_active, params={"sensor_name": PALM_CONTACT}
    ),
    # Fraction of time each digit (any part of it) touches the ground: a digit
    # near 1.0 while the hand travels is being dragged, not stepped.
    **{
      f"digit_contact_{body[:2]}": MetricsTermCfg(
        func=mdp.contact_active,
        params={"sensor_name": DIGIT_CONTACT, "primary": body},
      )
      for body in FINGER_ROOT_BODIES
    },
    # Sliding speed of the fingertip pads in contact (m/s): separates steps from
    # skating or vibration-driven shuffling.
    "tip_slip_speed": MetricsTermCfg(
      func=mdp.contact_slip_speed, params={"sensor_name": TIP_CONTACT}
    ),
    "actuator_saturation": MetricsTermCfg(
      func=mdp.actuator_saturation, params={"effort_limit": EFFORT_LIMIT}
    ),
    "mechanical_power": MetricsTermCfg(func=mdp.mechanical_power),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_leap_hand_cfg()},
      sensors=(palm_contact, digit_contact, tip_contact),
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
    # 500 Hz physics, 50 Hz policy. mjlab ignores the XML <option>, so it is set
    # here.
    # - Elliptic cones (exact Coulomb cone, as in the Menagerie model): 51k vs
    #   85k env-steps/s for pyramidal (4096 envs, RTX 5060 Laptop).
    # - Solver iterations stay at MujocoCfg's defaults (100 / 50 line search).
    #   With 10 / 20 (mjlab's quadruped setting) the elliptic solve does not
    #   converge and the standing hand creeps 32 mm in 4 s (0.5 mm here); the
    #   solver stops at its tolerance, so the higher caps cost little.
    # - impratio is mjlab's default 1, not the Menagerie's 100: with 100, MuJoCo
    #   Warp 3.11 contact sensors returned NaN forces in ~1e-6 of env-steps
    #   (physics state stayed finite); none were seen with 1.
    # Buffer sizes are per world, ~4x the measured worst case: lying palm-down
    # with straight digits (16 contacts, 64 constraint rows), upside down (12,
    # 52), random actions (<= 46 rows). mjlab's automatic sizing ran out of
    # memory on an 8 GB GPU at 4096 envs.
    sim=SimulationCfg(
      nconmax=64,
      njmax=256,
      mujoco=MujocoCfg(
        timestep=0.002,
        integrator="implicitfast",
        cone="elliptic",
        impratio=1.0,
      ),
    ),
    decimation=10,
    episode_length_s=EPISODE_LENGTH_S,
  )

  if play:
    cfg.episode_length_s = int(1e9)

  return cfg


def walking_leap_flat_omni_env_cfg(
  play: bool = False, imu: bool = True
) -> ManagerBasedRlEnvCfg:
  """Omnidirectional velocity tracking on flat ground (v1).

  Everything not listed here is v0 (robot, physics, actions, events,
  terminations, the other reward terms):
  - commands resampled every 3-5 s from the OMNI_* ranges, 15 % of them zero;
  - linear and yaw-rate tracking tightened (yaw also weighted) so that moving
    on command pays more than standing still;
  - a fingertip-sliding penalty, since the v0 gait partly dragged its tips;
  - reward-independent tracking-error and per-tip sliding metrics;
  - ``imu=False`` removes the palm-IMU terms from the actor (51 instead of 57
    dims). The critic keeps them: it only exists in simulation.
  """
  cfg = walking_leap_flat_env_cfg(play=play)

  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.VelocityCommandCfg)
  twist.ranges = mdp.VelocityCommandCfg.Ranges(
    lin_vel_x=OMNI_LIN_VEL_X, lin_vel_y=OMNI_LIN_VEL_Y, ang_vel_z=OMNI_ANG_VEL_Z
  )
  twist.rel_standing_envs = OMNI_STANDING_FRACTION
  twist.resampling_time_range = OMNI_RESAMPLING_TIME_S

  cfg.rewards["track_lin_vel"].params["std"] = LIN_VEL_TRACKING_STD
  cfg.rewards["track_ang_vel"].params["std"] = YAW_TRACKING_STD
  cfg.rewards["track_ang_vel"].weight = YAW_TRACKING_WEIGHT
  cfg.rewards["tip_slip"] = RewardTermCfg(
    func=mdp.contact_slip_penalty,
    weight=SLIP_PENALTY_WEIGHT,
    params={"sensor_name": TIP_CONTACT, "reference_speed": SLIP_REFERENCE_SPEED},
  )
  # Sliding per tip (m/s while in contact, 0 while lifted): shows which digit
  # drags, e.g. the thumb, the only rear support.
  for body in FINGERTIP_BODIES:
    cfg.metrics[f"tip_slip_rate_{body[:2]}"] = MetricsTermCfg(
      func=mdp.contact_slip_rate, params={"sensor_name": TIP_CONTACT, "primary": body}
    )

  # mjlab's Metrics/twist/error_vel_* normalize by the longest resampling
  # window, not the episode, so with resampling they are not per-step means.
  cfg.metrics["lin_vel_tracking_error"] = MetricsTermCfg(
    func=mdp.lin_vel_tracking_error, params={"command_name": "twist"}
  )
  cfg.metrics["yaw_rate_tracking_error"] = MetricsTermCfg(
    func=mdp.yaw_rate_tracking_error, params={"command_name": "twist"}
  )

  if not imu:
    actor_terms = cfg.observations["actor"].terms
    for name in IMU_OBSERVATIONS:
      del actor_terms[name]

  return cfg
