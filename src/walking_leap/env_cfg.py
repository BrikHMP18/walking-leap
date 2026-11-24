"""Flat-ground forward-walking environment for the LEAP Hand.

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
         velocity tasks; only vx > 0 is used for now.
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
    "actuator_saturation": MetricsTermCfg(
      func=mdp.actuator_saturation, params={"effort_limit": EFFORT_LIMIT}
    ),
    "mechanical_power": MetricsTermCfg(func=mdp.mechanical_power),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_leap_hand_cfg()},
      sensors=(palm_contact, digit_contact),
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
