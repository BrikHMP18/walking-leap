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


def walking_leap_gait_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Imitate the scripted crawl (assets/clips/gait.npz) and survive: the real hand
  with the IMU, everything random of ``robust`` and the cables, and falls. A fall
  no longer ends an episode: the clip waits while the hand is down and resumes where
  it gets up (``mdp.GaitMotionCommand``); a fifth of the episodes start fallen;
  rewards for an upright palm at the clip's height, and for moving as the plan
  intends (``mdp.planned_velocity_exp``), which the slipping replay undershoots.
  Observations are those of ``Walking-Leap-Motion-Gait-Flat-v0``, so training can
  continue from its checkpoints."""
  cfg = walking_leap_motion_env_cfg(
    play=play,
    robust=True,
    cables=True,
    clip="gait",
    feet=False,
    stiffness=MEASURED_STIFFNESS,
    torsion=TORSION,
  )
  motion = {"command_name": "motion"}
  old = cfg.commands["motion"]
  cfg.commands["motion"] = mdp.GaitMotionCommandCfg(
    **{f.name: getattr(old, f.name) for f in fields(old) if f.init},
    fallen_start_prob=0.0 if play else FALLEN_START_PROB,
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
