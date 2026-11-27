"""Crawling lying down on joystick commands: ``Walking-Leap-Lying-Flat-v0`` and
``Walking-Leap-Lying-Low-Flat-v0``.

The hand lies palm down and pulls or pushes itself with its fingers, as the dead crawls
(real/make_crawl_dead.py, make_crawl_deep.py): forward, back, turning, or both at once,
never standing. The standing walkers hop (one digit held up, the others jumping).

Every episode starts lying and settled, at the stand-up clip's first frame (the twin's
``Sim.lie``), and the actions are offsets from that pose (the exported policy's
``default``). No palm-contact cost or termination; an episode ends if the hand flips
(tilt over 50 deg) or gets up (the palm's origin 30 mm above lying; 15 mm for Low).

Rewards only, all bounded (lying_mdp): progress and turning capped at the command,
the linear tracking errors (capped high), an alive reward, standing still on a zero
command, level (Flat-v3's upright), the guard on motors at their cap, the bounded tip
sliding, torques, the action rate. Low adds a cost on the palm rising more than 8 mm,
so the palm stays on or near the table and the fingers do the work.

The hand and what the real loop does to it are the joystick walker's: Flat-v3's real
hand (IMU, stock tips, 350 mA, measured stiffness, torsion), targets up to 20 ms late,
the setpoint at most 5 rad/s, sensor noise, kicks, the cables' pull, random grip, mass
and motors; the palm's own sliding friction, never measured, is random too. The actor
sees the velocity student's 285 inputs, so real/export_policy.py exports it (with
``--lying``: its default is the lying pose).
"""

import math
from dataclasses import fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from walking_leap import lying_mdp as lm
from walking_leap import mdp
from walking_leap.agent_cfg import walking_leap_ppo_runner_cfg
from walking_leap.env_cfg import walking_leap_flat_real_env_cfg
from walking_leap.motion_cfg import (
  CABLE_BODIES,
  CABLE_FORCE,
  CABLE_TORQUE,
  PUSH_RANGE,
  STUDENT_HISTORY,
)
from walking_leap.robot import FINGERTIP_GEOMS, ROOT_BODY

# The stand-up clip's first frame (real/clips/stand_up.npz, row 0, to 1 mrad): how the
# real loop and the panel have the hand lying. Repo joint order.
LYING_JOINT_POS: dict[str, float] = {
  "if_mcp": 0.054,
  "if_rot": -0.322,
  "if_pip": -0.086,
  "if_dip": -0.091,
  "mf_mcp": 0.072,
  "mf_rot": 0.011,
  "mf_pip": -0.028,
  "mf_dip": -0.337,
  "rf_mcp": 0.046,
  "rf_rot": 0.236,
  "rf_pip": -0.061,
  "rf_dip": -0.362,
  "th_cmc": 0.066,
  "th_axl": 1.551,
  "th_mcp": -0.114,
  "th_ipl": -0.041,
}
# The palm's origin lying settled in that pose (twin.Sim.lie, 2 s): 34.2 mm, level
# within 0.7 deg. Dropped from 2 mm above it (and up to 5 mm more, the reset's range).
LYING_HEIGHT = 0.0342
STOOD_UP = 0.030  # m above lying: the hand got up (it stands at 129 mm)
STOOD_UP_LOW = 0.015
FLIPPED_DEG = 50.0

# The joystick's range, palm frame, from the deep crawl on the twin
# (real/make_crawl_deep.py, open loop): forward about 4-5 mm/s on a gripping palm, back
# faster (pushing), drifting about 3 deg/s.
LYING_FORWARD = 0.006  # m/s
LYING_BACK = -0.008  # m/s
LYING_TURN = 0.10  # rad/s, either way (5.7 deg/s)
LYING_STANDING = 0.2  # share of the commands that stand still
LYING_RESAMPLING_S = (2.0, 5.0)  # a stroke takes about 2.5 s
EPISODE_S = 20.0
MIN_SPEED = 0.001  # m/s: under this the command stands
MIN_RATE = 0.02  # rad/s

PROGRESS_WEIGHT = 2.0
TURN_WEIGHT = 1.0
SPEED_ERROR_WEIGHT = -0.5
TURN_ERROR_WEIGHT = -0.3
ERROR_CAP = 4.0  # times the reference: high
ALIVE_WEIGHT = 1.0
STILL_WEIGHT = -0.5
HELD_AT_CAP_WEIGHT = -2.0
RATE_WEIGHT = -0.05
RATE_CAP = 4.0
PALM_ABOVE_WEIGHT = -1.0
PALM_MARGIN = 0.008  # m: Low's palm may rise this much for free
PALM_SCALE = 0.010  # m: the cost's full value 10 mm beyond


def _noise(limit: float) -> Unoise:
  return Unoise(n_min=-limit, n_max=limit)


def walking_leap_lying_env_cfg(
  play: bool = False, low: bool = False
) -> ManagerBasedRlEnvCfg:
  """Lying joystick crawl (module docstring); ``low`` keeps the palm down."""
  cfg = walking_leap_flat_real_env_cfg(play=play, imu=True)
  if not play:
    cfg.episode_length_s = EPISODE_S

  # Lying, settled; the actions around that pose. The real loop: targets up to 20 ms
  # late, the setpoint at most 5 rad/s.
  hand = cfg.scene.entities["robot"]
  assert hand.articulation is not None
  motors = tuple(replace(a, delay_max_lag=10) for a in hand.articulation.actuators)
  cfg.scene.entities["robot"] = replace(
    hand,
    init_state=replace(
      hand.init_state,
      pos=(0.0, 0.0, LYING_HEIGHT + 0.002),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos=dict(LYING_JOINT_POS),
    ),
    articulation=replace(hand.articulation, actuators=motors),
  )
  old = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = mdp.SlewJointPositionActionCfg(
    **{f.name: getattr(old, f.name) for f in fields(old) if f.init}
  )

  # The velocity student's inputs, in its order (real/policy.py, VELOCITY_INPUTS), with
  # sensor noise; the critic sees them clean, plus Flat-v3's privileged state.
  actor = {
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "twist"}
    ),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel, noise=_noise(0.05)),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity, noise=_noise(0.03)
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel, params={"biased": True}, noise=_noise(0.01)
    ),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, noise=_noise(0.3)),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  privileged = {
    k: t for k, t in cfg.observations["critic"].terms.items() if k not in actor
  }
  critic = {
    **{k: ObservationTermCfg(func=t.func, params=t.params) for k, t in actor.items()},
    **privileged,
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor, enable_corruption=True, history_length=STUDENT_HISTORY
    ),
    "critic": ObservationGroupCfg(terms=critic, enable_corruption=False),
  }

  # The joystick: the whole range from the start, a fifth of the commands standing.
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.VelocityCommandCfg)
  twist.ranges = mdp.VelocityCommandCfg.Ranges(
    lin_vel_x=(LYING_BACK, LYING_FORWARD),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(-LYING_TURN, LYING_TURN),
  )
  twist.rel_standing_envs = LYING_STANDING
  twist.resampling_time_range = LYING_RESAMPLING_S
  twist.heading_command = False
  twist.rel_heading_envs = 0.0

  # Randomization, as the joystick walker's, plus the palm's and links' sliding friction.
  pull = {
    "asset_cfg": SceneEntityCfg("robot", body_names=CABLE_BODIES),
    "force_range": (-CABLE_FORCE, CABLE_FORCE),
    "torque_range": (-CABLE_TORQUE, CABLE_TORQUE),
  }
  cfg.events.update(
    {
      "push": EventTermCfg(
        func=envs_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": PUSH_RANGE},
      ),
      "foot_friction": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=FINGERTIP_GEOMS),
          "operation": "abs",
          "ranges": (0.5, 1.0),
          "shared_random": True,
        },
      ),
      "palm_friction": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_collision.*",)),
          "operation": "abs",
          "ranges": (0.2, 1.0),
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
      "cables_at_reset": EventTermCfg(
        func=envs_mdp.apply_external_force_torque, mode="reset", params=pull
      ),
      "cables": EventTermCfg(
        func=envs_mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(1.0, 4.0),
        params=pull,
      ),
    }
  )

  # Rewards (per second), all bounded. Flat-v3's that stay: upright (level), torques,
  # the bounded tip sliding. Gone: its exp tracking (replaced by the linear terms), the
  # palm's 8 cm floor, palm contact, the digits' air-time cap.
  for name in ("track_lin_vel", "track_ang_vel", "palm_height", "palm_contact",
               "long_air_time", "action_rate"):  # fmt: skip
    cfg.rewards.pop(name, None)
  twist_cmd = {"command_name": "twist"}
  cfg.rewards.update(
    {
      "progress": RewardTermCfg(
        func=lm.capped_progress,
        weight=PROGRESS_WEIGHT,
        params={**twist_cmd, "min_speed": MIN_SPEED, "reference_speed": LYING_FORWARD},
      ),
      "turn": RewardTermCfg(
        func=lm.capped_turn,
        weight=TURN_WEIGHT,
        params={**twist_cmd, "min_rate": MIN_RATE, "reference_rate": LYING_TURN},
      ),
      "speed_error": RewardTermCfg(
        func=lm.speed_error,
        weight=SPEED_ERROR_WEIGHT,
        params={**twist_cmd, "reference_speed": LYING_FORWARD, "cap": ERROR_CAP},
      ),
      "turn_error": RewardTermCfg(
        func=lm.turn_error,
        weight=TURN_ERROR_WEIGHT,
        params={**twist_cmd, "reference_rate": LYING_TURN, "cap": ERROR_CAP},
      ),
      "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=ALIVE_WEIGHT),
      "still": RewardTermCfg(
        func=lm.still,
        weight=STILL_WEIGHT,
        params={
          **twist_cmd,
          "reference_speed": LYING_FORWARD,
          "reference_rate": LYING_TURN,
          "min_speed": MIN_SPEED,
          "min_rate": MIN_RATE,
        },
      ),
      "held_at_cap": RewardTermCfg(
        func=lm.held_at_cap,
        weight=HELD_AT_CAP_WEIGHT,
        params={"ratio": 0.94, "grace": 0.5},
      ),
      "action_rate": RewardTermCfg(
        func=lm.rate_bounded, weight=RATE_WEIGHT, params={"cap": RATE_CAP}
      ),
    }
  )
  if low:
    cfg.rewards["palm_above"] = RewardTermCfg(
      func=lm.palm_above,
      weight=PALM_ABOVE_WEIGHT,
      params={"height": LYING_HEIGHT + PALM_MARGIN, "scale": PALM_SCALE},
    )

  # Terminations: no palm contact; flipping or getting up ends it.
  cfg.terminations.pop("palm_on_ground", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(FLIPPED_DEG)}
  )
  cfg.terminations["stood_up"] = TerminationTermCfg(
    func=lm.stood_up,
    params={"height": LYING_HEIGHT + (STOOD_UP_LOW if low else STOOD_UP)},
  )
  return cfg


# Low-v1 (11-27, after v0's first look): v0 at iteration 400 pinned the side-to-side
# knuckles at their current limit on the real loop's twin (the loop stops at 1.5 s)
# while training saw motors saturated 8 % of the time and charged little for it; and
# any stroke pressing the tips pitches the front up (the origin 35-60 mm over lying),
# which v0's "got up" at +15/30 mm ended. So: the at-cap cost after 0.2 s and heavier,
# a cost on the share of motors saturated, "got up" as standing (high and level) or
# very high, and the palm's cost from 25 mm up.
GOT_UP = 0.045  # m above lying, and level within GOT_UP_LEVEL_DEG: on its fingertips
GOT_UP_LEVEL_DEG = 20.0
GOT_UP_HIGH = 0.070  # m above lying, however tilted (it stands at +95 mm)
V1_HELD_WEIGHT = -4.0
V1_HELD_GRACE = 0.2  # s
SATURATED_WEIGHT = -2.0
V1_PALM_MARGIN = 0.025  # m
V1_PALM_SCALE = 0.020  # m


def walking_leap_lying_v1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """``Walking-Leap-Lying-Low-Flat-v1``: Low-v0 with the guard and "got up" fixed
  (above)."""
  cfg = walking_leap_lying_env_cfg(play=play, low=True)
  held = cfg.rewards["held_at_cap"]
  held.weight, held.params["grace"] = V1_HELD_WEIGHT, V1_HELD_GRACE
  cfg.rewards["saturated"] = RewardTermCfg(
    func=lm.saturated_share, weight=SATURATED_WEIGHT, params={"ratio": 0.94}
  )
  cfg.rewards["palm_above"].params.update(
    {"height": LYING_HEIGHT + V1_PALM_MARGIN, "scale": V1_PALM_SCALE}
  )
  cfg.terminations["stood_up"] = TerminationTermCfg(
    func=lm.stood_up_level,
    params={
      "height": LYING_HEIGHT + GOT_UP,
      "level_deg": GOT_UP_LEVEL_DEG,
      "high": LYING_HEIGHT + GOT_UP_HIGH,
    },
  )
  return cfg


# v1 (Lying) and Low-v2, after Low-v1's design: v0 at iteration 700, told to stand still on
# the twin, still pressed the middle finger and the thumb into the table and squeezed
# the side-to-side knuckles together, four motors at their limit for 1.5 s (the loop
# stops). Training's force-based terms hardly fire there (its contact is softer), so the
# setpoints far from the joints are charged too (held_far), on top of Low-v1's fixes.
FAR_ERROR = 0.11  # rad: the twin's servo (1.6 N*m/rad) is at its 0.18 N*m cap beyond it
FAR_GRACE = 0.5  # s
FAR_WEIGHT = -4.0


def walking_leap_lying_v2_env_cfg(
  play: bool = False, low: bool = True
) -> ManagerBasedRlEnvCfg:
  """``Walking-Leap-Lying-Flat-v1`` (``low=False``: no palm cost) and
  ``Walking-Leap-Lying-Low-Flat-v2``: Low-v1 with ``held_far``."""
  cfg = walking_leap_lying_v1_env_cfg(play=play)
  if not low:
    del cfg.rewards["palm_above"]
  cfg.rewards["held_far"] = RewardTermCfg(
    func=lm.held_far,
    weight=FAR_WEIGHT,
    params={"error": FAR_ERROR, "grace": FAR_GRACE},
  )
  return cfg


# Lying-Flat-v2 and Low-v3: v1/Low-v2 at iteration 200 still pinned a motor on the twin,
# and their held_far stayed at 0 in training (the exploration noise). So a cost on the
# setpoints far from their joints now, and half the initial exploration noise
# (``walking_leap_lying_v3_ppo_runner_cfg``).
FAR_SPAN = 0.3  # rad beyond FAR_ERROR: the cost's full value
FAR_NOW_WEIGHT = -3.0


def walking_leap_lying_v3_env_cfg(
  play: bool = False, low: bool = True
) -> ManagerBasedRlEnvCfg:
  """``Walking-Leap-Lying-Flat-v2`` (``low=False``) and ``-Low-Flat-v3``: v1/Low-v2 with
  ``far_setpoints``."""
  cfg = walking_leap_lying_v2_env_cfg(play=play, low=low)
  cfg.rewards["far_now"] = RewardTermCfg(
    func=lm.far_setpoints,
    weight=FAR_NOW_WEIGHT,
    params={"error": FAR_ERROR, "span": FAR_SPAN},
  )
  return cfg


def walking_leap_lying_low_v4_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """``Walking-Leap-Lying-Low-Flat-v4``: Low-v3 with the far setpoints charged on the
  worst motor (Lying-v2 at iteration 100 still held the middle finger's knuckle pressed
  into the table on the twin; averaged over 16 motors that cost about 0.2 a second)."""
  cfg = walking_leap_lying_v3_env_cfg(play=play, low=True)
  cfg.rewards["far_now"].params["worst"] = True
  return cfg


def walking_leap_lying_goal_env_cfg(
  play: bool = False, low: bool = True
) -> ManagerBasedRlEnvCfg:
  """``Walking-Leap-Lying-Flat-v3`` (``low=False``) and ``-Low-Flat-v5``: the far
  costs on the policy's targets, not the motor model's setpoint. Lying-v2 at iteration
  600, the loop's guard off on the twin, crawled (back 12.8 mm/s, turns 4.7 and 3.9
  deg/s) with one motor at its limit the whole run; in training its setpoints never
  strayed 0.1 rad from the joints (the motor model keeps them within its cap's reach),
  so held_far and far_now could not see it. Fine-tuned from v2 / Low-v4 checkpoints."""
  cfg = walking_leap_lying_v3_env_cfg(play=play, low=low)
  cfg.rewards["held_far"].func = lm.held_far_goal
  cfg.rewards["far_now"] = RewardTermCfg(
    func=lm.far_goals,
    weight=FAR_NOW_WEIGHT,
    params={"error": FAR_ERROR, "span": FAR_SPAN},
  )
  return cfg


def walking_leap_lying_track_v1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """``Walking-Leap-Lying-Track-Flat-v1``: the tracker with the far costs on the policy's
  targets (Lying-v3's). Track-v0 at iteration 600 on the twin played forward and back
  but held the thumb's base at its limit 1.5 s at the end of the turn and back clips.
  Fine-tuned from Track-v0."""
  cfg = walking_leap_lying_track_env_cfg(play=play)
  cfg.rewards["held_far"] = RewardTermCfg(
    func=lm.held_far_goal,
    weight=FAR_WEIGHT,
    params={"error": FAR_ERROR, "grace": FAR_GRACE},
  )
  cfg.rewards["far_now"] = RewardTermCfg(
    func=lm.far_goals,
    weight=FAR_NOW_WEIGHT,
    params={"error": FAR_ERROR, "span": FAR_SPAN},
  )
  return cfg


def walking_leap_lying_low_v6_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """``Walking-Leap-Lying-Low-Flat-v6``: Low-v5 paying forward and turning three times
  more. Lying-v3 at 900 and Low-v5 at 400 passed the twin's guard but only backed up
  (8 mm/s, in training's sim too): pushing is easy, pulling and turning were not
  learned. Fine-tuned from Low-v5."""
  cfg = walking_leap_lying_goal_env_cfg(play=play, low=True)
  cfg.rewards["turn"].weight = 3 * TURN_WEIGHT
  cfg.rewards["forward"] = RewardTermCfg(
    func=lm.capped_forward,
    weight=2 * PROGRESS_WEIGHT,
    params={
      "command_name": "twist",
      "min_speed": MIN_SPEED,
      "reference_speed": LYING_FORWARD,
    },
  )
  return cfg


def walking_leap_lying_v3_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The lying learner with half the initial exploration noise (std 0.5)."""
  cfg = walking_leap_lying_ppo_runner_cfg()
  cfg.actor.distribution_cfg = {**cfg.actor.distribution_cfg, "init_std": 0.5}
  return cfg


def walking_leap_lying_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Flat-v3's learner (the student's network, so the export reads it), its own
  experiment folder."""
  return replace(
    walking_leap_ppo_runner_cfg(),
    experiment_name="walking_leap_lying",
    max_iterations=4_000,
  )


def walking_leap_lying_track_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """``Walking-Leap-Lying-Track-Flat-v0``: a tracker of the deep crawls (one library,
  assets/clips/crawl_deep.npz: forward, back, turn left, turn right, as the twin plays
  them; real/make_crawl_deep.py --library), on the real hand's model with the robust
  imitation's randomization and the cables, targets up to 20 ms late, the driver's
  5 rad/s setpoint limit. Each episode starts at a random frame of the library, lying."""
  from walking_leap.motion_cfg import walking_leap_motion_env_cfg
  from walking_leap.robot import MEASURED_STIFFNESS, TORSION

  cfg = walking_leap_motion_env_cfg(
    play=play,
    robust=True,
    cables=True,
    clip="crawl_deep",
    feet=False,
    imu=True,
    stiffness=MEASURED_STIFFNESS,
    torsion=TORSION,
  )
  old = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = mdp.SlewJointPositionActionCfg(
    **{f.name: getattr(old, f.name) for f in fields(old) if f.init}
  )
  # Lying on the library's crawls overflows the motion task's 256 constraint rows (MuJoCo
  # Warp asked for up to 304 in the first iterations): dropped contacts.
  cfg.sim.njmax, cfg.sim.nconmax = 512, 128
  # The real loop's guard (Low-v1's costs).
  cfg.rewards["held_at_cap"] = RewardTermCfg(
    func=lm.held_at_cap,
    weight=V1_HELD_WEIGHT,
    params={"ratio": 0.94, "grace": V1_HELD_GRACE},
  )
  cfg.rewards["saturated"] = RewardTermCfg(
    func=lm.saturated_share, weight=SATURATED_WEIGHT, params={"ratio": 0.94}
  )
  return cfg
