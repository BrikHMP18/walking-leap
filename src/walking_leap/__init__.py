"""LEAP Hand finger-walking tasks for mjlab.

Imported by mjlab through the ``mjlab.tasks`` entry point in pyproject.toml.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from walking_leap.agent_cfg import (
  walking_leap_crawl_ppo_runner_cfg,
  walking_leap_motion_ppo_runner_cfg,
  walking_leap_ppo_runner_cfg,
  walking_leap_real_ppo_runner_cfg,
)
from walking_leap.env_cfg import (
  walking_leap_flat_env_cfg,
  walking_leap_flat_omni_env_cfg,
  walking_leap_flat_real_env_cfg,
)
from walking_leap.motion_cfg import walking_leap_motion_env_cfg
from walking_leap.robot import MEASURED_STIFFNESS, TORSION

register_mjlab_task(
  task_id="Walking-Leap-Flat-v0",
  env_cfg=walking_leap_flat_env_cfg(),
  play_env_cfg=walking_leap_flat_env_cfg(play=True),
  rl_cfg=walking_leap_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Walking-Leap-Flat-v1",
  env_cfg=walking_leap_flat_omni_env_cfg(),
  play_env_cfg=walking_leap_flat_omni_env_cfg(play=True),
  rl_cfg=walking_leap_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Walking-Leap-Flat-NoImu-v1",
  env_cfg=walking_leap_flat_omni_env_cfg(imu=False),
  play_env_cfg=walking_leap_flat_omni_env_cfg(play=True, imu=False),
  rl_cfg=walking_leap_ppo_runner_cfg(),
)

# Motion imitation with the hand as it is being built (feet, IMU, 350 mA, folded-thumb
# stance): v0 nominal, v1 with kicks, cables, noise and random grip, balance and motors.
for version, robust in (("v0", False), ("v1", True)):
  register_mjlab_task(
    task_id=f"Walking-Leap-Motion-Flat-{version}",
    env_cfg=walking_leap_motion_env_cfg(robust=robust, cables=robust),
    play_env_cfg=walking_leap_motion_env_cfg(play=True, robust=robust, cables=robust),
    rl_cfg=walking_leap_motion_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )

# The real hand as it runs now (IMU, stock fingertips, 350 mA, measured stiffness,
# torsional friction), with and without the palm IMU in the actor: walking from
# rewards only (Flat-v3; v2 had unbounded penalties and learned to fall) and imitating the real crawl (Motion-Crawl-Flat-v0).
for variant, imu in (("", True), ("-NoImu", False)):
  register_mjlab_task(
    task_id=f"Walking-Leap-Flat{variant}-v3",
    env_cfg=walking_leap_flat_real_env_cfg(imu=imu),
    play_env_cfg=walking_leap_flat_real_env_cfg(play=True, imu=imu),
    rl_cfg=walking_leap_real_ppo_runner_cfg(),
  )
  crawl = dict(
    clip="crawl", feet=False, imu=imu, stiffness=MEASURED_STIFFNESS, torsion=TORSION
  )
  register_mjlab_task(
    task_id=f"Walking-Leap-Motion-Crawl-Flat{variant}-v0",
    env_cfg=walking_leap_motion_env_cfg(**crawl),
    play_env_cfg=walking_leap_motion_env_cfg(play=True, **crawl),
    rl_cfg=walking_leap_crawl_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )

# Imitating the scripted crawl of gait.py (assets/clips/gait.npz: one cycle per command,
# forward, back, left, right, turns and stand, 150 of them at random), on the real
# hand with the palm IMU, nothing random yet.
gait_hand = dict(clip="gait", feet=False, stiffness=MEASURED_STIFFNESS, torsion=TORSION)
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-v0",
  env_cfg=walking_leap_motion_env_cfg(**gait_hand),
  play_env_cfg=walking_leap_motion_env_cfg(play=True, **gait_hand),
  rl_cfg=walking_leap_crawl_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
