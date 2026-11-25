"""LEAP Hand finger-walking tasks for mjlab.

Imported by mjlab through the ``mjlab.tasks`` entry point in pyproject.toml.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from walking_leap.agent_cfg import (
  walking_leap_motion_ppo_runner_cfg,
  walking_leap_ppo_runner_cfg,
)
from walking_leap.env_cfg import walking_leap_flat_env_cfg
from walking_leap.motion_cfg import walking_leap_motion_env_cfg

register_mjlab_task(
  task_id="Walking-Leap-Flat-v0",
  env_cfg=walking_leap_flat_env_cfg(),
  play_env_cfg=walking_leap_flat_env_cfg(play=True),
  rl_cfg=walking_leap_ppo_runner_cfg(),
)

# The hand as it is being built: feet, IMU, 350 mA, folded-thumb stance.
register_mjlab_task(
  task_id="Walking-Leap-Flat-v1",
  env_cfg=walking_leap_flat_env_cfg(as_built=True),
  play_env_cfg=walking_leap_flat_env_cfg(play=True, as_built=True),
  rl_cfg=walking_leap_ppo_runner_cfg(),
)

# The same with rewards for a gait the real hand could do.
register_mjlab_task(
  task_id="Walking-Leap-Flat-v2",
  env_cfg=walking_leap_flat_env_cfg(as_built=True, gait_terms=True),
  play_env_cfg=walking_leap_flat_env_cfg(play=True, as_built=True, gait_terms=True),
  rl_cfg=walking_leap_ppo_runner_cfg(),
)

# Motion imitation with the same hand: v0 nominal, v1 with kicks, cables, noise and
# random grip, balance and motors.
for version, robust in (("v0", False), ("v1", True)):
  register_mjlab_task(
    task_id=f"Walking-Leap-Motion-Flat-{version}",
    env_cfg=walking_leap_motion_env_cfg(robust=robust, cables=robust),
    play_env_cfg=walking_leap_motion_env_cfg(play=True, robust=robust, cables=robust),
    rl_cfg=walking_leap_motion_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )
