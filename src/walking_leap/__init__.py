"""LEAP Hand finger-walking tasks for mjlab.

Imported by mjlab through the ``mjlab.tasks`` entry point in pyproject.toml.
"""

from mjlab.tasks.registry import register_mjlab_task

from walking_leap.agent_cfg import walking_leap_ppo_runner_cfg
from walking_leap.env_cfg import walking_leap_flat_env_cfg

register_mjlab_task(
  task_id="Walking-Leap-Flat-v0",
  env_cfg=walking_leap_flat_env_cfg(),
  play_env_cfg=walking_leap_flat_env_cfg(play=True),
  rl_cfg=walking_leap_ppo_runner_cfg(),
)
