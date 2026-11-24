"""LEAP Hand finger-walking tasks for mjlab.

Imported by mjlab through the ``mjlab.tasks`` entry point in pyproject.toml.
"""

from mjlab.tasks.registry import register_mjlab_task

from leap_walking.agent_cfg import leap_walking_ppo_runner_cfg
from leap_walking.env_cfg import leap_walking_flat_env_cfg

register_mjlab_task(
  task_id="Leap-Walking-Flat-v0",
  env_cfg=leap_walking_flat_env_cfg(),
  play_env_cfg=leap_walking_flat_env_cfg(play=True),
  rl_cfg=leap_walking_ppo_runner_cfg(),
)
