"""LEAP Hand finger-walking tasks for mjlab.

Imported by mjlab through the ``mjlab.tasks`` entry point in pyproject.toml.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from walking_leap.agent_cfg import (
  walking_leap_crawl_ppo_runner_cfg,
  walking_leap_gait_student_runner_cfg,
  walking_leap_motion_ppo_runner_cfg,
  walking_leap_ppo_runner_cfg,
  walking_leap_real_ppo_runner_cfg,
  walking_leap_two_teacher_runner_cfg,
  walking_leap_velocity_ppo_runner_cfg,
  walking_leap_velocity_v2_ppo_runner_cfg,
  walking_leap_velocity_v3_ppo_runner_cfg,
)
from walking_leap.env_cfg import (
  walking_leap_flat_env_cfg,
  walking_leap_flat_omni_env_cfg,
  walking_leap_flat_real_env_cfg,
)
from walking_leap.motion_cfg import (
  walking_leap_gait_env_cfg,
  walking_leap_gait_student_env_cfg,
  walking_leap_gait_v2_env_cfg,
  walking_leap_gait_v3_env_cfg,
  walking_leap_getup_env_cfg,
  walking_leap_motion_env_cfg,
  walking_leap_two_teacher_student_env_cfg,
  walking_leap_velocity_env_cfg,
  walking_leap_velocity_v1_env_cfg,
  walking_leap_velocity_v2_env_cfg,
  walking_leap_velocity_v3_env_cfg,
  walking_leap_velocity_v4_env_cfg,
  walking_leap_velocity_v5_env_cfg,
)
from walking_leap.robot import MEASURED_STIFFNESS, TORSION
from walking_leap.runner import (
  ExploringStudentPpoRunner,
  GaitStudentRunner,
  StudentPpoRunner,
)

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

# The same, surviving: everything random, pushes, cables, falls that do not end the
# episode (the clip waits and resumes where the hand gets up), a fifth of the episodes
# started fallen, a reward for moving as planned. Same observations as v0.
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-v1",
  env_cfg=walking_leap_gait_env_cfg(),
  play_env_cfg=walking_leap_gait_env_cfg(play=True),
  rl_cfg=walking_leap_crawl_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# A student of Gait-v1 that walks from a velocity command: it copies a v1 checkpoint
# (the teacher, loaded with --agent.resume) seeing only the plan's velocity, the IMU,
# the encoders and its last actions, never the clip. Teacher-student distillation.
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-Student-v0",
  env_cfg=walking_leap_gait_student_env_cfg(),
  play_env_cfg=walking_leap_gait_student_env_cfg(play=True),
  rl_cfg=walking_leap_gait_student_runner_cfg(),
  runner_cls=GaitStudentRunner,
)

# The walking teacher for the keyboard-velocity policy (docs/velocity_policy.md): Gait-v1
# on the smooth clip (gait_smooth.npz: targets the real loop accepts), with the real
# driver's setpoint limit and pushes up to 0.4 m/s per axis; continues from Gait-v1.
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-v2",
  env_cfg=walking_leap_gait_v2_env_cfg(),
  play_env_cfg=walking_leap_gait_v2_env_cfg(play=True),
  rl_cfg=walking_leap_crawl_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# The getting-up teacher: every episode starts fallen; IMU and encoders only.
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-Getup-v0",
  env_cfg=walking_leap_getup_env_cfg(),
  play_env_cfg=walking_leap_getup_env_cfg(play=True),
  rl_cfg=walking_leap_motion_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# The keyboard-velocity student of Gait-v2 (standing) and Getup (fallen): pass the merged
# teachers (scripts/merge_teachers.py) with --agent.resume.
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-Student-v1",
  env_cfg=walking_leap_two_teacher_student_env_cfg(),
  play_env_cfg=walking_leap_two_teacher_student_env_cfg(play=True),
  rl_cfg=walking_leap_two_teacher_runner_cfg(),
  runner_cls=GaitStudentRunner,
)

# PPO on the student's own task: random keyboard commands (combined, changed any time),
# getting up, pushes; continues from Student-v1 (--agent.resume on its checkpoint).
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-Velocity-v0",
  env_cfg=walking_leap_velocity_env_cfg(),
  play_env_cfg=walking_leap_velocity_env_cfg(play=True),
  rl_cfg=walking_leap_velocity_ppo_runner_cfg(),
  runner_cls=StudentPpoRunner,
)

# Velocity-v0 with the velocity reward split (translation and turn on their own), the
# scales Flat-v3 walked with; continues from Velocity-v0.
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-Velocity-v1",
  env_cfg=walking_leap_velocity_v1_env_cfg(),
  play_env_cfg=walking_leap_velocity_v1_env_cfg(play=True),
  rl_cfg=walking_leap_velocity_ppo_runner_cfg(),
  runner_cls=StudentPpoRunner,
)

# Velocity-v1 with more weight on the translation, wider noise (0.3) and mjlab's learning
# rate: v1 kept standing (forward 4.3 of 30 mm/s). Continues from v1, or from scratch.
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-Velocity-v2",
  env_cfg=walking_leap_velocity_v2_env_cfg(),
  play_env_cfg=walking_leap_velocity_v2_env_cfg(play=True),
  rl_cfg=walking_leap_velocity_v2_ppo_runner_cfg(),
  runner_cls=ExploringStudentPpoRunner,
)

# Velocity-v2 with light pushes (0.1 m/s) and a larger entropy bonus: first walk fast,
# then push harder. Continues from v2.
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-Velocity-v3",
  env_cfg=walking_leap_velocity_v3_env_cfg(),
  play_env_cfg=walking_leap_velocity_v3_env_cfg(play=True),
  rl_cfg=walking_leap_velocity_v3_ppo_runner_cfg(),
  runner_cls=ExploringStudentPpoRunner,
)

# A diagnosis: Velocity-v3 with no perturbation at all (no pushes, cables, random
# physics or noise), as Flat-v3 walked. Continues from v2.
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-Velocity-v4",
  env_cfg=walking_leap_velocity_v4_env_cfg(),
  play_env_cfg=walking_leap_velocity_v4_env_cfg(play=True),
  rl_cfg=walking_leap_velocity_v3_ppo_runner_cfg(),
  runner_cls=ExploringStudentPpoRunner,
)

# Curriculum, step one: Gait-v2 with light pushes (Gait-v1's), on the smooth clip and
# the driver's limit; continues from Gait-v1. Velocity-v0..v4 barely translated.
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-v3",
  env_cfg=walking_leap_gait_v3_env_cfg(),
  play_env_cfg=walking_leap_gait_v3_env_cfg(play=True),
  rl_cfg=walking_leap_crawl_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# The keyboard policy taught by imitation: the command is the clip's planned velocity,
# the rewards are the clip's (palm path, digits' poses); continues from the velocity
# policies (it keeps getting up and turning, and learns to translate as the clip does).
register_mjlab_task(
  task_id="Walking-Leap-Motion-Gait-Flat-Velocity-v5",
  env_cfg=walking_leap_velocity_v5_env_cfg(),
  play_env_cfg=walking_leap_velocity_v5_env_cfg(play=True),
  rl_cfg=walking_leap_velocity_v3_ppo_runner_cfg(),
  runner_cls=ExploringStudentPpoRunner,
)
