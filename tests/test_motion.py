"""Reference clips and the motion-imitation environment."""

import mjlab  # noqa: F401  (loads entry-point tasks)
import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

from walking_leap import clips, mdp, robot
from walking_leap.motion_cfg import TRACKED_BODIES

TASK = "Walking-Leap-Motion-Flat-v0"
NUM_ENVS = 4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

##
# Clips.
##


@pytest.fixture(scope="module")
def hello() -> tuple[clips.Motion, clips.Kinematics]:
  kin = clips.Kinematics()
  return clips.solve(clips.hello(), kin), kin


def test_home_is_the_folded_thumb_stance():
  home = clips.Kinematics().home
  for digit, names in (("mf", "mcp rot pip dip"), ("th", "cmc axl mcp ipl")):
    for q, name in zip(home[digit], names.split(), strict=True):
      key = f"th_{name}" if digit == "th" else rf"(if|mf|rf)_{name}"
      assert q == pytest.approx(robot.FOLDED_JOINT_POS[key], abs=0.005)


def test_hello_is_kinematically_sound(hello):
  report = clips.report(*hello)
  assert report["ik_error_mm"] < 0.1
  assert report["limit_margin_rad"] > 0.04
  assert report["peak_joint_speed_rad_s"] < 0.6 * robot.NO_LOAD_SPEED
  assert report["balance_margin_mm"] > 6.0
  assert report["lowest_lifted_foot_mm"] > -0.1
  assert report["frames_touching_itself"] == 0


def test_hello_file_is_up_to_date(hello):
  """assets/clips/hello.npz is what clips.py writes (rebuild it after an edit)."""
  motion, _ = hello
  saved = np.load(clips.CLIPS_DIR / "hello.npz")
  assert saved["fps"][0] == clips.FPS
  assert np.allclose(saved["joint_pos"], motion.qpos[:, 7:], atol=2e-3)
  assert np.allclose(saved["body_pos_w"][:, 0], motion.qpos[:, :3], atol=1e-4)
  # The joint velocities are the positions' slope.
  slope = np.gradient(saved["joint_pos"], 1 / clips.FPS, axis=0)
  assert np.allclose(saved["joint_vel"], slope, atol=1e-2)


##
# Environment.
##


@pytest.fixture(scope="module")
def env():
  cfg = load_env_cfg(TASK, play=True)  # Starts at frame 0, no noise.
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg, device=DEVICE)
  yield env
  env.close()


def reference_action(env) -> torch.Tensor:
  """The action whose joint targets are the clip's joints now."""
  term = env.action_manager.get_term("joint_pos")
  command = env.command_manager.get_term("motion")
  return (command.joint_pos - term._offset) / term._scale


def test_tasks_registered():
  assert {TASK, "Walking-Leap-Motion-Flat-v1"} <= set(list_tasks())


def test_actor_sees_only_encoders_and_imu(env):
  obs, _ = env.reset()
  assert env.observation_manager.active_terms["actor"] == [
    "reference_joints",
    "reference_gravity",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
  ]
  # 32 + 3 + 3 + 3 + 16 + 16 + 16.
  assert obs["actor"].shape == (NUM_ENVS, 89)
  n = len(TRACKED_BODIES)
  assert obs["critic"].shape == (NUM_ENVS, 89 + 3 + 3 + 6 + 9 * n)


def test_starts_on_the_reference(env):
  obs, _ = env.reset()
  r = env.scene["robot"]
  command = env.command_manager.get_term("motion")
  assert torch.allclose(r.data.joint_pos, command.joint_pos, atol=2e-3)
  assert torch.allclose(r.data.root_link_pos_w, command.anchor_pos_w, atol=2e-3)
  down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(NUM_ENVS, 3)
  assert torch.allclose(obs["actor"][:, 32:35], down, atol=1e-3)


def test_reference_gravity_follows_the_clip(env):
  """The bow tips the reference's nose down: its gravity gains +x."""
  env.reset()
  command = env.command_manager.get_term("motion")
  saved = np.load(clips.CLIPS_DIR / "hello.npz")
  pitch = 2.0 * saved["body_quat_w"][:, 0, 2]
  frame = int(pitch.argmax())
  command.time_steps[:] = frame
  gravity = mdp.motion_gravity(env, "motion")
  assert torch.all(gravity[:, 0] > 0.2)
  assert torch.allclose(gravity.norm(dim=-1), torch.ones(NUM_ENVS, device=env.device))


def test_playing_the_clip_stays_close(env):
  """First 5 s (push-ups, stretch): the reference's joints as targets, no learning."""
  env.reset()
  command = env.command_manager.get_term("motion")
  worst = 0.0
  for _ in range(250):
    obs, rew, terminated, _, _ = env.step(reference_action(env))
    assert torch.isfinite(obs["critic"]).all() and torch.isfinite(rew).all()
    assert not terminated.any()
    error = (command.anchor_pos_w - command.robot_anchor_pos_w).norm(dim=-1)
    worst = max(worst, float(error.max()))
  assert worst < 0.02
