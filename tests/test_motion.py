"""Reference clips and the motion-imitation environment."""

import mjlab  # noqa: F401  (loads entry-point tasks)
import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

from walking_leap import clips, gait, mdp, robot
from walking_leap.motion_cfg import (
  CABLE_BODIES,
  CABLE_FORCE,
  CABLE_TORQUE,
  TRACKED_BODIES,
)

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


def test_cables_pull_and_change():
  """v1: a steady wrench on the mount and the palm, within its range, that changes."""
  cfg = load_env_cfg("Walking-Leap-Motion-Flat-v1")
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg, device=DEVICE)
  try:
    env.reset()
    r = env.scene["robot"]
    ids = r.find_bodies(CABLE_BODIES)[0]
    others = [i for i in range(r.num_bodies) if i not in ids]
    seen = []
    for k in range(260):  # 5.2 s, longer than the longest interval.
      wrench = r.data.body_external_wrench[:, ids]
      if k % 20 == 0:
        seen.append(wrench.clone())
      env.step(reference_action(env))
      assert torch.all(wrench[..., :3].abs() <= CABLE_FORCE)
      assert torch.all(wrench[..., 3:].abs() <= CABLE_TORQUE)
      assert torch.all(r.data.body_external_wrench[:, others] == 0.0)
    assert torch.all(seen[0].abs().sum(dim=-1) > 0.0)
    assert not torch.allclose(seen[0], seen[-1])
  finally:
    env.close()


def test_play_has_no_disturbance():
  assert not {"push", "cables", "cables_at_reset"} & set(
    load_env_cfg("Walking-Leap-Motion-Flat-v1", play=True).events
  )


##
# The real crawl (docs/data/real_clips/crawl.npz) on the real hand's model.
##

CRAWL_TASK = "Walking-Leap-Motion-Crawl-Flat-v0"
CRAWL_NO_IMU_TASK = "Walking-Leap-Motion-Crawl-Flat-NoImu-v0"
GAIT_TASK = "Walking-Leap-Motion-Gait-Flat-v0"


def test_crawl_file_is_up_to_date():
  """assets/clips/crawl.npz is the real crawl replayed on the real hand's model."""
  motion = clips.replay("crawl", clips.RealHand())
  saved = np.load(clips.CLIPS_DIR / "crawl.npz")
  targets = np.load(clips.REAL_CLIPS_DIR / "crawl.npz")["q"]
  assert saved["fps"][0] == clips.FPS and len(saved["joint_pos"]) == len(targets)
  assert np.allclose(saved["joint_pos"], motion.qpos[:, 7:], atol=2e-3)
  assert np.allclose(saved["body_pos_w"][:, 0], motion.qpos[:, :3], atol=1e-4)
  # It crawls forward (palm +x at the start) about 10 cm in its 3 cycles.
  moved = saved["body_pos_w"][-1, 0, :2] - saved["body_pos_w"][0, 0, :2]
  assert 0.08 < moved[0] < 0.13 and abs(moved[1]) < 0.02


def test_crawl_tasks_track_the_digits_last_links():
  for task, imu_terms in ((CRAWL_TASK, True), (CRAWL_NO_IMU_TASK, False)):
    cfg = load_env_cfg(task)
    command = cfg.commands["motion"]
    assert command.motion_file.endswith("crawl.npz")
    assert not any(b.endswith("_foot") for b in command.body_names)
    actor = list(cfg.observations["actor"].terms)
    assert ("base_ang_vel" in actor) == imu_terms
    assert ("projected_gravity" in actor) == imu_terms
    assert "reference_gravity" in actor  # the clip's tilt, not a sensor
    assert "projected_gravity" in cfg.observations["critic"].terms
    hand = cfg.scene.entities["robot"]
    assert hand.init_state.pos[2] == robot.FOLDED_ROOT_HEIGHT_STOCK


@pytest.mark.parametrize("task", [CRAWL_TASK, GAIT_TASK])
def test_crawl_as_targets_stays_close(task):
  """5 s of the clip's joints as targets, no learning: near the reference."""
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg, device=DEVICE)
  try:
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
  finally:
    env.close()


##
# The scripted crawl of gait.py, one cycle per command.
##


@pytest.fixture(scope="module")
def crawler():
  return gait.Crawler(clips.RealHand().model, thumb_drag=True)


@pytest.mark.parametrize(
  "name", ["forward", "back", "left", "right", "turn_left", "turn_right"]
)
def test_gait_commands_move_the_plan(crawler, name):
  """Planned (kinematics), each command moves the hand by about its step a cycle."""
  q = crawler.start.copy()
  _, q = crawler.cycle(q, np.zeros(3), drag_s=gait.FIRST_DRAG_S)
  d = crawler.ik.kin(q)
  com0, yaw0 = d.subtree_com[crawler.palm][:2].copy(), crawler.heading(d)
  for _ in range(3):
    _, q = crawler.cycle(q, np.array(gait.GAIT_COMMANDS[name]))
  d = crawler.ik.kin(q)
  moved = crawler.rot(-yaw0) @ (d.subtree_com[crawler.palm][:2] - com0) / 3
  turned = (crawler.heading(d) - yaw0) / 3
  forward, left, turn = gait.GAIT_COMMANDS[name]
  assert abs(moved[0] - forward) < 0.005 and abs(moved[1] - left) < 0.005
  assert abs(turned - turn) < np.radians(2.0)


def test_gait_file_is_up_to_date():
  """assets/clips/gait.npz is gait.py's crawl replayed on the real hand's model, and
  the hand stays up for all of it."""
  hand = clips.RealHand()
  motion = clips.replay("gait", hand, q=gait.training_targets(hand.model))
  saved = np.load(clips.CLIPS_DIR / "gait.npz")
  assert saved["fps"][0] == clips.FPS and len(saved["joint_pos"]) == len(motion.qpos)
  assert np.allclose(saved["joint_pos"], motion.qpos[:, 7:], atol=2e-3)
  palm = saved["body_pos_w"][:, 0]
  assert palm[:, 2].min() > 0.10  # the palm stays up (it starts at 0.13 m)
  w, x, y, z = saved["body_quat_w"][:, 0].T
  assert (1 - 2 * (x * x + y * y)).min() > np.cos(np.radians(15.0))  # tilt < 15 deg


def test_gait_task_sees_the_imu():
  cfg = load_env_cfg(GAIT_TASK)
  assert cfg.commands["motion"].motion_file.endswith("gait.npz")
  actor = list(cfg.observations["actor"].terms)
  assert "base_ang_vel" in actor and "projected_gravity" in actor
