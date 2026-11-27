"""Reference clips and the motion-imitation environment."""

import mjlab  # noqa: F401  (loads entry-point tasks)
import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.lab_api import math as mjmath

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


@pytest.mark.parametrize("name", ["gait", "gait_smooth"])
def test_gait_file_is_up_to_date(name):
  """assets/clips/gait.npz (and gait_smooth.npz, planned with gait.SMOOTH_IK) is
  gait.py's crawl replayed on the real hand's model, and the hand stays up for all of
  it. The smooth one's targets stay under the real loop's 6 rad/s."""
  hand = clips.RealHand()
  smooth = gait.SMOOTH_IK if name == "gait_smooth" else 0.0
  targets, velocity = gait.training_targets(hand.model, smooth)
  if smooth:
    assert np.abs(np.diff(targets, axis=0)).max() / gait.KEYFRAME_S < 6.0
  motion = clips.replay(name, hand, q=targets)
  saved = np.load(clips.CLIPS_DIR / f"{name}.npz")
  assert saved["fps"][0] == clips.FPS and len(saved["joint_pos"]) == len(motion.qpos)
  assert np.allclose(saved["joint_pos"], motion.qpos[:, 7:], atol=2e-3)
  assert np.allclose(saved["command"], velocity, atol=1e-6)  # planned velocity
  palm = saved["body_pos_w"][:, 0]
  assert palm[:, 2].min() > 0.10  # the palm stays up (it starts at 0.13 m)
  w, x, y, z = saved["body_quat_w"][:, 0].T
  assert (1 - 2 * (x * x + y * y)).min() > np.cos(np.radians(15.0))  # tilt < 15 deg


def test_gait_task_sees_the_imu():
  cfg = load_env_cfg(GAIT_TASK)
  assert cfg.commands["motion"].motion_file.endswith("gait.npz")
  actor = list(cfg.observations["actor"].terms)
  assert "base_ang_vel" in actor and "projected_gravity" in actor


##
# The gait task that survives falls (v1, mdp.GaitMotionCommand).
##

GAIT_V1_TASK = "Walking-Leap-Motion-Gait-Flat-v1"


def test_gait_v1_keeps_v0_observations():
  """Same inputs as v0, so v1 can continue from v0's checkpoints."""
  v0, v1 = load_env_cfg(GAIT_TASK), load_env_cfg(GAIT_V1_TASK)
  for group in ("actor", "critic"):
    assert list(v0.observations[group].terms) == list(v1.observations[group].terms)
  assert not {"palm_height", "palm_tilt", "foot_height"} & set(v1.terminations)


def test_gait_v1_clip_waits_while_fallen_and_resumes_where_the_hand_gets_up():
  cfg = load_env_cfg(GAIT_V1_TASK)
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = 0
  command_cfg = cfg.commands["motion"]
  command_cfg.fallen_start_prob = 1.0
  env = ManagerBasedRlEnv(cfg, device=DEVICE)
  try:
    env.reset()
    command = env.command_manager.get_term("motion")
    assert command.fallen.all()
    frame = command.time_steps.clone()
    for _ in range(10):
      env.step(torch.zeros(env.num_envs, 16, device=env.device))
    assert (command.time_steps == frame).all()  # the clip waited
    assert (command.planned_velocity == 0).all()
    # Put the hand back on its feet as the clip stands now, 5 cm aside, turned 30 deg.
    robot = env.scene["robot"]
    turn = torch.tensor([np.cos(np.pi / 12), 0, 0, np.sin(np.pi / 12)], device=DEVICE)
    turn = turn.repeat(NUM_ENVS, 1).float()
    root = torch.zeros(NUM_ENVS, 13, device=DEVICE)
    root[:, :3] = command.body_pos_w[:, 0] + torch.tensor(
      [0.05, 0.0, 0.0], device=DEVICE
    )
    root[:, 3:7] = mjmath.quat_mul(turn, command.body_quat_w[:, 0])
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(
      command.joint_pos, torch.zeros_like(command.joint_pos)
    )
    for _ in range(command._recover_steps + 2):
      env.step(reference_action(env))
    assert not command.fallen.any()
    assert (command.time_steps > frame).all()  # the clip goes on
    # It goes on from where the hand stood up, not where it fell.
    gap = (command.anchor_pos_w - command.robot_anchor_pos_w)[:, :2].norm(dim=-1)
    assert gap.max() < 0.02
  finally:
    env.close()


GAIT_STUDENT_TASK = "Walking-Leap-Motion-Gait-Flat-Student-v0"


def test_gait_student_sees_a_command_not_the_clip():
  """The student gets what the real hand has with a keyboard; the teacher keeps
  v1's actor inputs, so a v1 checkpoint is a teacher."""
  v1, student = load_env_cfg(GAIT_V1_TASK), load_env_cfg(GAIT_STUDENT_TASK)
  assert list(student.observations["teacher"].terms) == list(
    v1.observations["actor"].terms
  )
  terms = student.observations["student"].terms
  assert list(terms) == [
    "command",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
  ]
  assert student.observations["student"].enable_corruption
  play = load_env_cfg(GAIT_STUDENT_TASK, play=True)
  assert not play.observations["student"].enable_corruption


def test_gait_student_runner_takes_a_ppo_teacher_and_plays_the_student(tmp_path):
  """train --agent.resume loads a PPO (v1) actor as the teacher; play, loading
  {"actor": True}, gets the trained student."""
  from dataclasses import asdict

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg
  from rsl_rl.models import MLPModel

  from walking_leap.runner import GaitStudentRunner

  cfg = load_env_cfg(GAIT_STUDENT_TASK)
  cfg.scene.num_envs = 2
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=DEVICE))
  try:
    obs = env.get_observations()
    assert obs["student"].shape[-1] == 5 * (3 + 3 + 3 + 16 + 16 + 16)

    # A v1 actor, as PPO builds it, with weights of its own.
    actor_cfg = asdict(load_rl_cfg(GAIT_V1_TASK).actor)
    for key in (
      "class_name",
      "cnn_cfg",
      "rnn_type",
      "rnn_hidden_dim",
      "rnn_num_layers",
    ):
      actor_cfg.pop(key)
    actor = MLPModel(obs, {"actor": ["teacher"]}, "actor", 16, **actor_cfg)
    for p in actor.parameters():
      torch.nn.init.normal_(p)
    ppo = tmp_path / "model_5999.pt"
    torch.save({"actor_state_dict": actor.state_dict(), "iter": 5999, "infos": {}}, ppo)

    agent = asdict(load_rl_cfg(GAIT_STUDENT_TASK))
    runner = GaitStudentRunner(env, agent, str(tmp_path / "run"), DEVICE)
    runner.load(str(ppo))
    assert runner.alg.teacher_loaded
    assert runner.current_learning_iteration == 0  # a new training, not v1's
    x = obs.to(DEVICE)
    with torch.inference_mode():
      assert torch.allclose(runner.alg.teacher(x), actor.to(DEVICE)(x))

    for p in runner.alg.student.parameters():
      torch.nn.init.normal_(p)
    runner.save(str(tmp_path / "student.pt"))
    player = GaitStudentRunner(
      env, asdict(load_rl_cfg(GAIT_STUDENT_TASK)), None, DEVICE
    )
    player.load(str(tmp_path / "student.pt"), load_cfg={"actor": True})
    with torch.inference_mode():
      assert torch.allclose(
        player.get_inference_policy(DEVICE)(x), runner.alg.student(x)
      )
  finally:
    env.close()


GAIT_V2_TASK = "Walking-Leap-Motion-Gait-Flat-v2"
GETUP_TASK = "Walking-Leap-Motion-Gait-Flat-Getup-v0"
STUDENT_V1_TASK = "Walking-Leap-Motion-Gait-Flat-Student-v1"
VELOCITY_TASK = "Walking-Leap-Motion-Gait-Flat-Velocity-v0"


def test_keyboard_velocity_tasks_fit_together():
  """Gait-v2 continues from Gait-v1; the two-teacher student's groups are those
  teachers' actors; the velocity task's actor is the student, so it continues from it.
  Getup and the student never see the clip."""
  v1, v2 = load_env_cfg(GAIT_V1_TASK), load_env_cfg(GAIT_V2_TASK)
  getup, student = load_env_cfg(GETUP_TASK), load_env_cfg(STUDENT_V1_TASK)
  velocity = load_env_cfg(VELOCITY_TASK)
  velocity_v1 = load_env_cfg("Walking-Leap-Motion-Gait-Flat-Velocity-v1")
  assert list(velocity_v1.observations["actor"].terms) == list(
    velocity.observations["actor"].terms
  )  # v1 continues from v0
  assert {"track_lin_vel", "track_yaw_vel"} <= set(velocity_v1.rewards)
  velocity_v2 = load_env_cfg("Walking-Leap-Motion-Gait-Flat-Velocity-v2")
  assert list(velocity_v2.observations["actor"].terms) == list(
    velocity.observations["actor"].terms
  )  # v2 continues from v1
  velocity_v3 = load_env_cfg("Walking-Leap-Motion-Gait-Flat-Velocity-v3")
  assert list(velocity_v3.observations["actor"].terms) == list(
    velocity.observations["actor"].terms
  )  # v3 continues from v2
  actor = list(v1.observations["actor"].terms)
  assert list(v2.observations["actor"].terms) == actor
  assert list(student.observations["teacher"].terms) == actor
  getup_actor = list(getup.observations["actor"].terms)
  assert not any(t.startswith("reference") for t in getup_actor)
  assert list(student.observations["teacher_getup"].terms) == getup_actor
  for cfg, group in ((student, "student"), (velocity, "actor")):
    g = cfg.observations[group]
    assert list(g.terms) == ["command", *getup_actor] and g.history_length == 5
  assert not any(n.startswith(("body_", "palm_")) for n in velocity.rewards)


def test_slew_action_moves_the_setpoint_at_the_drivers_rate():
  """Targets asked to jump far: the servo setpoint follows at 5 rad/s at most (the real
  driver's 0.05 rad a 100 Hz tick)."""
  cfg = load_env_cfg(GAIT_V2_TASK, play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device=DEVICE)
  try:
    env.reset()
    term = env.action_manager.get_term("joint_pos")
    limit = term.cfg.max_rate * env.step_dt + 1e-5
    env.step(torch.zeros(2, 16, device=DEVICE))
    for sign in (1.0, -1.0, 1.0):
      before = term.setpoint.clone()
      env.step(torch.full((2, 16), 50.0 * sign, device=DEVICE))
      moved = (term.setpoint - before).abs()
      assert moved.max() <= limit and moved.max() > 0.5 * limit
  finally:
    env.close()
