"""Environment validation through mjlab (GPU if available, else CPU)."""

import math

import mjlab  # noqa: F401  (loads entry-point tasks)
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from walking_leap import mdp, robot
from walking_leap.env_cfg import DIGIT_CONTACT, TIP_CONTACT

TASK = "Walking-Leap-Flat-v0"
NUM_ENVS = 4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ACTOR_TERMS = [
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command",
]
PRIVILEGED_TERMS = [
  "base_lin_vel",
  "root_height",
  "digit_contact",
  "digit_contact_forces",
  "palm_contact",
]


@pytest.fixture(scope="module")
def env():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg, device=DEVICE)
  yield env
  env.close()


def zero_action(env) -> torch.Tensor:
  return torch.zeros(env.num_envs, 16, device=env.device)


def step_rewards(env, env_idx: int = 0) -> dict[str, float]:
  """Unscaled per-term reward rates (raw value * weight) of the last step."""
  return {k: v[0] for k, v in env.reward_manager.get_active_iterable_terms(env_idx)}


def test_task_registered():
  assert TASK in list_tasks()
  rl = load_rl_cfg(TASK)
  assert rl.actor.hidden_dims == (256, 128, 64)
  assert rl.critic.hidden_dims == (256, 128, 64)
  assert rl.obs_groups == {"actor": ("actor",), "critic": ("critic",)}


def test_spaces_and_observation_separation(env):
  obs, _ = env.reset()
  terms = env.observation_manager.active_terms
  assert terms["actor"] == ACTOR_TERMS
  assert terms["critic"] == ACTOR_TERMS + PRIVILEGED_TERMS
  assert not set(PRIVILEGED_TERMS) & set(terms["actor"])
  # 3 + 3 + 16 + 16 + 16 + 3 actor; + 3 + 1 + 4 + 12 + 1 privileged.
  assert obs["actor"].shape == (NUM_ENVS, 57)
  assert obs["critic"].shape == (NUM_ENVS, 78)
  assert torch.equal(obs["critic"][:, :57], obs["actor"])
  assert env.action_space.shape == (NUM_ENVS, 16)
  term = env.action_manager.get_term("joint_pos")
  assert tuple(term.target_names) == robot.JOINT_NAMES


def test_reset_to_nominal_pose(env):
  obs, _ = env.reset()
  r = env.scene["robot"]
  assert torch.allclose(r.data.joint_pos, r.data.default_joint_pos, atol=1e-6)
  assert torch.all(r.data.joint_vel == 0)
  g = r.data.projected_gravity_b
  assert torch.allclose(
    g, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand_as(g), atol=1e-5
  )
  z = r.data.root_link_pos_w[:, 2]
  assert torch.all(z >= robot.INIT_ROOT_HEIGHT - 1e-6)
  assert torch.all(z <= robot.INIT_ROOT_HEIGHT + 0.005 + 1e-6)
  xy = r.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  assert torch.allclose(xy, torch.zeros_like(xy), atol=1e-6)
  # Command: constant forward velocity in the palm frame.
  cmd = env.command_manager.get_command("twist")
  expected = torch.tensor([0.05, 0.0, 0.0], device=env.device)
  assert torch.allclose(cmd, expected.expand_as(cmd))


def test_zero_action_stands(env):
  env.reset()
  r = env.scene["robot"]
  xy0 = r.data.root_link_pos_w[:, :2].clone()
  for _ in range(100):  # 2 s
    obs, rew, terminated, truncated, _ = env.step(zero_action(env))
    assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
    assert torch.isfinite(rew).all()
    assert not terminated.any() and not truncated.any()
  # Settles on the tips ~1.6 mm below the reset height, in place.
  z = r.data.root_link_pos_w[:, 2]
  assert torch.all((z - robot.INIT_ROOT_HEIGHT).abs() < 0.003)
  assert torch.all((r.data.root_link_pos_w[:, :2] - xy0).norm(dim=-1) < 0.003)
  rewards = step_rewards(env)
  assert rewards["palm_contact"] == 0.0
  assert rewards["palm_height"] == pytest.approx(1.0)
  assert rewards["upright"] == pytest.approx(1.0, abs=1e-3)
  # Standing still against a 0.05 m/s command: exp(-1) of the weight (2.0).
  assert rewards["track_lin_vel"] == pytest.approx(2.0 * math.exp(-1.0), rel=0.05)
  assert rewards["joint_torques"] <= 0.0 and rewards["action_rate"] <= 0.0


def test_random_actions_finite_and_within_limits(env):
  env.reset()
  term = env.action_manager.get_term("joint_pos")
  limits = robot.joint_limits()
  lo = torch.tensor([limits[n][0] for n in robot.JOINT_NAMES], device=env.device)
  hi = torch.tensor([limits[n][1] for n in robot.JOINT_NAMES], device=env.device)
  gen = torch.Generator(device=env.device).manual_seed(0)
  for _ in range(150):
    action = 3.0 * torch.randn(env.num_envs, 16, device=env.device, generator=gen)
    obs, rew, *_ = env.step(action)
    assert torch.isfinite(obs["critic"]).all() and torch.isfinite(rew).all()
    targets = term._processed_actions
    assert torch.all(targets >= lo - 1e-6) and torch.all(targets <= hi + 1e-6)


def test_independent_reset_and_physical_termination(env):
  env.reset()
  for _ in range(5):
    env.step(zero_action(env))
  r = env.scene["robot"]
  before = r.data.root_link_pos_w[1:].clone()
  steps_before = env.episode_length_buf.clone()
  # Flip env 0 upside down: a physical failure, not a timeout.
  pose = r.data.root_link_pose_w[:1].clone()
  pose[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device)
  r.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.device))
  _, _, terminated, truncated, _ = env.step(zero_action(env))
  assert terminated[0] and not truncated[0]
  assert env.termination_manager.get_term("fell_over")[0]
  assert not terminated[1:].any()
  # Env 0 was reset; the others kept running undisturbed.
  assert env.episode_length_buf[0] == 0
  assert torch.all(env.episode_length_buf[1:] == steps_before[1:] + 1)
  assert torch.allclose(r.data.root_link_pos_w[1:], before, atol=2e-3)
  assert torch.allclose(r.data.joint_pos[0], r.data.default_joint_pos[0], atol=1e-6)


def test_time_limit_is_truncation(env):
  env.reset()
  env.episode_length_buf[:] = env.max_episode_length - 1
  _, _, terminated, truncated, _ = env.step(zero_action(env))
  assert truncated.all() and not terminated.any()
  assert torch.all(env.episode_length_buf == 0)


def test_palm_contact_penalty_and_termination(env):
  """Straight fingers drop the palm onto the ground: penalized, then ended."""
  env.reset()
  term = env.action_manager.get_term("joint_pos")
  flat = -term._offset / term._scale  # action whose joint target is 0 rad
  penalized, ended_at = False, None
  for k in range(150):
    _, _, terminated, truncated, _ = env.step(flat)
    if step_rewards(env)["palm_contact"] < 0:
      penalized = True
    if terminated[0]:
      ended_at = k
      assert env.termination_manager.get_term("palm_on_ground")[0]
      break
  assert penalized
  # Fires only after > 1 s (50 policy steps) of continuous palm contact.
  assert ended_at is not None and ended_at >= 50


def test_gait_metrics(env):
  """Per-digit contact and tip slip: planted and still at rest, exact when slid."""
  env.reset()
  for _ in range(50):  # 1 s
    env.step(zero_action(env))
  assert env.scene[TIP_CONTACT].primary_names == list(robot.FINGERTIP_BODIES)
  for body in robot.FINGER_ROOT_BODIES:
    assert torch.all(mdp.contact_active(env, DIGIT_CONTACT, primary=body) == 1.0)
  assert torch.all(env.scene[TIP_CONTACT].data.found > 0)  # on the four pads
  assert torch.all(mdp.contact_slip_speed(env, TIP_CONTACT) < 1e-3)

  # Move the standing hand rigidly (joints still): every pad slides with it.
  r = env.scene["robot"]
  vel = torch.zeros(env.num_envs, 6, device=env.device)
  vel[:, 0] = 0.3  # m/s along world x
  r.write_root_link_velocity_to_sim(vel)
  env.sim.forward()
  slip = mdp.contact_slip_speed(env, TIP_CONTACT)
  assert torch.allclose(slip, torch.full_like(slip, 0.3), atol=2e-3)

  # Spin about the palm's vertical axis: each pad slides at w * horizontal
  # distance from the palm origin, so the mean is w times the mean distance.
  vel[:, 0], vel[:, 5] = 0.0, 1.0  # rad/s about world z
  r.write_root_link_velocity_to_sim(vel)
  env.sim.forward()
  pos = env.scene[TIP_CONTACT].data.pos
  radius = (pos[..., :2] - r.data.root_link_pos_w[:, None, :2]).norm(dim=-1)
  slip = mdp.contact_slip_speed(env, TIP_CONTACT)
  assert torch.allclose(slip, radius.mean(dim=-1), atol=2e-3)
  env.reset()
