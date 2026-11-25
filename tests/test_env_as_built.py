"""Walking environment with the as-built hand and the gait terms (v2; v1 is a subset)."""

import mjlab  # noqa: F401  (loads entry-point tasks)
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

from walking_leap import mdp, robot
from walking_leap.env_cfg import DIGIT_CONTACT

TASK = "Walking-Leap-Flat-v2"
NUM_ENVS = 4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def env():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg, device=DEVICE)
  yield env
  env.close()


def rewards(env, env_idx: int = 0) -> dict[str, float]:
  return {k: v[0] for k, v in env.reward_manager.get_active_iterable_terms(env_idx)}


def test_same_observations_as_the_stock_hand(env):
  assert {"Walking-Leap-Flat-v1", TASK} <= set(list_tasks())
  obs, _ = env.reset()
  assert obs["actor"].shape == (NUM_ENVS, 57)
  assert obs["critic"].shape == (NUM_ENVS, 78)


def test_stands_in_the_folded_stance(env):
  env.reset()
  r = env.scene["robot"]
  xy0 = r.data.root_link_pos_w[:, :2].clone()
  for _ in range(100):  # 2 s
    _, _, terminated, truncated, _ = env.step(
      torch.zeros(NUM_ENVS, 16, device=env.device)
    )
    assert not terminated.any() and not truncated.any()
  z = r.data.root_link_pos_w[:, 2]
  assert torch.all((z - robot.FOLDED_ROOT_HEIGHT).abs() < 0.004)
  assert torch.all((r.data.root_link_pos_w[:, :2] - xy0).norm(dim=-1) < 0.004)
  terms = rewards(env)
  assert terms["footprint"] == pytest.approx(1.0, abs=0.03)
  assert terms["saturation"] == 0.0
  assert abs(terms["feet_slip"]) < 1e-3
  # Every foot is down and stays there.
  assert terms["feet_down"] == 1.0
  assert terms["short_steps"] == 0.0 and terms["feet_held_up"] == 0.0


def test_feet_down_needs_three_feet(env):
  """Held in the air the hand has four feet up."""
  env.reset()
  assert torch.all(mdp.feet_down(env, DIGIT_CONTACT, allowed_up=4) == 1.0)
  r = env.scene["robot"]
  pose = r.data.root_link_pose_w.clone()
  pose[:, 2] += 0.3
  r.write_root_link_pose_to_sim(pose)
  env.step(torch.zeros(NUM_ENVS, 16, device=env.device))
  assert torch.all(mdp.feet_down(env, DIGIT_CONTACT, allowed_up=1) == 0.0)
  assert torch.all(mdp.feet_down(env, DIGIT_CONTACT, allowed_up=3) == 0.0)
  assert torch.all(mdp.feet_down(env, DIGIT_CONTACT, allowed_up=4) == 1.0)


def test_a_foot_held_up_costs(env):
  """In free fall from 3 m all four feet are up, and count after half a second."""
  env.reset()
  r = env.scene["robot"]
  pose = r.data.root_link_pose_w.clone()
  pose[:, 2] += 3.0
  r.write_root_link_pose_to_sim(pose)
  held = []
  for _ in range(30):  # 0.6 s, 1.8 m
    env.step(torch.zeros(NUM_ENVS, 16, device=env.device))
    held.append(mdp.feet_held_up(env, DIGIT_CONTACT, 0.5).clone())
  assert torch.all(held[20] == 0.0)
  assert torch.all(held[-1] == 4.0)


def test_footprint_drops_when_a_foot_leaves_its_place(env):
  env.reset()
  action = torch.zeros(NUM_ENVS, 16, device=env.device)
  action[:, robot.JOINT_NAMES.index("if_pip")] = -3.0  # Straighten the index.
  action[:, robot.JOINT_NAMES.index("if_mcp")] = -3.0
  for _ in range(25):
    env.step(action)
  assert rewards(env)["footprint"] < 0.8
