"""Rewards and terminations for crawling lying down (``Walking-Leap-Lying-*``).

Every term is bounded, so no cost can be escaped by ending the episode, and distance is
paid as net progress capped at the command, so nothing pays per event:
- ``capped_progress`` / ``capped_turn``: the palm's speed (yaw rate) the commanded way,
  capped at the command and floored at minus it.
- ``speed_error`` / ``turn_error``: the tracking error, linear, capped.
- ``still``: the palm moving or turning while the command is zero, capped.
- ``held_at_cap``: the real loop's guard (a motor at its cap for 1.5 s stops a run).
- ``palm_above``: the palm's height over its lying height, beyond a margin, capped.
- ``rate_bounded``: the action rate, capped.
- ``stood_up``: the palm risen well above lying (termination); flipping is mjlab's
  ``bad_orientation``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

_ROBOT = SceneEntityCfg("robot")


def capped_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_speed: float,
  reference_speed: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """The palm's speed along the commanded direction (palm frame), capped at the
  commanded speed and floored at minus it, over ``reference_speed``. Zero for a
  planar command under ``min_speed``."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)[:, :2]
  speed = torch.norm(cmd, dim=-1)
  along = (asset.data.root_link_lin_vel_b[:, :2] * cmd).sum(dim=-1) / speed.clamp(
    min=1e-6
  )
  capped = torch.minimum(torch.maximum(along, -speed), speed)
  return torch.where(speed >= min_speed, capped, 0.0) / reference_speed


def capped_turn(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_rate: float,
  reference_rate: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """The palm's yaw rate the commanded way, capped at the commanded rate and floored at
  minus it, over ``reference_rate``; zero for a turn under ``min_rate``."""
  asset: Entity = env.scene[asset_cfg.name]
  want = env.command_manager.get_command(command_name)[:, 2]
  rate = asset.data.root_link_ang_vel_b[:, 2] * torch.sign(want)
  capped = torch.minimum(torch.maximum(rate, -want.abs()), want.abs())
  return torch.where(want.abs() >= min_rate, capped, 0.0) / reference_rate


def speed_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  reference_speed: float,
  cap: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """|v_xy - command_xy| (palm frame) over ``reference_speed``, at most ``cap``.
  Negative weight."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)[:, :2]
  err = torch.norm(asset.data.root_link_lin_vel_b[:, :2] - cmd, dim=-1)
  return (err / reference_speed).clamp(max=cap)


def turn_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  reference_rate: float,
  cap: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """|yaw rate - commanded| over ``reference_rate``, at most ``cap``. Negative weight."""
  asset: Entity = env.scene[asset_cfg.name]
  want = env.command_manager.get_command(command_name)[:, 2]
  err = (asset.data.root_link_ang_vel_b[:, 2] - want).abs()
  return (err / reference_rate).clamp(max=cap)


def still(
  env: ManagerBasedRlEnv,
  command_name: str,
  reference_speed: float,
  reference_rate: float,
  min_speed: float,
  min_rate: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """On a zero command: the palm's planar speed and yaw rate, each over its reference
  and capped at 1 (0 to 2). Zero otherwise. Negative weight."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  zero = (torch.norm(cmd[:, :2], dim=-1) < min_speed) & (cmd[:, 2].abs() < min_rate)
  v = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=-1) / reference_speed
  w = asset.data.root_link_ang_vel_b[:, 2].abs() / reference_rate
  return torch.where(zero, v.clamp(max=1.0) + w.clamp(max=1.0), 0.0)


class held_at_cap:
  """The real loop's guard as a cost: each motor's time at ``ratio`` of its own torque
  cap or more, beyond ``grace`` s, up to 1 s; the worst motor counts (0 to 1). The loop
  on the real hand stops a run when a motor stays over 330 mA (of the 350 cap: ``ratio``
  0.94) for 1.5 s. Negative weight."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.asset: Entity = env.scene["robot"]
    self.t = torch.zeros(
      env.num_envs, self.asset.data.actuator_force.shape[-1], device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.t[slice(None) if env_ids is None else env_ids] = 0.0

  def __call__(
    self, env: ManagerBasedRlEnv, ratio: float, grace: float
  ) -> torch.Tensor:
    force = self.asset.data.actuator_force.abs()
    cap = env.sim.model.actuator_forcerange[..., 1]
    if cap.dim() == 1:
      cap = cap[None]
    at_cap = force >= ratio * cap[..., : force.shape[-1]]
    self.t = torch.where(at_cap, self.t + env.step_dt, 0.0)
    return (self.t - grace).clamp(0.0, 1.0).max(dim=-1).values


def palm_above(
  env: ManagerBasedRlEnv,
  height: float,
  scale: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """How far the palm's origin is over ``height`` (m), over ``scale``, at most 1.
  Negative weight."""
  asset: Entity = env.scene[asset_cfg.name]
  over = asset.data.root_link_pos_w[:, 2] - height
  return (over / scale).clamp(0.0, 1.0)


def rate_bounded(env: ManagerBasedRlEnv, cap: float) -> torch.Tensor:
  """mjlab's action rate (sum of squared changes), at most ``cap``. Negative weight."""
  am = env.action_manager
  return torch.sum(torch.square(am.action - am.prev_action), dim=1).clamp(max=cap)


def stood_up(
  env: ManagerBasedRlEnv, height: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """The palm's origin above ``height`` (m): the hand got up. A termination."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] > height


def saturated_share(env: ManagerBasedRlEnv, ratio: float) -> torch.Tensor:
  """The share of motors at ``ratio`` of their own torque cap or more now (0 to 1).
  Negative weight."""
  asset: Entity = env.scene["robot"]
  force = asset.data.actuator_force.abs()
  cap = env.sim.model.actuator_forcerange[..., 1]
  if cap.dim() == 1:
    cap = cap[None]
  return (force >= ratio * cap[..., : force.shape[-1]]).float().mean(dim=-1)


def stood_up_level(
  env: ManagerBasedRlEnv,
  height: float,
  level_deg: float,
  high: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Got up: the palm's origin above ``height`` and within ``level_deg`` of level (on its
  fingertips, standing), or above ``high`` however tilted. A pressing crawl pitches the
  front up 25-40 deg, its origin 35-60 mm over lying: not this. A termination."""
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.root_link_pos_w[:, 2]
  g = asset.data.projected_gravity_b
  tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1.0, 1.0)))
  return ((z > height) & (tilt < level_deg)) | (z > high)


class held_far:
  """Each motor's time with its setpoint more than ``error`` rad from its joint, beyond
  ``grace`` s, up to 1 s; the worst motor counts (0 to 1). The real loop's guard reads
  the current, which a far setpoint drives to the limit (the twin's 1.6 N*m/rad saturates
  at 0.11 rad); this reads the positions, whatever the contacts' softness: Lying v0 held
  four motors pressed into the table or each other on the twin, at their limit, while
  training's softer contact barely saturated them. Negative weight."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.asset: Entity = env.scene["robot"]
    self.t = torch.zeros_like(self.asset.data.joint_pos)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.t[slice(None) if env_ids is None else env_ids] = 0.0

  def __call__(
    self, env: ManagerBasedRlEnv, error: float, grace: float
  ) -> torch.Tensor:
    data = self.asset.data
    far = (data.joint_pos_target - data.joint_pos).abs() > error
    self.t = torch.where(far, self.t + env.step_dt, 0.0)
    return (self.t - grace).clamp(0.0, 1.0).max(dim=-1).values


def far_setpoints(
  env: ManagerBasedRlEnv, error: float, span: float, worst: bool = False
) -> torch.Tensor:
  """Now, the motors' setpoints beyond ``error`` rad from their joints: each one's excess
  over ``span``, capped at 1, averaged over the motors (0 to 1). ``held_far`` alone never
  fired in training: the exploration noise keeps any setpoint from staying far for long,
  while the deployed policy (its mean) holds it there. ``worst``: the worst motor's
  instead of the average (one motor pressing into the table is what stops the loop).
  Negative weight."""
  data = env.scene["robot"].data
  excess = ((data.joint_pos_target - data.joint_pos).abs() - error) / span
  excess = excess.clamp(0.0, 1.0)
  return excess.max(dim=-1).values if worst else excess.mean(dim=-1)


def _goal_off(env: ManagerBasedRlEnv) -> torch.Tensor:
  """|the policy's joint target - the joint| (rad), per motor. The real loop sends the
  target as the driver's goal (moved at 5 rad/s); the motor model's setpoint
  (``joint_pos_target``) stays within its torque cap's reach of the joint in training
  (0.18 / 2.6 = 0.07 rad), so a target pressed far into the table never shows there."""
  term = env.action_manager.get_term("joint_pos")
  return (term._processed_actions - env.scene["robot"].data.joint_pos).abs()


class held_far_goal(held_far):
  """``held_far`` on the policy's targets (``_goal_off``)."""

  def __call__(
    self, env: ManagerBasedRlEnv, error: float, grace: float
  ) -> torch.Tensor:
    self.t = torch.where(_goal_off(env) > error, self.t + env.step_dt, 0.0)
    return (self.t - grace).clamp(0.0, 1.0).max(dim=-1).values


def far_goals(env: ManagerBasedRlEnv, error: float, span: float) -> torch.Tensor:
  """``far_setpoints`` (worst motor) on the policy's targets (``_goal_off``)."""
  return ((_goal_off(env) - error) / span).clamp(0.0, 1.0).max(dim=-1).values


def capped_forward(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_speed: float,
  reference_speed: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """``capped_progress`` only while the command goes forward (pulling: the hard way)."""
  cmd = env.command_manager.get_command(command_name)
  pay = capped_progress(env, command_name, min_speed, reference_speed, asset_cfg)
  return torch.where(cmd[:, 0] > 0, pay, 0.0)
