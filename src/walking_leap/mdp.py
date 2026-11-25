"""MDP terms that mjlab does not provide. Everything else is reused from mjlab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


##
# Commands.
##


class VelocityCommand(UniformVelocityCommand):
  """mjlab's ``UniformVelocityCommand`` without the Viser joystick panel.

  mjlab 1.6.1 builds joystick sliders whose bounds start at 0.1, so any command
  range below that (0.05 m/s forward, 0 lateral and yaw here) makes
  ``play --viewer viser`` crash. Sampling, metrics and arrows are unchanged.
  """

  create_gui = CommandTerm.create_gui


@dataclass(kw_only=True)
class VelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> VelocityCommand:
    return VelocityCommand(self, env)


##
# Observations (critic only: privileged, unless said otherwise).
##


def root_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Palm-origin height above the (flat, z=0) ground. Shape [B, 1]."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def motion_gravity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Gravity direction in the reference palm's frame. Shape [B, 3].

  The part of the reference's orientation an IMU can follow: no heading. For the
  actor, next to ``projected_gravity`` (the same of the real palm).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[command.cfg.entity_name]
  return quat_apply_inverse(command.anchor_quat_w, asset.data.gravity_vec_w)


##
# Rewards.
##


def root_height_floor(
  env: ManagerBasedRlEnv,
  min_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """exp(-(shortfall/std)^2), shortfall = max(min_height - h, 0).

  One-sided on purpose: the hand may stand anywhere above ``min_height`` (it is
  not pulled toward one prescribed height), while sagging toward the ground is
  penalized smoothly. Vertical bouncing is discouraged by the z term of
  ``track_linear_velocity``, not here.
  """
  asset: Entity = env.scene[asset_cfg.name]
  shortfall = torch.clamp(min_height - asset.data.root_link_pos_w[:, 2], min=0.0)
  return torch.exp(-torch.square(shortfall / std))


def footprint(
  env: ManagerBasedRlEnv,
  spots: tuple[tuple[float, float], ...],
  std: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """exp(-mean over feet of (distance to its spot / std)^2), xy in the palm frame.

  Pulls every foot toward its place in the standing pose without saying when to step
  ("Fingers as Legs", arXiv 2609.17172). ``asset_cfg`` lists the feet's bodies in the
  order of ``spots``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  feet = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
  offset = feet - asset.data.root_link_pos_w[:, None, :]
  quat = asset.data.root_link_quat_w[:, None, :].expand(-1, feet.shape[1], -1)
  local = quat_apply_inverse(quat, offset)[..., :2]
  target = torch.tensor(spots, device=env.device)
  distance_sq = torch.sum(torch.square(local - target), dim=-1)
  return torch.exp(-distance_sq.mean(dim=-1) / std**2)


def feet_down(
  env: ManagerBasedRlEnv, sensor_name: str, allowed_up: int = 1
) -> torch.Tensor:
  """1 while at most ``allowed_up`` feet are off the ground, else 0: a crawl.

  A reward for time in the air instead was met by balancing on two feet and holding
  the others up; the same as a cost made falling at once the best early policy.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  up = (sensor.data.found == 0).sum(dim=-1)
  return (up <= allowed_up).float()


def short_steps(
  env: ManagerBasedRlEnv, sensor_name: str, min_time: float
) -> torch.Tensor:
  """Flights and stances shorter than ``min_time`` s, as they end (a cost): buzzing."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.last_air_time is not None and data.last_contact_time is not None
  landed = sensor.compute_first_contact(dt=env.step_dt)
  lifted = sensor.compute_first_air(dt=env.step_dt)
  short = (landed & (data.last_air_time < min_time)).float()
  short += (lifted & (data.last_contact_time < min_time)).float()
  return short.sum(dim=-1)


def feet_held_up(
  env: ManagerBasedRlEnv, sensor_name: str, max_air_time: float
) -> torch.Tensor:
  """Feet in the air for longer than ``max_air_time`` s (a cost): a leg not used."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_air_time is not None
  return (sensor.data.current_air_time > max_air_time).float().sum(dim=-1)


def contact_active(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1.0 where any primary of the contact sensor touches its secondary, else 0."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).any(dim=-1).float()


##
# Terminations.
##


def sustained_contact(
  env: ManagerBasedRlEnv, sensor_name: str, max_duration: float
) -> torch.Tensor:
  """Terminate once any primary has been in continuous contact > max_duration s.

  Requires ``track_air_time=True`` on the sensor. Brief touches (e.g. while
  recovering from a stumble) are allowed; resting or dragging is not.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  contact_time = sensor.data.current_contact_time
  assert contact_time is not None, f"'{sensor_name}' needs track_air_time=True"
  return torch.any(contact_time > max_duration, dim=-1)


##
# Metrics (logged as Episode_Metrics/*, independent of the reward).
##


def forward_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Palm velocity along its own forward (+x) axis, m/s."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, 0]


def net_displacement(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Horizontal distance (m) of the palm from its environment origin.

  With ``reduce="last"`` this is the net distance covered in the episode, as
  resets place the palm at the origin (no xy randomization).
  """
  asset: Entity = env.scene[asset_cfg.name]
  delta = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  return torch.norm(delta, dim=-1)


def actuator_saturation(
  env: ManagerBasedRlEnv, effort_limit: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Fraction of actuators at their torque limit (a metric, or a cost)."""
  asset: Entity = env.scene[asset_cfg.name]
  saturated = asset.data.actuator_force.abs() >= 0.99 * effort_limit
  return saturated.float().mean(dim=-1)


def mechanical_power(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Positive joint mechanical power sum(max(tau * qdot, 0)), W.

  Mechanical output only; it is not the electrical power drawn by the servos.
  """
  asset: Entity = env.scene[asset_cfg.name]
  power = asset.data.qfrc_actuator * asset.data.joint_vel
  return torch.clamp(power, min=0.0).sum(dim=-1)
