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
  """Fraction of actuators at their torque limit."""
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
