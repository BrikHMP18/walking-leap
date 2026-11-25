"""MDP terms that mjlab does not provide. Everything else is reused from mjlab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

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
# Observations (critic only: privileged).
##


def root_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Palm-origin height above the (flat, z=0) ground. Shape [B, 1]."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


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


def contact_active(
  env: ManagerBasedRlEnv, sensor_name: str, primary: str | None = None
) -> torch.Tensor:
  """1.0 where the sensor's primary touches its secondary, else 0.

  ``primary`` selects one primary by name (e.g. one digit); ``None`` means any
  primary. Assumes one slot per primary (``reduce="netforce"`` or
  ``num_slots=1``).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  found = sensor.data.found > 0
  if primary is not None:
    return found[:, sensor.primary_names.index(primary)].float()
  return found.any(dim=-1).float()


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


def lin_vel_tracking_error(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Horizontal velocity tracking error |v_cmd,xy - v_xy| (m/s), palm frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b[:, :2]
  return torch.norm(command[:, :2] - actual, dim=-1)


def yaw_rate_tracking_error(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Yaw-rate tracking error |w_cmd,z - w_z| (rad/s), palm frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.abs(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])


def _contact_slip(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """Horizontal sliding speed (m/s) at each primary's contact, and contact flags.

  For each primary body: the world velocity of the body's material point at the
  contact position, v_link + w x (p - x_link). A fingertip that rolls or pivots
  on the ground therefore reads ~0; only sliding counts. Needs body primaries,
  ``fields`` with "found" and "pos", and one slot per primary. Shapes [B, P].
  """
  sensor: ContactSensor = env.scene[sensor_name]
  asset: Entity = env.scene[asset_cfg.name]
  assert sensor.data.found is not None and sensor.data.pos is not None
  ids, _ = asset.find_bodies(sensor.primary_names, preserve_order=True)
  lever = sensor.data.pos - asset.data.body_link_pos_w[:, ids]
  vel = asset.data.body_link_lin_vel_w[:, ids] + torch.cross(
    asset.data.body_link_ang_vel_w[:, ids], lever, dim=-1
  )
  return torch.norm(vel[..., :2], dim=-1), (sensor.data.found > 0).float()


def contact_slip_speed(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Mean sliding speed (m/s) over the primaries in contact, 0 when none is."""
  speed, in_contact = _contact_slip(env, sensor_name, asset_cfg)
  return (speed * in_contact).sum(dim=-1) / in_contact.sum(dim=-1).clamp(min=1.0)


def contact_slip_rate(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  primary: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Sliding speed (m/s) of one primary while in contact, 0 while lifted.

  Its episode mean is the distance that contact slides per second, so a dragged
  digit stands out even if the other digits step cleanly.
  """
  speed, in_contact = _contact_slip(env, sensor_name, asset_cfg)
  i = env.scene[sensor_name].primary_names.index(primary)
  return speed[:, i] * in_contact[:, i]


def contact_slip_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  reference_speed: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Sum over primaries in contact of (sliding speed / reference_speed)^2.

  Dimensionless; use with a negative weight. Squared so that the sub-mm/s
  vibration of a standing hand costs nothing while dragging is expensive.
  """
  speed, in_contact = _contact_slip(env, sensor_name, asset_cfg)
  return (in_contact * torch.square(speed / reference_speed)).sum(dim=-1)


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
