"""MDP terms that mjlab does not provide. Everything else is reused from mjlab."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_from_angle_axis,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

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


class GaitMotionCommand(MotionCommand):
  """mjlab's ``MotionCommand`` that survives falls (the gait task, v1).

  When the palm tilts past ``fall_tilt`` the hand is *fallen*: the clip stops
  advancing. Once the palm is back within ``recover_tilt`` and above
  ``recover_height`` for ``recover_s``, the clip resumes from where it stopped,
  moved (a turn about the vertical and a shift) so that its palm stands where the
  real one got up: the imitation then continues from there instead of pulling the
  hand back to where it fell. With ``fallen_start_prob`` an episode starts with the
  hand dropped from ``fallen_start_height`` tilted by ``fallen_start_tilt`` (rad)
  about a random horizontal axis. The clip's ``command`` array (each frame's planned
  velocity, ``gait.plan``) is available as ``planned_velocity``, zero while fallen.
  """

  cfg: GaitMotionCommandCfg

  def __init__(self, cfg: GaitMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n, frames = self.num_envs, self.motion.time_step_total
    data = np.load(cfg.motion_file)
    planned = data["command"] if "command" in data.files else np.zeros((frames, 3))
    self._planned = torch.tensor(planned, dtype=torch.float32, device=self.device)
    self.offset_pos = torch.zeros(n, 3, device=self.device)
    self.offset_quat = torch.zeros(n, 4, device=self.device)
    self.offset_quat[:, 0] = 1.0
    self.fallen = torch.zeros(n, dtype=torch.bool, device=self.device)
    self.fallen_steps = torch.zeros(n, dtype=torch.long, device=self.device)
    self.upright_steps = torch.zeros(n, dtype=torch.long, device=self.device)
    self._recover_steps = max(1, round(cfg.recover_s / env.step_dt))
    self._resetting = False
    self.metrics["fallen"] = torch.zeros(n, device=self.device)
    self.metrics["recoveries"] = torch.zeros(n, device=self.device)

  # The clip as each env sees it: turned by offset_quat, shifted by offset_pos.

  def _turn(self, quat: torch.Tensor, x: torch.Tensor, rotate: bool) -> torch.Tensor:
    if x.dim() == 3:
      quat = quat[:, None, :].expand(-1, x.shape[1], -1)
    return quat_apply(quat, x) if rotate else quat_mul(quat, x)

  def _shift(self) -> torch.Tensor:
    return self.offset_pos + self._env.scene.env_origins

  @property
  def body_pos_w(self) -> torch.Tensor:
    pos = self.motion.body_pos_w[self.time_steps]
    return self._turn(self.offset_quat, pos, True) + self._shift()[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._turn(self.offset_quat, self.motion.body_quat_w[self.time_steps], False)

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    vel = self.motion.body_lin_vel_w[self.time_steps]
    return self._turn(self.offset_quat, vel, True)

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    vel = self.motion.body_ang_vel_w[self.time_steps]
    return self._turn(self.offset_quat, vel, True)

  def _anchor(self, array: torch.Tensor) -> torch.Tensor:
    return array[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    pos = self._anchor(self.motion.body_pos_w)
    return quat_apply(self.offset_quat, pos) + self._shift()

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return quat_mul(self.offset_quat, self._anchor(self.motion.body_quat_w))

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return quat_apply(self.offset_quat, self._anchor(self.motion.body_lin_vel_w))

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return quat_apply(self.offset_quat, self._anchor(self.motion.body_ang_vel_w))

  @property
  def planned_velocity(self) -> torch.Tensor:
    """The clip's planned (forward m/s, left m/s, turn rad/s) now; zero while
    fallen. Shape [B, 3]."""
    return self._planned[self.time_steps] * (~self.fallen)[:, None]

  def palm_tilt(self) -> torch.Tensor:
    """The real palm's tilt from upright (rad). Shape [B]."""
    gravity = quat_apply_inverse(
      self.robot_anchor_quat_w, self.robot.data.gravity_vec_w
    )
    return torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0))

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    self._resetting = True
    try:
      return super().reset(env_ids)
    finally:
      self._resetting = False

  def _resample_command(self, env_ids: torch.Tensor):
    self.offset_pos[env_ids] = 0.0
    self.offset_quat[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
    self.fallen[env_ids] = False
    self.fallen_steps[env_ids] = 0
    self.upright_steps[env_ids] = 0
    super()._resample_command(env_ids)
    if self._resetting and self.cfg.fallen_start_prob > 0.0:
      self._start_fallen(env_ids)

  def _start_fallen(self, env_ids: torch.Tensor) -> None:
    draw = torch.rand(len(env_ids), device=self.device)
    ids = env_ids[draw < self.cfg.fallen_start_prob]
    if len(ids) == 0:
      return
    low, high = self.cfg.fallen_start_tilt
    angle = sample_uniform(low, high, (len(ids),), device=self.device)
    heading = sample_uniform(-math.pi, math.pi, (len(ids),), device=self.device)
    axis = torch.stack(
      [torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)], dim=-1
    )
    pos = self.body_pos_w[ids, 0].clone()
    pos[:, 2] = self._env.scene.env_origins[ids, 2] + self.cfg.fallen_start_height
    quat = quat_mul(quat_from_angle_axis(angle, axis), self.body_quat_w[ids, 0])
    still = torch.zeros(len(ids), 6, device=self.device)
    self.robot.write_root_state_to_sim(torch.cat([pos, quat, still], -1), env_ids=ids)
    self.fallen[ids] = True

  def _reanchor(self, ids: torch.Tensor) -> None:
    """Move the clip so that its palm, now, stands where the real palm is."""
    clip_pos = self._anchor(self.motion.body_pos_w)[ids]
    clip_quat = self._anchor(self.motion.body_quat_w)[ids]
    turn = yaw_quat(quat_mul(self.robot_anchor_quat_w[ids], quat_inv(clip_quat)))
    here = self.robot_anchor_pos_w[ids] - self._env.scene.env_origins[ids]
    shift = here - quat_apply(turn, clip_pos)
    shift[:, 2] = 0.0
    self.offset_quat[ids], self.offset_pos[ids] = turn, shift

  def _update_command(self, env_ids: torch.Tensor | None = None):
    mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    mask[slice(None) if env_ids is None else env_ids] = True
    tilt = self.palm_tilt()
    height = self.robot_anchor_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
    self.fallen |= mask & (tilt > self.cfg.fall_tilt)
    upright = (tilt < self.cfg.recover_tilt) & (height > self.cfg.recover_height)
    counting = mask & self.fallen & upright
    self.upright_steps = torch.where(counting, self.upright_steps + 1, 0)
    back = self.fallen & (self.upright_steps >= self._recover_steps)
    if back.any():
      ids = torch.nonzero(back).squeeze(-1)
      self._reanchor(ids)
      self.fallen[ids] = False
      self.fallen_steps[ids] = 0
      self.upright_steps[ids] = 0
      self.metrics["recoveries"][ids] += 1.0
    self.fallen_steps += (mask & self.fallen).long()
    # The clip advances only while standing.
    self.time_steps += (mask & ~self.fallen).long()
    wrap_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if wrap_ids.numel() > 0:
      self._resample_command(wrap_ids)
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()
    self.update_relative_body_poses()
    if env_ids is None and self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()
    self.metrics["fallen"] = self.fallen.float()


@dataclass(kw_only=True)
class GaitMotionCommandCfg(MotionCommandCfg):
  fall_tilt: float = 0.8  # rad (46 deg): beyond this the hand is fallen
  recover_tilt: float = 0.3  # rad (17 deg)
  recover_height: float = 0.09  # m, the palm at least this high to count as up
  recover_s: float = 0.3  # s within both before the clip resumes
  fallen_start_prob: float = 0.0
  # rad (57 to 115 deg): falls onto a side, the fingertips or the wrist, which an
  # open-loop search got up from (scripts/getup_check.py: 0.96 to 1.00); from the back
  # (164 deg) it only rolled onto a side (0.51).
  fallen_start_tilt: tuple[float, float] = (1.0, 2.0)
  fallen_start_height: float = 0.16  # m, the palm dropped from this height

  def build(self, env: ManagerBasedRlEnv) -> GaitMotionCommand:
    return GaitMotionCommand(self, env)


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


def long_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
  cap: float | None = None,
) -> torch.Tensor:
  """Sum over primaries of their time off the secondary beyond ``threshold`` (s).

  Zero for steps shorter than ``threshold``; grows for as long as a primary (e.g.
  a digit) is held up, up to ``cap`` (s) each if given. A cap keeps the penalty
  rate bounded, so that ending the episode is never a way out of it. Needs
  ``track_air_time=True`` on the sensor.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air = sensor.data.current_air_time
  assert air is not None, f"'{sensor_name}' needs track_air_time=True"
  return torch.clamp(air - threshold, min=0.0, max=cap).sum(dim=-1)


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
  bounded: bool = False,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Sum over primaries in contact of (sliding speed / reference_speed)^2.

  Dimensionless; use with a negative weight. Squared so that the sub-mm/s
  vibration of a standing hand costs nothing while dragging is expensive.
  ``bounded``: 1 - exp(-(speed / reference_speed)^2) instead, at most 1 per
  primary, so that the fast sliding of early, random motion cannot make ending
  the episode the best way out of the penalty.
  """
  speed, in_contact = _contact_slip(env, sensor_name, asset_cfg)
  ratio = torch.square(speed / reference_speed)
  per_primary = 1.0 - torch.exp(-ratio) if bounded else ratio
  return (in_contact * per_primary).sum(dim=-1)


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


##
# The gait task's survival terms (GaitMotionCommand).
##


def palm_upright(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 with the palm upright, 0 upside down, linear in the cosine of the tilt."""
  asset: Entity = env.scene[_ROBOT.name]
  return (1.0 - asset.data.projected_gravity_b[:, 2]) / 2.0


def palm_height_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """exp(-(dz / std)^2), dz the real palm's height below or above the clip's."""
  command = cast(GaitMotionCommand, env.command_manager.get_term(command_name))
  dz = command.robot_anchor_pos_w[:, 2] - command.anchor_pos_w[:, 2]
  return torch.exp(-((dz / std) ** 2))


def planned_velocity_exp(
  env: ManagerBasedRlEnv, command_name: str, std_lin: float, std_yaw: float
) -> torch.Tensor:
  """How well the palm moves as the clip's plan intends (not as its replay does):
  the mean of exp(-(e / std)^2) for the horizontal velocity in the palm's heading
  frame (m/s) and the turn rate (rad/s). Zero while fallen."""
  command = cast(GaitMotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[command.cfg.entity_name]
  planned = command.planned_velocity
  heading = yaw_quat(asset.data.root_link_quat_w)
  velocity = quat_apply_inverse(heading, asset.data.root_link_lin_vel_w)[:, :2]
  lin = torch.sum((velocity - planned[:, :2]) ** 2, dim=-1) / std_lin**2
  yaw = (asset.data.root_link_ang_vel_w[:, 2] - planned[:, 2]) ** 2 / std_yaw**2
  return 0.5 * (torch.exp(-lin) + torch.exp(-yaw)) * (~command.fallen).float()


def fallen_too_long(
  env: ManagerBasedRlEnv, command_name: str, limit_s: float
) -> torch.Tensor:
  """The hand has been fallen for more than ``limit_s`` (s)."""
  command = cast(GaitMotionCommand, env.command_manager.get_term(command_name))
  return command.fallen_steps.float() * env.step_dt > limit_s
