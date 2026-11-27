"""The runner for teacher-student distillation, as mjlab's train and play use it.

Why a class here: mjlab 1.6.1 has a runner for PPO only (``MjlabOnPolicyRunner``),
and rsl-rl's ``DistillationRunner`` lacks what mjlab's scripts rely on: the
``registry_name`` argument train passes for tasks with a motion command, the
checkpoint's environment state, and play loading the policy as ``{"actor": True}``.
This joins the two; the training loop is rsl-rl's.
"""

from copy import deepcopy

import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.runners import DistillationRunner
from torch import nn


class GaitStudentRunner(MjlabOnPolicyRunner, DistillationRunner):
  """``DistillationRunner`` with mjlab's checkpoints. A PPO checkpoint loads as the
  teacher (train's ``--agent.resume``); a distillation checkpoint loaded as
  ``{"actor": True}`` (play, scripts) loads the student, which is the policy."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    # As MjlabOnPolicyRunner does for "actor" and "critic".
    for key in ("student", "teacher"):
      for opt in ("cnn_cfg", "distribution_cfg"):
        if train_cfg[key].get(opt) is None:
          train_cfg[key].pop(opt, None)
      if train_cfg[key].get("rnn_type") is None:
        for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
          train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if load_cfg is not None and load_cfg.get("actor"):
      load_cfg = {"student": True}
    return super().load(path, load_cfg, strict, map_location)


class SwitchTeacher(nn.Module):
  """Two PPO actors as one teacher (rsl-rl's model interface): the getting-up teacher
  while the hand is fallen (observation ``fallen``), the walking teacher otherwise. Each
  reads its own observation group (``teacher``, ``teacher_getup``) and keeps the
  normalizer it trained with. Its state dict is ``walk.*`` and ``getup.*``: merge the two
  checkpoints with scripts/merge_teachers.py, then load the result as any teacher."""

  is_recurrent = False

  def __init__(self, obs, obs_groups, obs_set, output_dim, **cfg) -> None:
    super().__init__()
    # MLPModel pops keys out of its config: each gets its own copy.
    self.walk = MLPModel(
      obs, {"walk": ["teacher"]}, "walk", output_dim, **deepcopy(cfg)
    )
    self.getup = MLPModel(
      obs, {"getup": ["teacher_getup"]}, "getup", output_dim, **deepcopy(cfg)
    )

  def forward(self, obs, stochastic_output: bool = False) -> torch.Tensor:
    down = obs["fallen"][:, :1] > 0.5
    return torch.where(down, self.getup(obs), self.walk(obs))

  def reset(self, dones=None, hidden_state=None) -> None:
    pass

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones=None) -> None:
    pass

  def update_normalization(self, obs) -> None:
    pass


class StudentPpoRunner(MotionTrackingOnPolicyRunner):
  """PPO that starts from a distilled student: a distillation checkpoint loads its
  student as the actor (the critic starts fresh, the iteration count at 0) and widens
  the actor's noise to ``RESUME_STD`` for exploring; anything else loads as usual."""

  RESUME_STD = 0.1

  def load(self, path, load_cfg=None, strict=True, map_location=None) -> dict:
    saved = torch.load(path, map_location=map_location, weights_only=False)
    if "student_state_dict" not in saved:
      return super().load(path, load_cfg, strict, map_location)
    self.alg.actor.load_state_dict(saved["student_state_dict"], strict=strict)
    with torch.no_grad():
      for name, p in self.alg.actor.named_parameters():
        if name.endswith("std_param"):
          p.fill_(self.RESUME_STD)
    return saved.get("infos") or {}


class ExploringStudentPpoRunner(StudentPpoRunner):
  """``StudentPpoRunner`` that widens the noise to 0.3: Velocity-v1 at std 0.1 sat in
  "standing is safe" (forward 4.3 of 30 mm/s after 500 iterations)."""

  RESUME_STD = 0.3

  def load(self, path, load_cfg=None, strict=True, map_location=None) -> dict:
    # A PPO checkpoint (continuing Velocity-v1) loads as usual, then gets the wider noise.
    infos = super().load(path, load_cfg, strict, map_location)
    with torch.no_grad():
      for name, p in self.alg.actor.named_parameters():
        if name.endswith("std_param"):
          p.fill_(self.RESUME_STD)
    return infos
