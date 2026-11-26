"""The runner for teacher-student distillation, as mjlab's train and play use it.

Why a class here: mjlab 1.6.1 has a runner for PPO only (``MjlabOnPolicyRunner``),
and rsl-rl's ``DistillationRunner`` lacks what mjlab's scripts rely on: the
``registry_name`` argument train passes for tasks with a motion command, the
checkpoint's environment state, and play loading the policy as ``{"actor": True}``.
This joins the two; the training loop is rsl-rl's.
"""

from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner


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
