"""PPO (RSL-RL) configuration. Algorithm values are mjlab's defaults. Also the
teacher-student distillation of the scripted-crawl policy."""

from dataclasses import dataclass, field, replace
from typing import Literal

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.config import RslRlBaseRunnerCfg


def walking_leap_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    # mjlab defaults, spelled out so every PPO knob is visible in one place.
    algorithm=RslRlPpoAlgorithmCfg(
      learning_rate=1e-3,
      schedule="adaptive",
      desired_kl=0.01,
      gamma=0.99,
      lam=0.95,
      clip_param=0.2,
      entropy_coef=0.005,
      value_loss_coef=1.0,
      num_learning_epochs=5,
      num_mini_batches=4,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    num_steps_per_env=24,
    max_iterations=3_000,
    save_interval=100,
    seed=42,
    experiment_name="walking_leap_flat",
    # Local logging by default; pass --agent.logger wandb to use Weights & Biases
    # (the W&B logger also writes the TensorBoard files). cluster/train.sbatch
    # uses W&B in offline mode.
    logger="tensorboard",
    wandb_project="walking-leap",
    # W&B gets metrics only; checkpoints are archived on Hugging Face.
    upload_model=False,
  )


def walking_leap_motion_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Same learner as walking, another experiment folder."""
  return replace(
    walking_leap_ppo_runner_cfg(),
    experiment_name="walking_leap_motion",
    max_iterations=2_000,
  )


def walking_leap_real_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Walking on the real hand's model, rewards only (v3): 6000 iterations, as
  IMU-free locomotion was trained for 5000 in Bhatt et al., "Blind Dexterity"
  (arXiv:2608.29487), with some margin."""
  return replace(walking_leap_ppo_runner_cfg(), max_iterations=6_000)


def walking_leap_crawl_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Imitating the real crawl: 3000 iterations (hello: 2000 nominal, 3000
  randomized)."""
  return replace(walking_leap_motion_ppo_runner_cfg(), max_iterations=3_000)


##
# Distillation (teacher -> student). mjlab 1.6.1 has config classes for PPO only;
# these mirror rsl-rl's Distillation and DistillationRunner arguments.
##


@dataclass
class RslRlDistillationAlgorithmCfg:
  """rsl-rl's ``Distillation``: behaviour cloning of the teacher's actions on the
  states the student visits (DAgger)."""

  num_learning_epochs: int = 1
  gradient_length: int = 15
  """Mini-batches (of one step each) accumulated per gradient step."""
  learning_rate: float = 1e-3
  max_grad_norm: float | None = 1.0
  loss_type: Literal["mse", "huber"] = "mse"
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  class_name: str = "Distillation"


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "DistillationRunner"
  student: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  teacher: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  algorithm: RslRlDistillationAlgorithmCfg = field(
    default_factory=RslRlDistillationAlgorithmCfg
  )


def walking_leap_gait_student_runner_cfg() -> RslRlDistillationRunnerCfg:
  """A velocity-command student of the scripted-crawl policy. The teacher is the
  PPO actor (same network, so its checkpoint loads as is: pass it with
  ``--agent.resume True --agent.load-run <run> --agent.load-checkpoint <file>``).
  The student is the same network on its own inputs, with the actor's Gaussian
  head so that PPO can continue from it; its noise during the rollouts is the
  teacher's final one (0.05)."""
  ppo = walking_leap_crawl_ppo_runner_cfg()
  return RslRlDistillationRunnerCfg(
    student=replace(
      ppo.actor,
      distribution_cfg={**ppo.actor.distribution_cfg, "init_std": 0.05},
    ),
    teacher=ppo.actor,
    # Huber, not MSE: the teacher's raw actions reach far beyond the joint limits
    # (RMS 10.5, up to 78, against limits about 1-8 from the stance at scale 0.25;
    # measured on the Gait-v0 run), and the action term clips them there.
    algorithm=RslRlDistillationAlgorithmCfg(loss_type="huber"),
    obs_groups={"student": ("student",), "teacher": ("teacher",)},
    num_steps_per_env=24,
    max_iterations=1_500,
    save_interval=100,
    seed=42,
    experiment_name=ppo.experiment_name,  # where the teacher's runs are
    logger="tensorboard",
    wandb_project=ppo.wandb_project,
    upload_model=False,
  )


def walking_leap_two_teacher_runner_cfg() -> RslRlDistillationRunnerCfg:
  """The student of Gait-v2 and Getup (``runner.SwitchTeacher``): the teacher is both
  PPO actors (same network each), loaded from scripts/merge_teachers.py's file."""
  cfg = walking_leap_gait_student_runner_cfg()
  return replace(
    cfg,
    teacher=replace(cfg.teacher, class_name="walking_leap.runner:SwitchTeacher"),
    obs_groups={
      "student": ("student",),
      "teacher": ("teacher", "teacher_getup", "fallen"),
    },
    max_iterations=1_000,
  )


def walking_leap_velocity_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO on the student's own task, continuing from the two-teacher student
  (``runner.StudentPpoRunner``): the actor is the student's network; a gentler start
  (learning rate 3e-4) so that a fresh critic does not undo the distillation."""
  ppo = walking_leap_crawl_ppo_runner_cfg()
  return replace(
    ppo,
    algorithm=replace(ppo.algorithm, learning_rate=3e-4),
    max_iterations=1_500,
  )


def walking_leap_velocity_v2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Velocity-v2: mjlab's PPO defaults again (learning rate 1e-3, adaptive) with the
  wider noise of ``runner.ExploringStudentPpoRunner``; also trains from scratch."""
  return replace(walking_leap_crawl_ppo_runner_cfg(), max_iterations=1_500)


def walking_leap_velocity_v3_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Velocity-v3: v2's learner with twice mjlab's entropy bonus (0.01), so that the
  noise does not shrink back as v2's did (0.3 -> 0.12 in 300 iterations)."""
  cfg = walking_leap_velocity_v2_ppo_runner_cfg()
  return replace(cfg, algorithm=replace(cfg.algorithm, entropy_coef=0.01))
