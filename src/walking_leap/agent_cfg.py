"""PPO (RSL-RL) configuration. Algorithm values are mjlab's defaults."""

from dataclasses import replace

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


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
