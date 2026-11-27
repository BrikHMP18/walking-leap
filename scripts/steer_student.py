"""Steer a velocity-command policy (the gait students, Walking-Leap-Motion-Gait-Flat-
Velocity-v0) with the keyboard, in mjlab's MuJoCo viewer: what the real hand's loop will do.

The keys of scripted_gaits.py: arrows move (forward, back, left, right), ',' and '.'
turn; the same key again stops. A key sends the velocity the student was trained
with for that command (the clip's planned velocity: the planner's step over its
cycle), zero while the hand is fallen, as in training. The task's play config
otherwise (nominal hand, no pushes or cables); Enter resets, Space pauses.

  uv run python scripts/steer_student.py outputs/hf/runs/<student run>
"""

import argparse
from pathlib import Path

import mjlab.tasks  # noqa: F401  (registers the walking_leap tasks)
import numpy as np
import torch
from eval_velocity import load_policy, read_run
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from scripted_gaits import TELEOP_KEYS
from watch import LabeledViewer, label

from walking_leap.clips import gait_command_velocities, gait_cycle_periods, trained_clip


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("run", type=Path)
  parser.add_argument("--task", default=None, help="default: from the run's name")
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument(
    "--fallen", action="store_true", help="every reset (Enter) drops the hand fallen"
  )
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  args = parser.parse_args()

  policy = read_run(args.run, args.task, args.checkpoint)
  clip = trained_clip(policy.checkpoint.parent)
  commands = gait_command_velocities(clip if clip.startswith("gait") else "gait")
  state = {"name": "stand", "command": torch.zeros(3, device=args.device)}

  cfg = load_env_cfg(policy.task, play=True)
  cfg.scene.num_envs = 1
  if args.fallen:
    cfg.commands["motion"].fallen_start_prob = 1.0
  # Student tasks call the policy's group "student"; Velocity-v0 (PPO) calls it "actor".
  group = "student" if "student" in cfg.observations else "actor"
  term = cfg.observations[group].terms["command"]

  def keyboard_command(env: ManagerBasedRlEnv) -> torch.Tensor:
    fallen = env.command_manager.get_term("motion").fallen
    return state["command"].expand(env.num_envs, 3) * (~fallen)[:, None]

  term.func, term.params = keyboard_command, {}
  if "clock" in cfg.observations[group].terms:  # a clock student: run its gait clock
    periods = gait_cycle_periods(
      clip if clip.startswith("gait_smooth") else "gait_smooth"
    )
    ticks = {"phase": 0.0}

    def keyboard_clock(env: ManagerBasedRlEnv) -> torch.Tensor:
      # The phase advances over the held key's cycle (larger turns: longer cycles).
      if not torch.any(state["command"]):
        ticks["phase"] = 0.0
      angle = 2 * np.pi * ticks["phase"]
      if torch.any(state["command"]):
        period = periods.get(state["name"], 1.04)
        ticks["phase"] = (ticks["phase"] + env.step_dt / period) % 1.0
      fallen = env.command_manager.get_term("motion").fallen[:, None]
      clock = torch.tensor([np.sin(angle), np.cos(angle)], device=env.device)
      still = torch.tensor([0.0, 1.0], device=env.device)
      return torch.where(fallen, still, clock.expand(env.num_envs, 2))

    clock_term = cfg.observations[group].terms["clock"]
    clock_term.func, clock_term.params = keyboard_clock, {}
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device))
  actor = load_policy(env, policy, args.device)

  def key(code: int) -> None:
    name = TELEOP_KEYS.get(code)
    if name is None:
      return
    name = "stand" if state["name"] == name else name
    velocity = commands.get(name, np.zeros(3))
    state["name"] = name
    state["command"] = torch.tensor(velocity, dtype=torch.float32, device=args.device)
    print(
      f"[steer] {name}: {1e3 * velocity[0]:+.0f} mm/s forward, "
      f"{1e3 * velocity[1]:+.0f} mm/s left, {np.degrees(velocity[2]):+.0f} deg/s",
      flush=True,
    )

  print(
    f"[steer] {policy.label} ({policy.checkpoint.name}): arrows move, ',' and '.' "
    "turn, the same key again stops; Enter resets, Space pauses",
    flush=True,
  )
  title, subtitle = label(policy.task, args.run, policy.checkpoint)

  def status() -> str:
    v = state["command"].cpu().numpy()
    return (
      f"command: {state['name']} ({1e3 * v[0]:+.0f} fwd, {1e3 * v[1]:+.0f} left mm/s,"
      f" {np.degrees(v[2]):+.0f} deg/s)\narrows move, ',' '.' turn, same key: stop"
    )

  LabeledViewer(env, actor, title, subtitle, status=status, key_callback=key).run()
  env.close()


if __name__ == "__main__":
  main()
