"""Steer a velocity-command student (Walking-Leap-Motion-Gait-Flat-Student-v0) with
the keyboard, in mjlab's MuJoCo viewer: what the real hand's loop will do.

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
from mjlab.viewer import NativeMujocoViewer
from scripted_gaits import TELEOP_KEYS

from walking_leap.clips import gait_command_velocities

STUDENT_TASK = "Walking-Leap-Motion-Gait-Flat-Student-v0"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("run", type=Path)
  parser.add_argument("--task", default=STUDENT_TASK)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  args = parser.parse_args()

  policy = read_run(args.run, args.task, args.checkpoint)
  commands = gait_command_velocities()
  state = {"name": "stand", "command": torch.zeros(3, device=args.device)}

  cfg = load_env_cfg(policy.task, play=True)
  cfg.scene.num_envs = 1
  term = cfg.observations["student"].terms["command"]

  def keyboard_command(env: ManagerBasedRlEnv) -> torch.Tensor:
    fallen = env.command_manager.get_term("motion").fallen
    return state["command"].expand(env.num_envs, 3) * (~fallen)[:, None]

  term.func, term.params = keyboard_command, {}
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
  NativeMujocoViewer(env, actor, key_callback=key).run()
  env.close()


if __name__ == "__main__":
  main()
