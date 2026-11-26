"""Does a scripted-crawl policy move as each command plans? Per command, measured.

For each command of ``gait.py`` (forward, back, left, right, both turns, stand), the
policy imitates a clip of that command alone (``scripts/gait_clips.py`` writes them:
a first cycle in place, then ``--cycles`` of the command). Over the whole clip, the
palm's displacement in its starting heading (forward and left in mm, turn in deg) is
compared with two references:

  planned    what the planner asked for: the command's step times the cycles
  clip       what the clip's palm does (the targets replayed on the real hand's model,
             open loop): what the policy was trained to imitate
  policy     the trained policy, ROLLOUTS hands, nominal model, no noise

A hand that falls (its episode ends early) is counted in ``fell`` and left out of
the displacement.

  uv run python scripts/gait_clips.py --hands 1 --cycles 6 --out outputs/clips/eval
  uv run python scripts/eval_gait.py outputs/hf/runs/<run> --clips outputs/clips/eval
"""

import argparse
import json
import math
from pathlib import Path

import mjlab.tasks  # noqa: F401  (registers the walking_leap tasks)
import numpy as np
import torch
from eval_velocity import Policy, load_policy, read_run
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from walking_leap import gait

ROOT = Path(__file__).parents[1]
ROLLOUTS = 4


def yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat_wxyz.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def displacement(p0, q0, p1, q1) -> torch.Tensor:
  """(forward mm, left mm, turn deg) from pose 0 to pose 1, in pose 0's heading."""
  y0 = yaw(q0)
  d = p1[:, :2] - p0[:, :2]
  c, s = torch.cos(y0), torch.sin(y0)
  fwd = c * d[:, 0] + s * d[:, 1]
  left = -s * d[:, 0] + c * d[:, 1]
  turn = torch.remainder(yaw(q1) - y0 + math.pi, 2 * math.pi) - math.pi
  return torch.stack([fwd * 1000, left * 1000, torch.rad2deg(turn)], 1)


def rollout(policy: Policy, clip: Path, rollouts: int, device: str) -> dict:
  cfg = load_env_cfg(policy.task, play=True)
  cfg.commands["motion"].motion_file = str(clip)
  cfg.scene.num_envs = rollouts
  cfg.seed = 0
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device))
  base = env.unwrapped
  command = base.command_manager.get_term("motion")
  actor = load_policy(env, policy, device)
  steps = command.motion.time_step_total - 2  # stop before the clip wraps around

  def poses():
    return (
      command.robot_anchor_pos_w.clone(),
      command.robot_anchor_quat_w.clone(),
      command.anchor_pos_w.clone(),
      command.anchor_quat_w.clone(),
    )

  obs = env.get_observations()
  start = poses()
  alive = torch.ones(rollouts, dtype=torch.bool, device=device)
  with torch.inference_mode():
    for _ in range(steps):
      obs, _, dones, _ = env.step(actor(obs))
      alive &= ~dones.bool()
  end = poses()
  env.close()
  robot = displacement(start[0], start[1], end[0], end[1])[alive].cpu().numpy()
  ref = displacement(start[2], start[3], end[2], end[3])[0].cpu().numpy()
  return {
    "seconds": steps / 50.0,
    "clip": ref.round(1).tolist(),
    "policy_mean": robot.mean(0).round(1).tolist() if len(robot) else None,
    "policy_std": robot.std(0).round(1).tolist() if len(robot) else None,
    "fell": float((~alive).float().mean().cpu()),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("runs", nargs="+", type=Path)
  parser.add_argument("--clips", type=Path, default=ROOT / "outputs" / "clips" / "eval")
  parser.add_argument(
    "--cycles", type=int, default=6, help="as gait_clips.py wrote them"
  )
  parser.add_argument("--task", default=None)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--rollouts", type=int, default=ROLLOUTS)
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  args = parser.parse_args()

  for path in args.runs:
    policy = read_run(path, args.task, args.checkpoint)
    results = {}
    for name, step in gait.GAIT_COMMANDS.items():
      fwd, left, turn = step  # m, m, rad per cycle
      planned = [fwd * 1000 * args.cycles, left * 1000 * args.cycles]
      planned = np.round(planned + [math.degrees(turn) * args.cycles], 1).tolist()
      r = rollout(policy, args.clips / f"gait_{name}.npz", args.rollouts, args.device)
      results[name] = {"planned": planned, **r}
    out = ROOT / "outputs" / "eval" / f"gait_{policy.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
      json.dumps({"checkpoint": str(policy.checkpoint), **results}, indent=1)
    )

    print(
      f"\n{policy.label} ({policy.checkpoint.name}), {args.cycles} cycles per command"
    )
    print(
      "command     | planned fwd/left mm, turn deg | clip             | policy (mean)    | fell"
    )
    for name, r in results.items():

      def f(v):
        return " ".join(f"{x:+5.0f}" for x in v) if v else "  -  "

      print(
        f"{name:11s} | {f(r['planned']):29s} | {f(r['clip']):16s} | "
        f"{f(r['policy_mean']):16s} | {r['fell']:.0%}"
      )
    print(f"wrote {out}")


if __name__ == "__main__":
  main()
