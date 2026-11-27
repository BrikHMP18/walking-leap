"""Does a keyboard policy move when a key is held? The panel's use, in simulation.

For each key (W S A D Q E: forward, back, left, right, turn left, turn right) the hand
stands 2 s with no command, then the key is held 8 s: the command is the velocity the
policy trained with for it (``clips.gait_command_velocities``), zero while fallen, and a
clock student's gait clock runs from the key press as on the hand (real/policy.py). The
palm's displacement over those 8 s, in its heading at the press, averaged over HANDS
hands (the task's play config: nominal hand, no pushes).

  uv run python scripts/eval_keys.py outputs/hf/runs/<run> [--checkpoint model_N.pt]
"""

import argparse
import json
import math
from pathlib import Path

import mjlab.tasks  # noqa: F401  (registers the walking_leap tasks)
import torch
from eval_velocity import load_policy, read_run
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from walking_leap.clips import gait_command_velocities, gait_cycle_s, trained_clip

ROOT = Path(__file__).parents[1]
HANDS, STAND_S, HOLD_S = 8, 2.0, 8.0
KEYS = ("forward", "back", "left", "right", "turn_left", "turn_right")


def yaw(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def hold(policy, key: str, device: str) -> dict:
  cfg = load_env_cfg(policy.task, play=True)
  cfg.scene.num_envs, cfg.seed = HANDS, 0
  group = "student" if "student" in cfg.observations else "actor"
  terms = cfg.observations[group].terms
  state = {"command": torch.zeros(3, device=device), "ticks": 0}

  def command(env):
    fallen = env.command_manager.get_term("motion").fallen
    return state["command"].expand(env.num_envs, 3) * (~fallen)[:, None]

  terms["command"].func, terms["command"].params = command, {}
  if "clock" in terms:
    period = gait_cycle_s()

    def clock(env):
      angle = 2 * math.pi * ((state["ticks"] * env.step_dt / period) % 1.0)
      if torch.any(state["command"]):
        state["ticks"] += 1
      fallen = env.command_manager.get_term("motion").fallen[:, None]
      c = torch.tensor([math.sin(angle), math.cos(angle)], device=env.device)
      still = torch.tensor([0.0, 1.0], device=env.device)
      return torch.where(fallen, still, c.expand(env.num_envs, 2))

    terms["clock"].func, terms["clock"].params = clock, {}
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device))
  actor = load_policy(env, policy, device)
  robot = env.unwrapped.scene["robot"]
  fell = torch.zeros(HANDS, dtype=torch.bool, device=device)
  obs = env.get_observations()
  stand, steps = (
    round(STAND_S / env.unwrapped.step_dt),
    round((STAND_S + HOLD_S) / env.unwrapped.step_dt),
  )
  clip = trained_clip(policy.checkpoint.parent)
  velocity = gait_command_velocities(clip if clip.startswith("gait") else "gait")[key]
  with torch.inference_mode():
    for k in range(steps):
      if k == stand:
        state["command"] = torch.tensor(velocity, dtype=torch.float32, device=device)
        p0 = robot.data.root_link_pos_w[:, :2].clone()
        q0 = robot.data.root_link_quat_w.clone()
      obs, _, dones, _ = env.step(actor(obs))
      if k >= stand:
        fell |= env.unwrapped.command_manager.get_term("motion").fallen | dones.bool()
    d = robot.data.root_link_pos_w[:, :2] - p0
    y0 = yaw(q0)
    fwd = (torch.cos(y0) * d[:, 0] + torch.sin(y0) * d[:, 1]) * 1e3
    left = (-torch.sin(y0) * d[:, 0] + torch.cos(y0) * d[:, 1]) * 1e3
    turn = torch.rad2deg(
      torch.remainder(yaw(robot.data.root_link_quat_w) - y0 + math.pi, 2 * math.pi)
      - math.pi
    )
  env.close()
  return {
    "command": [round(1e3 * float(velocity[0]), 1), round(1e3 * float(velocity[1]), 1),
                round(math.degrees(float(velocity[2])), 1)],
    "forward_mm_s": float(fwd.mean()) / HOLD_S,
    "left_mm_s": float(left.mean()) / HOLD_S,
    "turn_deg_s": float(turn.mean()) / HOLD_S,
    "fell": float(fell.float().mean()),
  }  # fmt: skip


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("run", type=Path)
  parser.add_argument("--task", default=None)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  args = parser.parse_args()
  policy = read_run(args.run, args.task, args.checkpoint)
  results = {key: hold(policy, key, args.device) for key in KEYS}
  print(f"\n{policy.label} ({policy.checkpoint.name}): {HOLD_S:.0f} s holding each key")
  print(
    "key         | command fwd/left mm/s, turn deg/s | got: fwd mm/s left mm/s turn deg/s | fell"
  )
  for key, r in results.items():
    c = r["command"]
    print(
      f"{key:11s} | {c[0]:+6.1f} {c[1]:+6.1f} {c[2]:+6.1f}            |"
      f" {r['forward_mm_s']:+6.1f} {r['left_mm_s']:+6.1f} {r['turn_deg_s']:+6.1f}"
      f"            | {r['fell']:.0%}"
    )
  out = ROOT / "outputs" / "eval" / f"keys_{policy.label}.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(
    json.dumps({"checkpoint": str(policy.checkpoint), **results}, indent=1)
  )
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
