"""Do velocity policies walk when told to? Fixed commands, measured response.

Each policy gets one constant command for a whole 10 s episode, at the edges of its
task's command range (forward, back, both sides, both turns) and zero. ROLLOUTS
episodes per command, from the task's reset (random heading, a small drop), no
noise. Next to the policies, a hand that stands still (action 0, the nominal pose)
gives the reference. Measured over the last 9 s (the first second settles), in the
palm frame:

  achieved   mean velocity along the commanded axis, and its share of the command
  fell       share of episodes that ended in a fall before 10 s

The task is read from the run's name (`<task>_s<seed>`, as cluster/train.sbatch names
runs; --task otherwise) and the checkpoint is the run's last one (--checkpoint
otherwise).

  uv run python scripts/eval_velocity.py outputs/hf/runs/<run> [<run> ...]
  uv run python scripts/eval_velocity.py outputs/hf/runs/<run> --view --command forward

--view opens mjlab's MuJoCo viewer on one hand holding one command (it stays the same
after a reset with Enter).
"""

import argparse
import copy
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab.tasks  # noqa: F401  (registers the walking_leap tasks)
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer

from walking_leap import mdp

ROOT = Path(__file__).parents[1]
STEPS = 500  # 10 s at 50 Hz
SETTLE = 50  # the first second is not measured
# Command name: index of the commanded axis in (vx, vy, yaw rate); None for zero.
AXES = {
  "forward": 0,
  "back": 0,
  "thumb side": 1,
  "ring side": 1,
  "turn left": 2,
  "turn right": 2,
  "stand": None,
}


@dataclass
class Policy:
  label: str
  task: str
  checkpoint: Path | None  # None: stand still (action 0)


def read_run(path: Path, task: str | None, checkpoint: str | None) -> Policy:
  """The run's task (from its name unless given) and checkpoint (last unless given)."""
  agent = (path / "params" / "agent.yaml").read_text()
  match = re.search(r"^run_name: (.+)$", agent, re.MULTILINE)
  name = match.group(1).strip().strip("'\"") if match else ""
  task = task or re.sub(r"_s\d+$", "", name)
  if task not in list_tasks():
    raise SystemExit(f"{path}: cannot tell its task from {name!r}; pass --task")
  if checkpoint:
    file = path / checkpoint
  else:
    file = max(path.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  label = (name or path.name).removeprefix("Walking-Leap-")
  return Policy(label=label, task=task, checkpoint=file)


def fixed_commands(
  task: str, num_envs: int, hold: str | None = None
) -> tuple[ManagerBasedRlEnvCfg, dict[str, tuple[float, float, float]]]:
  """The task's play config without command resampling, and its edge commands.

  With ``hold``, the command range shrinks to that one command, so that every
  resample (a reset too) gives it.
  """
  cfg = load_env_cfg(task, play=True)
  twist = cfg.commands.get("twist")
  if not isinstance(twist, mdp.VelocityCommandCfg):
    raise SystemExit(f"{task} has no velocity command")
  r = twist.ranges
  commands = {
    "forward": (r.lin_vel_x[1], 0.0, 0.0),
    "back": (r.lin_vel_x[0], 0.0, 0.0),
    "thumb side": (0.0, r.lin_vel_y[1], 0.0),
    "ring side": (0.0, r.lin_vel_y[0], 0.0),
    "turn left": (0.0, 0.0, r.ang_vel_z[1]),
    "turn right": (0.0, 0.0, r.ang_vel_z[0]),
    "stand": (0.0, 0.0, 0.0),
  }
  if hold is not None:
    vx, vy, wz = commands[hold]
    twist.ranges = mdp.VelocityCommandCfg.Ranges(
      lin_vel_x=(vx, vx), lin_vel_y=(vy, vy), ang_vel_z=(wz, wz)
    )
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.rel_standing_envs = 0.0
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  return cfg, commands


def load_policy(env: RslRlVecEnvWrapper, policy: Policy, device: str):
  """The actor, as `play` loads it; None for the stand-still reference."""
  if policy.checkpoint is None:
    return None
  agent_cfg = load_rl_cfg(policy.task)
  runner_cls = load_runner_cls(policy.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(policy.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  return copy.deepcopy(runner.get_inference_policy(device=device))


def evaluate(task: str, policies: list[Policy], rollouts: int, device: str) -> dict:
  """Roll out the policies of one task side by side, a block of envs each."""
  per_policy = len(AXES) * rollouts
  cfg, commands = fixed_commands(task, len(policies) * per_policy)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device))
  base = env.unwrapped
  cmd = torch.tensor(list(commands.values()), device=device)
  cmd = cmd.repeat_interleave(rollouts, dim=0).repeat(len(policies), 1)
  term = base.command_manager.get_term("twist")
  term.vel_command_b[:] = cmd
  term.is_standing_env[:] = False
  actors = [load_policy(env, p, device) for p in policies]

  robot = base.scene["robot"]
  n = base.num_envs
  alive = torch.ones(n, dtype=torch.bool, device=device)
  vel_sum = torch.zeros(n, 3, device=device)
  counted = torch.zeros(n, device=device)
  obs = env.get_observations()
  with torch.inference_mode():
    for step in range(STEPS):
      actions = torch.zeros(n, base.action_manager.total_action_dim, device=device)
      for i, actor in enumerate(actors):
        if actor is not None:
          block = slice(i * per_policy, (i + 1) * per_policy)
          actions[block] = actor(obs)[block]
      obs, _, dones, extras = env.step(actions)
      alive &= ~(dones.bool() & ~extras["time_outs"].bool())
      if step >= SETTLE:
        v = torch.cat(
          [
            robot.data.root_link_lin_vel_b[:, :2],
            robot.data.root_link_ang_vel_b[:, 2:],
          ],
          dim=1,
        )
        vel_sum += v * alive.float()[:, None]
        counted += alive.float()
  env.close()

  results = {}
  for i, policy in enumerate(policies):
    results[policy.label] = {}
    for j, (name, axis) in enumerate(AXES.items()):
      start = i * per_policy + j * rollouts
      ids = slice(start, start + rollouts)
      ok = counted[ids] > 0
      mean_v = (vel_sum[ids][ok] / counted[ids][ok, None]).cpu()
      along = mean_v[:, axis if axis is not None else 0]
      target = commands[name][axis] if axis is not None else 0.0
      results[policy.label][name] = {
        "command": target,
        "achieved_mean": float(along.mean()) if len(along) else float("nan"),
        "achieved_std": float(along.std()) if len(along) > 1 else 0.0,
        "share": float(along.mean() / target) if target and len(along) else None,
        "fell": float((~alive[ids]).float().mean().cpu()),
        "checkpoint": str(policy.checkpoint) if policy.checkpoint else None,
      }
  return {"commands": commands, "results": results}


def view(policy: Policy, command: str, device: str) -> None:
  """One hand, one policy, one fixed command, in mjlab's native viewer."""
  cfg, commands = fixed_commands(policy.task, 1, hold=command)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device))
  actor = load_policy(env, policy, device)
  print(f"[eval_velocity] {policy.label}, {command}: {commands[command]}", flush=True)
  NativeMujocoViewer(env, actor).run()
  env.close()


def print_table(task: str, block: dict, rollouts: int) -> None:
  def unit(axis: int | None) -> tuple[str, float]:
    return ("deg/s", 57.2958) if axis == 2 else ("mm/s", 1000.0)

  labels = list(block["results"])
  print(f"\n{task}: velocity along the commanded axis ({rollouts} episodes, last 9 s)")
  print("command".ljust(24) + "".join(label.rjust(18) for label in labels))
  for name, axis in AXES.items():
    symbol, scale = unit(axis)
    target = block["commands"][name][axis] if axis is not None else 0.0
    head = f"{name} ({target * scale:+.0f} {symbol})" if axis is not None else name
    row = head.ljust(24)
    for label in labels:
      r = block["results"][label][name]
      share = f" {100 * r['share']:4.0f}%" if r["share"] is not None else ""
      row += f"{r['achieved_mean'] * scale:+7.1f}{share}".rjust(18)
    print(row)
  falls = {
    label: max(r["fell"] for r in block["results"][label].values()) for label in labels
  }
  print(
    "most falls in one command: "
    + ", ".join(f"{k} {100 * v:.0f}%" for k, v in falls.items())
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("runs", type=Path, nargs="+", help="run folders")
  parser.add_argument("--task", help="the runs' task, if their names do not tell")
  parser.add_argument("--checkpoint", help="file name, e.g. model_2999.pt")
  parser.add_argument("--rollouts", type=int, default=16)
  parser.add_argument("--no-stand-still", action="store_true")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--out", type=Path, default=ROOT / "outputs" / "eval" / "velocity.json"
  )
  parser.add_argument("--view", action="store_true", help="watch the first run instead")
  parser.add_argument("--command", choices=list(AXES), default="forward")
  args = parser.parse_args()

  policies = [read_run(path, args.task, args.checkpoint) for path in args.runs]
  if args.view:
    view(policies[0], args.command, args.device)
    return

  by_task: dict[str, list[Policy]] = {}
  for policy in policies:
    by_task.setdefault(policy.task, [])
    if not by_task[policy.task] and not args.no_stand_still:
      by_task[policy.task].append(Policy("stand still", policy.task, None))
    by_task[policy.task].append(policy)
  report = {
    task: evaluate(task, group, args.rollouts, args.device)
    for task, group in by_task.items()
  }
  for task, block in report.items():
    print_table(task, block, args.rollouts)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(
    json.dumps({"rollouts": args.rollouts, "tasks": report}, indent=2)
  )
  print(f"\nwrote {args.out}")


if __name__ == "__main__":
  main()
