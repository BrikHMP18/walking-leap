"""Does a scripted-crawl policy (Gait-v1 or its student) get up and take pushes?

Two tests on the task's play config (nominal hand, no random pushes or cables), each
next to a hand that holds the stance (action 0) as the reference:

  get up   every hand starts fallen as in training (tilted 57-115 deg about a random
           horizontal axis, dropped from 0.16 m); recovered = the command's own test
           (palm within 17 deg of upright and above 9 cm for 0.3 s) before the task
           gives up (``fallen_too_long``, 5 s)
  pushes   the hands crawl the training clip; after 2 s each gets one horizontal kick
           (an instant change of the palm's velocity, as training's pushes, which
           reach only 0.1 m/s per axis), in a random direction; fell = tilt past
           46 deg within the next 3 s; still down = fallen at the end (8 s after the
           kick) or reset by the task after 5 s down

  uv run python scripts/eval_survival.py outputs/hf/runs/<run> [--checkpoint model_5999.pt]
"""

import argparse
import json
import math
from pathlib import Path

import mjlab.tasks  # noqa: F401  (registers the walking_leap tasks)
import torch
from eval_velocity import Policy, load_policy, read_run
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

ROOT = Path(__file__).parents[1]
# m/s. A hand holding its stance did not fall from 0.3 m/s (3 %) or from spins of
# 8 rad/s about a horizontal axis (0 %); 0.5 m/s felled 22 %, 1.0 m/s 72 %.
KICKS = (0.5, 0.75, 1.0)
HANDS = 32  # per test and policy


def make_env(task: str, hands: int, fallen: bool, device: str) -> RslRlVecEnvWrapper:
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = hands
  cfg.seed = 0
  command = cfg.commands["motion"]
  if not hasattr(command, "fallen_start_prob"):
    raise SystemExit(f"{task} has no falls (Gait-v1 and its student do)")
  command.fallen_start_prob = 1.0 if fallen else 0.0
  return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device))


def act(actor, obs, base):
  if actor is None:
    return torch.zeros(
      base.num_envs, base.action_manager.total_action_dim, device=base.device
    )
  return actor(obs)


def getup(policy: Policy | None, task: str, device: str) -> dict:
  env = make_env(task, HANDS, fallen=True, device=device)
  base = env.unwrapped
  command = base.command_manager.get_term("motion")
  actor = load_policy(env, policy, device) if policy else None
  obs = env.get_observations()
  assert command.fallen.all()
  up_at = torch.full((HANDS,), float("nan"), device=device)
  gave_up = torch.zeros(HANDS, dtype=torch.bool, device=device)
  with torch.inference_mode():
    for step in range(round(6.0 / base.step_dt)):
      obs, _, dones, _ = env.step(act(actor, obs, base))
      gave_up |= dones.bool() & torch.isnan(up_at)
      newly = ~command.fallen & torch.isnan(up_at) & ~gave_up
      up_at[newly] = (step + 1) * base.step_dt
  env.close()
  ok = ~torch.isnan(up_at)
  return {
    "recovered": float(ok.float().mean()),
    "median_s": float(up_at[ok].median()) if ok.any() else None,
  }


def pushes(policy: Policy | None, task: str, kick: float, device: str) -> dict:
  env = make_env(task, HANDS, fallen=False, device=device)
  base = env.unwrapped
  command = base.command_manager.get_term("motion")
  robot = base.scene["robot"]
  actor = load_policy(env, policy, device) if policy else None
  gen = torch.Generator(device=device).manual_seed(0)
  heading = torch.rand(HANDS, generator=gen, device=device) * 2 * math.pi
  obs = env.get_observations()
  fell = torch.zeros(HANDS, dtype=torch.bool, device=device)
  gave_up = torch.zeros(HANDS, dtype=torch.bool, device=device)
  kick_step, steps = round(2.0 / base.step_dt), round(10.0 / base.step_dt)
  with torch.inference_mode():
    for step in range(steps):
      if step == kick_step:
        vel = robot.data.root_link_vel_w.clone()
        vel[:, 0] += kick * torch.cos(heading)
        vel[:, 1] += kick * torch.sin(heading)
        robot.write_root_link_velocity_to_sim(vel)
      obs, _, dones, _ = env.step(act(actor, obs, base))
      gave_up |= dones.bool()  # fallen_too_long: down for 5 s, the task reset it
      if kick_step <= step < kick_step + round(3.0 / base.step_dt):
        fell |= command.fallen
  down_at_end = command.fallen | gave_up
  env.close()
  return {
    "fell": float(fell.float().mean()),
    "still_down": float(down_at_end.float().mean()),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("runs", nargs="+", type=Path)
  parser.add_argument("--task", default=None)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  args = parser.parse_args()

  for path in args.runs:
    policy = read_run(path, args.task, args.checkpoint)
    results = {}
    for label, p in ((policy.label, policy), ("hold the stance", None)):
      r = {"get up": getup(p, policy.task, args.device)}
      for kick in KICKS:
        r[f"kick {kick} m/s"] = pushes(p, policy.task, kick, args.device)
      results[label] = r
    out = ROOT / "outputs" / "eval" / f"survival_{policy.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
      json.dumps({"checkpoint": str(policy.checkpoint), **results}, indent=1)
    )
    print(f"\n{policy.label} ({policy.checkpoint.name}), {HANDS} hands per test")
    for label, r in results.items():
      g = r["get up"]
      med = f", median {g['median_s']:.1f} s" if g["median_s"] is not None else ""
      print(f"  {label}: got up {g['recovered']:.0%}{med}")
      for kick in KICKS:
        k = r[f"kick {kick} m/s"]
        print(
          f"    kick {kick} m/s: fell {k['fell']:.0%}, still down at the end "
          f"{k['still_down']:.0%}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
  main()
