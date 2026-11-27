"""Watch a trained policy in mjlab's MuJoCo viewer, its name written on the window.

As ``uv run play`` (the task's play config, the run's last checkpoint), plus a label in
the top right corner: which policy this is and what it does, so that several windows
side by side are easy to tell apart. The task is read from the run's name.

  uv run python scripts/watch.py outputs/hf/runs/<run>             # e.g. the Getup run
  uv run python scripts/watch.py outputs/hf/runs/<run> --fallen    # every reset starts fallen

Enter resets (with --fallen: drops it again), Space pauses; double-click the palm, then
Ctrl + right-drag pushes it.
"""

import argparse
from pathlib import Path

import mjlab.tasks  # noqa: F401  (registers the walking_leap tasks)
import mujoco
import torch
from eval_velocity import load_policy, read_run
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer

# What each task's policy does, for the label (one short line).
DOES = {
  "Walking-Leap-Motion-Crawl-Flat-v0": "imitates the real hand's crawl",
  "Walking-Leap-Motion-Gait-Flat-v0": "imitates the scripted crawl",
  "Walking-Leap-Motion-Gait-Flat-v1": "scripted crawl, randomized, falls",
  "Walking-Leap-Motion-Gait-Flat-v2": "smooth scripted crawl, strong pushes",
  "Walking-Leap-Motion-Gait-Flat-v3": "smooth scripted crawl, light pushes",
  "Walking-Leap-Motion-Gait-Flat-v4": "smooth scripted crawl, moderate pushes",
  "Walking-Leap-Motion-Gait-Flat-Student-v2": "keyboard velocity, student of Gait-v3",
  "Walking-Leap-Motion-Gait-Flat-Student-v3": "keyboard velocity + gait clock (student)",
  "Walking-Leap-Motion-Gait-Flat-Velocity-v6": "keyboard velocity + gait clock (RL)",
  "Walking-Leap-Motion-Gait-Flat-Student-v4": "keyboard + clock, robust teacher",
  "Walking-Leap-Motion-Gait-Flat-Getup-v0": "gets up from a fall",
  "Walking-Leap-Motion-Gait-Flat-Student-v1": "keyboard velocity, gets up, takes pushes",
  "Walking-Leap-Motion-Gait-Flat-Velocity-v0": "keyboard velocity (RL), gets up, pushes",
  "Walking-Leap-Motion-Gait-Flat-Velocity-v1": "keyboard velocity (RL, split reward)",
  "Walking-Leap-Motion-Gait-Flat-Velocity-v2": "keyboard velocity (RL, explores more)",
  "Walking-Leap-Motion-Gait-Flat-Velocity-v3": "keyboard velocity (RL, light pushes)",
  "Walking-Leap-Motion-Gait-Flat-Velocity-v4": "keyboard velocity (RL, clean world)",
  "Walking-Leap-Motion-Gait-Flat-Velocity-v5": "keyboard velocity, imitating the clip",
}


class LabeledViewer(NativeMujocoViewer):
  """mjlab's native viewer with a label box: ``title`` and ``subtitle`` (top right),
  and ``status()`` (a callable, refreshed every frame) under them."""

  def __init__(self, env, policy, title: str, subtitle: str, status=None, **kwargs):
    super().__init__(env, policy, **kwargs)
    self.title, self.subtitle, self.status = title, subtitle, status

  def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
    # One set_texts call a frame with both boxes (mjlab's status, and the label): two
    # calls, its then ours, made the corner flicker between them.
    status = self.get_status()
    capped = " [CAPPED]" if status.capped else ""
    lines = f"{self.title}\n{self.subtitle}"
    if self.status is not None:
      lines += f"\n{self.status()}"
    viewer.set_texts(
      [
        (
          mujoco.mjtFontScale.mjFONTSCALE_150.value,
          mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
          "Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT",
          f"{self.env_idx + 1}/{self.env.num_envs}\n{status.step_count}\n"
          f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
          f"{status.speed_label}\n{status.target_realtime:.2f}x\n"
          f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)",
        ),
        (
          mujoco.mjtFontScale.mjFONTSCALE_200.value,
          mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
          lines,
          "",
        ),
      ]
    )


def label(task: str, run: Path, checkpoint: Path) -> tuple[str, str]:
  """(title, subtitle): the policy's short name and what it does, the run and file."""
  short = task.removeprefix("Walking-Leap-Motion-").removeprefix("Walking-Leap-")
  return f"{short}: {DOES.get(task, task)}", f"{run.name} / {checkpoint.name}"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("run", type=Path)
  parser.add_argument("--task", default=None, help="default: from the run's name")
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--fallen", action="store_true", help="every reset starts fallen")
  parser.add_argument(
    "--motion", type=Path, default=None, help="another clip for a tracking policy"
  )
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  args = parser.parse_args()
  policy = read_run(args.run, args.task, args.checkpoint)
  cfg = load_env_cfg(policy.task, play=True)
  cfg.scene.num_envs = 1
  if args.fallen:
    cfg.commands["motion"].fallen_start_prob = 1.0
  if args.motion is not None:
    cfg.commands["motion"].motion_file = str(args.motion)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device))
  actor = load_policy(env, policy, args.device)
  title, subtitle = label(policy.task, args.run, policy.checkpoint)
  if args.motion is not None:
    subtitle += f" / clip {args.motion.stem}"
  print(f"[watch] {title} ({subtitle})", flush=True)
  LabeledViewer(env, actor, title, subtitle).run()
  env.close()


if __name__ == "__main__":
  main()
