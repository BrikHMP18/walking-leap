"""The scripted crawl as clips for the real hand's loop and panel, one per command.

The planner of ``gait.py`` (smoothed: gait.SMOOTH_IK) walks CYCLES forward cycles from the stand; then the targets
ease back to the stand over RETURN_S and hold (the loop wants every clip to start and
end on the stand; after walking the digits are elsewhere, up to 0.55 rad). Writes:

  outputs/clips/gait_forward.npz   mjlab's motion format: the targets replayed on the
                                   real hand's model; a gait policy's reference
                                   (``real/export_policy.py --motion``)
  real/clips/gait_forward.npz      the loop's clip: the targets (``q``), the replay's
                                   palm "down" (``grav``) and path (``palm``), ``meta``

and plays the targets open loop on random simulated hands (gait_clips.py's ranges),
measuring what the loop guards: tilt (it stops at 40 deg) and the palm's "down" against
the clip's (it stops past 15 deg).

  uv run python scripts/gait_real_clip.py                     # every command but stand
  uv run python scripts/gait_real_clip.py --commands forward
"""

import argparse
import json
import sys
from pathlib import Path

import gait_clips
import mujoco
import numpy as np

from walking_leap import clips, gait

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "real"))
import wlc  # noqa: E402

# 3 cycles stayed up on 45 of 45 random hands; 6 tipped 13 of them past 20 deg in the
# fifth cycle (open loop drifts; keeping a long walk up is the policy's job).
CYCLES = 3
RETURN_S, HOLD_S = 3.0, 1.0


def ease(u: np.ndarray) -> np.ndarray:
  return u**3 * (10 - 15 * u + 6 * u * u)


def targets(model: mujoco.MjModel, command: str = "forward") -> np.ndarray:
  crawler = gait.Crawler(model, thumb_drag=True, smooth=gait.SMOOTH_IK)
  keys, _ = gait.plan(crawler, [command] * CYCLES)
  stand = wlc.Clip("stand_up").q[-1]
  n = round(RETURN_S / gait.KEYFRAME_S)
  back = keys[-1] + ease(np.linspace(0, 1, n + 1)[1:, None]) * (stand - keys[-1])
  hold = np.repeat(stand[None], round(HOLD_S / gait.KEYFRAME_S), axis=0)
  return np.vstack([keys, back, hold])


def palm_down(quat_wxyz: np.ndarray) -> np.ndarray:
  w, x, y, z = quat_wxyz.T
  return -np.stack(
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1
  )


def play(model: mujoco.MjModel, q: np.ndarray, grav: np.ndarray) -> dict:
  """Open loop from the stance, as clips.replay: worst tilt and distance from the
  clip's "down" (deg), whether it fell, how far it went (mm, start heading)."""
  d = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, d, model.key("init_state").id)
  d.qpos[7:], d.ctrl[:] = q[0], q[0]
  mujoco.mj_forward(model, d)
  for _ in range(round(0.5 / model.opt.timestep)):
    mujoco.mj_step(model, d)
  palm = model.body("palm").id
  x0 = d.xpos[palm][:2].copy()
  steps = round(gait.KEYFRAME_S / model.opt.timestep)
  tilt, off = 0.0, 0.0
  for target, g in zip(q, grav, strict=True):
    d.ctrl[:] = target
    for _ in range(steps):
      mujoco.mj_step(model, d)
    down = -d.xmat[palm].reshape(3, 3)[2]
    tilt = max(tilt, float(np.degrees(np.arccos(np.clip(-down[2], -1, 1)))))
    off = max(off, float(np.degrees(np.arccos(np.clip(down @ g, -1, 1)))))
  return dict(
    tilt=tilt,
    off=off,
    fell=tilt > 45 or d.xpos[palm][2] < 0.06,
    moved_mm=1e3 * float(np.linalg.norm(d.xpos[palm][:2] - x0)),
  )


def write(hand: clips.RealHand, command: str) -> None:
  """real/clips/gait_<command>.npz and outputs/clips/gait_<command>.npz, checked."""
  name = f"gait_{command}"
  q = targets(hand.model, command)
  motion = clips.replay(name, hand, q=q)
  clips.save(motion, hand, ROOT / "outputs" / "clips" / f"{name}.npz")

  quat, pos = motion.qpos[:, 3:7], motion.qpos[:, :3]
  grav = palm_down(quat)
  w, x, y, z = quat.T
  heading = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
  c, s = np.cos(heading[0]), np.sin(heading[0])
  dxy = pos[:, :2] - pos[0, :2]
  path = np.stack(
    [
      c * dxy[:, 0] + s * dxy[:, 1],
      -s * dxy[:, 0] + c * dxy[:, 1],
      heading - heading[0],
    ],
    1,
  )
  speed = np.abs(np.diff(q, axis=0)).max() * wlc.HZ
  meta = dict(
    joint_names=list(wlc.JOINT_NAMES),
    fps=wlc.HZ,
    start="stand",
    end="stand",
    title=f"Gait {command.replace('_', ' ')} x{CYCLES}",
    note=f"scripts/gait_real_clip.py: gait.py's smooth planner, {CYCLES} {command}"
    f" cycles, then {RETURN_S:.0f} s back to the stand; grav and palm from the replay"
    " on clips.RealHand",
  )
  out = ROOT / "real" / "clips" / f"{name}.npz"
  np.savez_compressed(
    out, q=q.astype(np.float32), grav=grav.astype(np.float32),
    palm=path.astype(np.float32), meta=json.dumps(meta),
  )  # fmt: skip
  wlc.Clip(str(out))  # the loop's own checks
  stand = wlc.Clip("stand_up").q[-1]
  runs = [
    play(gait_clips.random_hand(hand.model, np.random.default_rng(1 + i)), q, grav)
    for i in range(45)
  ]
  tilt = np.array([r["tilt"] for r in runs])
  off = np.array([r["off"] for r in runs])
  print(
    f"{name:16s} {len(q) / wlc.HZ:4.1f} s, {speed:.1f} rad/s, ends"
    f" {np.abs(q[-1] - stand).max():.3f} rad from the stand | replay"
    f" {path[-1, 0] * 1e3:+4.0f} fwd {path[-1, 1] * 1e3:+4.0f} left mm"
    f" {np.degrees(path[-1, 2]):+4.0f} deg | 45 random hands: {45 - sum(r['fell'] for r in runs)}"
    f" up, tilt <= {tilt.max():.1f} deg, down <= {off.max():.1f} deg from the clip's"
    f" (the loop stops at 40 and 15)",
    flush=True,
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
    "--commands", nargs="+", default=list(gait.GAIT_COMMANDS)[:-1], help="not stand"
  )
  args = parser.parse_args()
  hand = clips.RealHand()
  for command in args.commands:
    write(hand, command)


if __name__ == "__main__":
  main()
