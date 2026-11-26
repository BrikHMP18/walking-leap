"""A long crawl clip, to watch a crawl policy keep going (viewing only).

The crawl tasks train on assets/clips/crawl.npz: three cycles (28 s) of the real
hand's scripted crawl, after which mjlab puts the hand back at the clip's start. This
repeats one cycle of the six-cycle recording (docs/data/real_clips/crawl6.npz), cut
where the joints move least, and replays it open loop on the real hand's model as
``clips.main`` makes crawl.npz. The policies see only the clip's joints and gravity,
which repeat every cycle; the palm's height and tilt still end an episode.

  uv run python scripts/crawl_loop.py               # 40 cycles, 6.2 min
  uv run python scripts/crawl_loop.py --cycles 100
  uv run play Walking-Leap-Motion-Crawl-Flat-v0 --viewer native --num-envs 1 \
    --motion-file outputs/clips/crawl_loop.npz --checkpoint-file <run>/model_2999.pt

Writes outputs/clips/crawl_loop.npz (git-ignored) and prints how the replay went.
"""

import argparse
from pathlib import Path

import numpy as np

from walking_leap import clips

ROOT = Path(__file__).parents[1]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--cycles", type=int, default=40)
  parser.add_argument(
    "--out", type=Path, default=ROOT / "outputs" / "clips" / "crawl_loop.npz"
  )
  args = parser.parse_args()

  # crawl6 is crawl plus three cycles: the cycle is 442 2/3 frames (8.85 s). Repeat
  # a whole number of frames, cut where the seam (and the joints' speed) is smallest.
  q6 = np.load(clips.REAL_CLIPS_DIR / "crawl6.npz")["q"]
  q3 = np.load(clips.REAL_CLIPS_DIR / "crawl.npz")["q"]
  cycle = (len(q6) - len(q3)) / 3
  speed = np.abs(np.diff(q6, axis=0)).max(axis=1)
  best = (np.inf, 0, 0)
  for period in (int(np.floor(cycle)), int(np.ceil(cycle))):
    for cut in range(period, len(q6) - 2 * period):
      seam = np.abs(q6[cut + period] - q6[cut]).max()
      best = min(best, (seam + speed[cut] + speed[cut + period - 1], cut, period))
  _, cut, period = best
  q = np.concatenate([q6[:cut]] + [q6[cut : cut + period]] * args.cycles)
  seam = np.abs(q6[cut + period] - q6[cut]).max()
  print(
    f"cycle {period} frames, cut at frame {cut}, seam {seam:.4f} rad; "
    f"{len(q)} frames = {len(q) / clips.FPS / 60:.1f} min"
  )

  hand = clips.RealHand()
  motion = clips.replay("crawl_loop", hand, q=q)

  # The replay's palm: height, tilt, heading and distance per cycle.
  pos, (w, x, y, z) = motion.qpos[:, :3], motion.qpos[:, 3:7].T
  tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0)))
  yaw = np.degrees(np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
  starts = cut + period * np.arange(args.cycles)
  step = 1e3 * np.linalg.norm(np.diff(pos[starts, :2], axis=0), axis=1)
  print(
    f"replay: palm {1e3 * pos[:, 2].min():.1f}-{1e3 * pos[:, 2].max():.1f} mm high, "
    f"tilt up to {tilt.max():.1f} deg, {step.min():.1f}-{step.max():.1f} mm a cycle, "
    f"{1e3 * np.linalg.norm(pos[-1, :2] - pos[0, :2]):.0f} mm in all, "
    f"turned {yaw[-1] - yaw[0]:.0f} deg"
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  clips.save(motion, hand, args.out)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
