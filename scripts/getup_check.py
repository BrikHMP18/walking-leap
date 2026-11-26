"""Which falls can the real hand get up from? A search, before training for it.

For each fall (on its thumb side, ring side, fingertips, wrist, back) the hand is
dropped tilted on the real hand's model and left to settle. Then a cross-entropy
search looks for joint targets that stand it up: KEYS keyframes over SECONDS,
linearly interpolated, played open loop. The score is how upright the palm ends
(1 upright, 0 upside down) and whether it stands at a working height. A fall whose
best score stays low may still be possible with feedback, but open loop found
nothing; one that reaches ~1 is physically possible for these motors (350 mA).

  uv run python scripts/getup_check.py            # all falls, ~2-5 min on 8 cores
  uv run python scripts/getup_check.py --iters 10 --pop 32
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

from walking_leap import clips

KEYS, SECONDS = 6, 3.0
# Falls: tilt about the palm-frame axis (rad). +x is wrist to fingertips, +y the
# thumb side, so a turn about +x rolls the hand onto its ring side.
FALLS = {
  "ring side": ((1.0, 0.0, 0.0), np.radians(90.0)),
  "thumb side": ((1.0, 0.0, 0.0), np.radians(-90.0)),
  "fingertips": ((0.0, 1.0, 0.0), np.radians(80.0)),
  "wrist": ((0.0, 1.0, 0.0), np.radians(-80.0)),
  "back": ((1.0, 0.0, 0.0), np.radians(180.0)),
}

_hand = None


def hand() -> clips.RealHand:
  global _hand
  if _hand is None:
    _hand = clips.RealHand()
  return _hand


def fallen_state(name: str) -> np.ndarray:
  """The hand dropped from 0.16 m with the fall's tilt, settled for 1.5 s."""
  m, d = hand().model, hand().data
  mujoco.mj_resetDataKeyframe(m, d, m.key("init_state").id)
  axis, angle = FALLS[name]
  turn = np.zeros(4)
  mujoco.mju_axisAngle2Quat(turn, np.array(axis, dtype=float), angle)
  quat = np.zeros(4)
  mujoco.mju_mulQuat(quat, d.qpos[3:7].copy(), turn)
  d.qpos[3:7] = quat
  d.qpos[2] = 0.16
  d.ctrl[:] = d.qpos[7:]
  for _ in range(round(1.5 / m.opt.timestep)):
    mujoco.mj_step(m, d)
  return np.concatenate([d.qpos, d.qvel])


def score(args) -> float:
  """Play keyframe targets from the fallen state; 1 if the palm ends upright at a
  working height (mean over the last 0.5 s)."""
  state, keys = args
  m, d = hand().model, hand().data
  mujoco.mj_resetData(m, d)
  d.qpos[:], d.qvel[:] = state[: m.nq], state[m.nq :]
  mujoco.mj_forward(m, d)
  start = d.ctrl.copy()
  start[:] = d.qpos[7:]
  points = np.vstack([start, keys])
  times = np.linspace(0.0, SECONDS, len(points))
  steps = round(SECONDS / m.opt.timestep)
  palm = m.body("palm").id
  tail, total = round(0.5 / m.opt.timestep), 0.0
  clock = np.arange(steps) * m.opt.timestep
  targets = np.stack([np.interp(clock, times, points[:, j]) for j in range(start.size)])
  for i in range(steps):
    d.ctrl[:] = targets[:, i]
    mujoco.mj_step(m, d)
    if i >= steps - tail:
      upright = (1.0 + d.xmat[palm][8]) / 2.0
      standing = float(0.08 < d.xpos[palm][2] < 0.20)
      total += upright * (0.7 + 0.3 * standing)
  return total / tail


def search(name: str, iters: int, pop: int, workers: int, seed: int) -> dict:
  m = hand().model
  lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
  state = fallen_state(name)
  rng = np.random.default_rng(seed)
  mean = np.tile(state[7 : m.nq], (KEYS, 1))
  std = np.full_like(mean, 0.8)
  best, best_keys = -1.0, mean
  start_score = score((state, mean))
  with ProcessPoolExecutor(workers) as ex:
    for _ in range(iters):
      samples = np.clip(mean + std * rng.standard_normal((pop, *mean.shape)), lo, hi)
      scores = np.array(list(ex.map(score, [(state, s) for s in samples])))
      elite = samples[np.argsort(scores)[-max(2, pop // 6) :]]
      mean, std = elite.mean(0), elite.std(0) + 0.02
      if scores.max() > best:
        best, best_keys = float(scores.max()), samples[int(scores.argmax())]
  palm_up = state[: m.nq]
  quat = palm_up[3:7]
  tilt = np.degrees(np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1)))
  return dict(
    fall=name,
    settled_tilt_deg=round(float(tilt)),
    holding=round(start_score, 2),
    best=round(best, 2),
    keys=best_keys,
  )


def main() -> None:
  global KEYS, SECONDS
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--iters", type=int, default=20)
  parser.add_argument("--pop", type=int, default=48)
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--falls", nargs="*", default=list(FALLS), choices=list(FALLS))
  parser.add_argument("--keys", type=int, default=KEYS)
  parser.add_argument("--seconds", type=float, default=SECONDS)
  args = parser.parse_args()
  KEYS, SECONDS = args.keys, args.seconds
  print("fall         settled tilt | score holding still -> best found (1 = up)")
  for name in args.falls:
    r = search(name, args.iters, args.pop, args.workers, args.seed)
    verdict = "can get up" if r["best"] > 0.9 else "not found"
    print(
      f"{name:12s} {r['settled_tilt_deg']:5d} deg   | "
      f"{r['holding']:.2f} -> {r['best']:.2f}  {verdict}",
      flush=True,
    )


if __name__ == "__main__":
  main()
