"""Reference clips from the scripted crawl, tested on random simulated hands.

The crawl of scripted_gaits.py (``Crawler``, the thumb sliding along) becomes motion
clips for imitation, one per command: forward, back, left, right, turn left, turn
right, and stand (the targets held). Every cycle is planned from where the previous
one ends, so cycles chain in any order. The targets are played open loop at 50 Hz on
the real hand's model, as ``clips.replay`` plays the real clips; the palm follows
from physics. Before any clip is used for training, the same targets are played on
random simulated hands (ranges from docs/next_training.md), as the real crawl was.

  uv run python scripts/gait_clips.py               # write the clips, test 45 hands
  uv run python scripts/gait_clips.py --hands 10 --cycles 2

Writes outputs/clips/gait_<command>.npz (a first cycle in place that leaves the
stance, then ``--cycles`` of the command) and gait_mix.npz (a random sequence of
commands), all git-ignored, and prints the test.
"""

import argparse
import copy
from pathlib import Path

import mujoco
import numpy as np
import scripted_gaits as gaits

from walking_leap import clips, gait

ROOT = Path(__file__).parents[1]
COMMANDS = gait.GAIT_COMMANDS


def random_hand(model: mujoco.MjModel, rng: np.random.Generator) -> mujoco.MjModel:
  """The real hand's model with the ranges of docs/next_training.md: stiffness x0.7-1.1,
  torque cap x0.8-1.2, fingertip sliding friction 0.5-1.0 and torsion 0.001-0.005 m,
  other contacts' torsion 0.01-0.03 m, palm centre of mass +-8 mm. The floor's
  torsion is lowered to 0.001 m so that the fingertips' own value counts (MuJoCo
  takes the larger of the two geoms')."""
  m = copy.deepcopy(model)
  m.actuator_gainprm[:, 4] *= rng.uniform(0.7, 1.1, m.nu)  # position gain, V/rad
  m.actuator_forcerange *= rng.uniform(0.8, 1.2, (m.nu, 1))
  floor = m.geom("floor").id
  m.geom_friction[floor, 1] = 0.001
  for g in range(m.ngeom):
    if not m.geom_contype[g] or g == floor:
      continue
    if (m.geom(g).name or "").endswith("_tip"):
      m.geom_friction[g, 0] = rng.uniform(0.5, 1.0)
      m.geom_friction[g, 1] = rng.uniform(0.001, 0.005)
    else:
      m.geom_friction[g, 1] = rng.uniform(0.01, 0.03)
  m.body_ipos[m.body("palm").id] += rng.uniform(-0.008, 0.008, 3)
  return m


def play(model: mujoco.MjModel, keys: np.ndarray, settle_s: float = 0.5) -> dict:
  """Play the targets open loop from the folded-thumb stance (as ``clips.replay``);
  whether the hand stays up, how far it goes (start heading frame) and turns."""
  d = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, d, model.key("init_state").id)
  d.qpos[7:] = keys[0]
  d.ctrl[:] = keys[0]
  mujoco.mj_forward(model, d)
  for _ in range(round(settle_s / model.opt.timestep)):
    mujoco.mj_step(model, d)
  palm = model.body("palm").id
  start = d.xpos[palm][:2].copy()
  heading = d.xmat[palm].reshape(3, 3)[:2, 0]
  yaw0 = float(np.arctan2(heading[1], heading[0]))
  steps = round(gaits.KEYFRAME_S / model.opt.timestep)
  fell, palm_touch, n = False, 0, 0
  for target in keys:
    d.ctrl[:] = target
    for _ in range(steps):
      mujoco.mj_step(model, d)
      n += 1
      palm_touch += "palm/other" in gaits.touching(model, d)
    upright = d.xmat[palm].reshape(3, 3)[2, 2]
    if upright < np.cos(np.radians(45.0)) or d.xpos[palm][2] < 0.06:
      fell = True
      break
  heading = d.xmat[palm].reshape(3, 3)[:2, 0]
  turned = np.degrees(np.arctan2(heading[1], heading[0]) - yaw0)
  c, s = np.cos(yaw0), np.sin(yaw0)
  moved = np.array([[c, s], [-s, c]]) @ (d.xpos[palm][:2] - start)
  return dict(
    fell=fell,
    forward_mm=1e3 * moved[0],
    left_mm=1e3 * moved[1],
    turned_deg=float((turned + 180.0) % 360.0 - 180.0),
    palm_touch_pct=100.0 * palm_touch / n,
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--hands", type=int, default=45)
  parser.add_argument("--cycles", type=int, default=3)
  parser.add_argument("--mix", type=int, default=12, help="commands in gait_mix")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "clips")
  parser.add_argument(
    "--smooth", action="store_true", help="plan with gait.SMOOTH_IK (gait_smooth.npz)"
  )
  args = parser.parse_args()

  hand = clips.RealHand()
  smooth = gait.SMOOTH_IK if args.smooth else 0.0
  crawler = gait.Crawler(gaits.build(), thumb_drag=True, smooth=smooth)
  rng = np.random.default_rng(args.seed)
  sequences = {name: [name] * args.cycles for name in COMMANDS}
  mix = [str(rng.choice(list(COMMANDS))) for _ in range(args.mix)]
  sequences["mix"] = mix
  print("mix:", ", ".join(mix))
  hands = [
    random_hand(hand.model, np.random.default_rng(args.seed + 1 + i))
    for i in range(args.hands)
  ]

  print(
    f"{'clip':11s} {'s':>5s} | nominal: fwd mm, left mm, turn deg | "
    f"{args.hands} random hands: standing, fwd mm, left mm, turn deg | fastest joint"
  )
  for name, commands in sequences.items():
    keys, _ = gait.plan(crawler, commands)
    motion = clips.replay(f"gait_{name}", hand, q=keys)
    clips.save(motion, hand, args.out / f"gait_{name}.npz")
    nominal = play(hand.model, keys)
    runs = [play(m, keys) for m in hands]
    up = [r for r in runs if not r["fell"]]

    def span(key, runs=up):
      values = [r[key] for r in runs]
      return f"{min(values):+.0f}..{max(values):+.0f}" if values else "-"

    print(
      f"{name:11s} {len(keys) * gaits.KEYFRAME_S:5.1f} | "
      f"{'FELL ' if nominal['fell'] else ''}{nominal['forward_mm']:+5.0f} "
      f"{nominal['left_mm']:+5.0f} {nominal['turned_deg']:+5.0f} | "
      f"{len(up):2d}/{len(runs)} {span('forward_mm'):>10s} {span('left_mm'):>10s} "
      f"{span('turned_deg'):>9s} | {np.abs(np.diff(keys, axis=0)).max() * 50:4.1f} rad/s"
    )
  print(f"wrote {args.out}/gait_*.npz")


if __name__ == "__main__":
  main()
