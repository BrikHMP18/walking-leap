"""The dead crawl, lying: the hand never stands up. Index, middle and ring raise, straighten
in the air, plant at full reach and curl, dragging the palm along the table; the thumb
stays folded. Writes the loop's clip (real/clips/crawl_dead.npz) from the twin's physics.

  uv run python real/make_crawl_dead.py            # write the clip
  uv run python real/make_crawl_dead.py --check    # the same motion on 12 simulated hands

Physics: twin.Sim (the hand as built on stock fingertips, the servos a PD capped at 350 mA,
targets one 10 ms half-tick late), with the sliding friction of the palm and phalanges set
to PALM_MU, a guess: it was never measured, and the result depends on it. In that twin the
drag is weak: lying, about 70 % of the hand's 0.80 kg is on the palm, and three fingers at
the cap barely move it (from -14 to +28 mm in four cycles over the checked hands). The
clip's "down" and path are the twin's at PALM_MU; the other hands stay within 5 deg of it.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import twin  # noqa: E402
import wlc  # noqa: E402

J = wlc.JOINT_NAMES
I = {n: i for i, n in enumerate(J)}  # noqa: E741
HZ = wlc.HZ
FINGERS = ("if", "mf", "rf")
PALM_MU = 0.5  # printed plastic on a table: a guess (the twin keeps MuJoCo's 1.0)
P = dict(  # rad for the joints, s for the times
  lift=-0.2,  # mcp while in the air
  reach_pip=-0.25,  # pip straightened in the air
  plant=(0.15, 0.1, 0.1),  # mcp, pip, dip: the tip down at full reach
  pull=(0.3, 0.8, 0.5),  # curled: the tip stays, the palm comes
  bias=-0.2,  # the pull's size: index 1 - bias, middle 1, ring 1 + bias
  t_lift=0.8,
  t_reach=0.7,
  t_plant=0.5,
  t_pull=0.8,
  stagger=0.12,  # index, middle, ring
  cycles=4,
)


def ease(u):
  u = np.clip(u, 0.0, 1.0)
  return u**3 * (10 - 15 * u + 6 * u * u)


def keys(f, rest):
  """One finger's (mcp, pip, dip) over a cycle: [(seconds, value)], from its pose lying."""
  k = {"if": 1 - P["bias"], "mf": 1.0, "rf": 1 + P["bias"]}[f]
  plant = np.array(P["plant"])
  pull = plant + k * (np.array(P["pull"]) - plant)
  # Raise first (the knuckle up, the curl half undone) so the tip leaves the table before
  # it moves forward: a tip sliding forward pushes the palm back.
  raise_ = np.array([P["lift"], 0.5 * pull[1], 0.5 * pull[2]])
  up = np.array([P["lift"], P["reach_pip"], 0.0])
  return [
    (P["t_lift"], raise_),
    (P["t_reach"], up),
    (P["t_plant"], plant),
    (P["t_pull"], pull),
  ]


def targets():
  """The clip's targets at 50 Hz: still, the cycles, back to lying, still. The side-to-side
  knuckles go parallel while crawling (spread, the hand turns more)."""
  lying = wlc.Clip("stand_up").q[0].copy()
  cycle = sum(int(round(t * HZ)) for t, _ in keys("mf", lying))
  lead, start, ramp, tail = (
    int(round(P["stagger"] * HZ)),
    int(0.8 * HZ),
    int(0.8 * HZ),
    int(1.6 * HZ),
  )
  n = start + P["cycles"] * cycle + 2 * lead + tail
  q = np.repeat(lying[None], n, 0)
  for j, f in enumerate(FINGERS):
    cols = [I[f + "_mcp"], I[f + "_pip"], I[f + "_dip"]]
    rest = lying[cols]
    t, pts = start + j * lead, [(start + j * lead, rest)]
    for _ in range(P["cycles"]):
      for dur, val in keys(f, rest):
        t += int(round(dur * HZ))
        pts.append((t, val))
    pts.append((t + ramp, rest))
    for (ta, a), (tb, b) in zip(pts[:-1], pts[1:], strict=True):
      for k in range(ta, min(tb, n)):
        q[k, cols] = a + (b - a) * ease((k - ta) / max(1, tb - ta))
  w = np.zeros(n)
  end = n - tail + ramp
  w[start : end - ramp] = 1.0
  w[start : start + ramp] = ease(np.arange(ramp) / ramp)
  w[end - ramp : end] = ease(1 - np.arange(ramp) / ramp)
  for f in FINGERS:
    q[:, I[f + "_rot"]] *= 1 - w
  return q


def simulate(q, kp=1.6, mu=0.6, palm_mu=PALM_MU):
  """The twin's physics: per frame the palm's x, y, heading and "down" in its axes, the
  tilt, and the longest any motor stayed at its cap (s)."""
  s = twin.Sim(kp=kp, mu=mu)
  for i in range(s.m.ngeom):
    if s.m.geom_contype[i] and i != s.floor and not s.m.geom(i).name.endswith("_tip"):
      s.m.geom_friction[i, 0] = palm_mu
  s.lie(q[0])
  half = int(round(0.01 / s.m.opt.timestep))
  prev, rows, sat, worst = q[0], [], np.zeros(16), 0.0
  for k in range(len(q)):
    for target in (prev, q[k]):
      for _ in range(half):
        s.step(target)
    prev = q[k]
    R = s.d.xmat[s.palm].reshape(3, 3)
    rows.append((*s.pose(), *(-R[2, :])))
    sat = np.where(np.abs(s.tau) > 0.94 * s.cap, sat + 1 / HZ, 0.0)
    worst = max(worst, float(sat.max()))
  a = np.array(rows)
  tilt = float(np.degrees(np.arccos(np.clip(-a[:, 5], -1, 1))).max())
  return a, tilt, worst


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument(
    "--check", action="store_true", help="12 simulated hands, nothing written"
  )
  a = ap.parse_args()
  q = targets()
  speed = float(np.abs(np.diff(q, axis=0)).max() * HZ)
  print(f"{len(q) / HZ:.1f} s, fastest joint {speed:.2f} rad/s")
  if a.check:
    for kp, mu, pm in itertools.product((1.6, 3.0), (0.4, 0.8), (0.3, 0.5, 1.0)):
      p, tilt, sat = simulate(q, kp, mu, pm)
      d = (p[-1, :2] - p[0, :2]) * 1e3
      print(f"kp {kp} tip mu {mu} palm mu {pm}: {d[0]:+.0f} mm x, {d[1]:+.0f} mm y, "
            f"{np.degrees(p[-1, 2] - p[0, 2]):+.0f} deg, tilt <= {tilt:.0f}, at the cap <= {sat:.2f} s")  # fmt: skip
    return 0
  p, tilt, sat = simulate(q)
  c, s = np.cos(p[0, 2]), np.sin(p[0, 2])
  d = p[:, :2] - p[0, :2]
  palm = np.stack(
    [c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], p[:, 2] - p[0, 2]], 1
  )
  grav = p[:, 3:6] / np.linalg.norm(p[:, 3:6], axis=1, keepdims=True)
  meta = dict(
    joint_names=list(J), fps=HZ, start="lying", end="lying", title="Dead crawl",
    note=(f"lying, never standing: index, middle and ring raise, reach, plant and curl, dragging "
          f"the palm; {P['cycles']} cycles; plan from the twin with palm friction {PALM_MU} (a guess)"),
  )  # fmt: skip
  out = f"{HERE}/clips/crawl_dead.npz"
  np.savez_compressed(out, q=q.astype(np.float32), grav=grav.astype(np.float32),
                      palm=palm.astype(np.float32), meta=json.dumps(meta))  # fmt: skip
  print(f"{out}: {palm[-1, 0] * 1e3:+.0f} mm forward, {palm[-1, 1] * 1e3:+.0f} mm left, "
        f"{np.degrees(palm[-1, 2]):+.1f} deg, tilt <= {tilt:.1f} deg, at the cap <= {sat:.2f} s")  # fmt: skip
  return 0


if __name__ == "__main__":
  sys.exit(main())
