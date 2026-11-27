"""The deep crawl, lying: the dead crawl (make_crawl_dead.py) digging deeper. The hand never
stands up. Each of index, middle and ring raises, reaches, plants its tip pressed hard into
the table (the knuckle flexed, so the tip carries weight off the palm) and pulls a long
stroke; the thumb stays folded. Four versions, from one stroke: forward (reach far, pull
in), back (plant curled, push out), and turns (one side pulls while the other pushes).
Writes the loop's clips real/clips/crawl_deep_{forward,back,left,right}.npz from the twin's
physics, and with --library the four end to end as a clip library for a tracker
(src/walking_leap/assets/clips/crawl_deep.npz, Walking-Leap-Lying-Track-Flat-v0).

  uv run python real/make_crawl_deep.py            # write the four clips
  uv run python real/make_crawl_deep.py --check    # each on 12 simulated hands
  uv run python real/make_crawl_deep.py --library  # the tracker's library

Physics: twin.Sim (the hand as built on stock fingertips, the servos a PD capped at 350 mA,
targets one 10 ms half-tick late), as make_crawl_dead.py. The palm's sliding friction is
never measured: the clips are planned at the twin's own (the model's 0.2) and checked at
0.3 to 1.0. Open loop the result depends on it a lot (--check).
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
# The strokes, rad for the joints, s for the times, picked on the twin (open loop, 4 hands:
# tips' grip 0.4-0.6, palm 0.2-1.0, servos 1.6-3.0; every joint under 3 rad/s). Forward:
# all three stroke, the index's 1.26 times the ring's; it curves right (the thumb dragging
# on the left, a guess). One cycle per clip (2026-11-27: one press of the panel's key, about
# 7 s; it was four, 18 s). Searched again that day on 24 twin hands (servos 1.6-3.0, tips'
# grip 0.4-0.8, palm 0.3-1.0, caps 350 and 450 mA) once the real loop's tilt stop went to
# 70 deg: the folded thumb pressing into the table (th_mcp +0.6, th_ipl +0.3) takes weight
# off the palm, and the knuckles pressing to 0.9 rad through the pull, the curl to 1.0 and a
# lift before going back to lying: +29 mm a cycle on the median hand (+16 to +57), against
# +5 for the stroke before; tilt <= 45 deg, -11 deg of heading a cycle. A harder thumb or
# more index moved it less. Back: plant curled and push out (pushing moves the palm more
# than pulling). Turns: the tips planted and swept sideways.
FORWARD = dict(
  lift=-0.205, reach_pip=-0.341, plant=(0.45, -0.06, 0.363), pull=(0.9, 1.0, 0.918),
  t_lift=0.885, t_reach=0.607, t_plant=0.945, t_pull=1.488, stagger=0.072, bias=-0.262,
  thumb=(0.6, 0.3), cycles=1, end_lift=True,
)  # fmt: skip
BACK = dict(
  lift=-0.143, reach_pip=-0.147, plant=(0.522, 0.22, -0.229), pull=(0.337, 0.735, 1.173),
  t_lift=0.533, t_reach=0.579, t_plant=0.341, t_pull=1.101, stagger=0.267, bias=0.0,
  thumb=(0.0, 0.0), cycles=1,
)  # fmt: skip
SPIN = dict(
  lift=-0.2, spin=0.3, spin_plant=(0.3, 0.3, 0.3), t_lift=0.5, t_reach=0.3, t_plant=0.4,
  t_pull=1.2, stagger=0.0, cycles=1,
)  # fmt: skip
P = FORWARD
# Each finger's stroke, by version: +1 pulls (forward), -1 pushes (back), 0 lies still.
VERSIONS = {
  "forward": {"if": 1.0, "mf": 1.0, "rf": 1.0},
  "back": {"if": -1.0, "mf": -1.0, "rf": -1.0},
}
SPINS = {"left": -1, "right": 1}  # the sweep's way: +1 turns the palm right
PARAMS = {"forward": FORWARD, "back": BACK, "left": SPIN, "right": SPIN}
TITLES = {
  "forward": "Deep crawl forward",
  "back": "Deep crawl back",
  "left": "Deep crawl turn left",
  "right": "Deep crawl turn right",
}


def ease(u):
  u = np.clip(u, 0.0, 1.0)
  return u**3 * (10 - 15 * u + 6 * u * u)


def keys(p, way, k=1.0):
  """One finger's (mcp, pip, dip) over a cycle: [(seconds, value)]. ``way`` +1 reaches
  far and pulls in, -1 plants curled and pushes out; ``k`` scales the stroke. (A finger
  whose way is 0 stays lying: ``targets``.)"""
  plant = np.array(p["plant"])
  pull = plant + k * (np.array(p["pull"]) - plant)
  a, b = (plant, pull) if way > 0 else (pull, plant)
  # Raise first (the knuckle up, the curl half undone) so the tip leaves the table before
  # it moves: a tip sliding the stroke's way pushes the palm the other.
  raise_ = np.array([p["lift"], 0.5 * (b[1] + a[1]), 0.5 * (b[2] + a[2])])
  up = (
    np.array([p["lift"], p["reach_pip"], 0.0])
    if way > 0
    else np.array([p["lift"], a[1], a[2]])
  )
  return [
    (p["t_lift"], raise_),
    (p["t_reach"], up),
    (p["t_plant"], a),
    (p["t_pull"], b),
  ]


def spin_targets(p, way, cycles=None):
  """Turning on the spot: index, middle and ring lift, swing sideways (the side-to-side
  knuckles), plant pressed in and sweep the other way while planted, turning the palm.
  ``way`` +1 or -1 picks the direction; ``p["spin"]`` is the sweep's half size (rad)."""
  lying = wlc.Clip("stand_up").q[0].copy()
  cycles = p["cycles"] if cycles is None else cycles
  a, plant = p["spin"], np.array(p["spin_plant"])
  ts = [p["t_lift"], p["t_reach"], p["t_plant"], p["t_pull"]]
  start, ramp, tail = int(0.8 * HZ), int(0.8 * HZ), int(1.6 * HZ)
  lead = int(round(p["stagger"] * HZ))
  cycle = sum(int(round(t * HZ)) for t in ts)
  n = start + cycles * cycle + 2 * lead + tail
  q = np.repeat(lying[None], n, 0)
  for j, f in enumerate(FINGERS):
    cols = [I[f + "_mcp"], I[f + "_rot"], I[f + "_pip"], I[f + "_dip"]]
    rest = lying[cols]
    keys_ = [
      np.array([p["lift"], way * a, 0.5 * plant[1], 0.5 * plant[2]]),
      np.array([p["lift"], way * a, plant[1], plant[2]]),
      np.array([plant[0], way * a, plant[1], plant[2]]),
      np.array([plant[0], -way * a, plant[1], plant[2]]),
    ]
    t, pts = start + j * lead, [(start + j * lead, rest)]
    for _ in range(cycles):
      for dur, val in zip(ts, keys_, strict=True):
        t += int(round(dur * HZ))
        pts.append((t, val))
    pts.append((t + ramp, rest))
    for (ta, va), (tb, vb) in zip(pts[:-1], pts[1:], strict=True):
      for k in range(ta, min(tb, n)):
        q[k, cols] = va + (vb - va) * ease((k - ta) / max(1, tb - ta))
  return q


def targets(version="forward", p=None):
  """The clip's targets at 50 Hz: still, the cycles, back to lying, still. The side-to-side
  knuckles go parallel while crawling."""
  p = PARAMS.get(version, P) if p is None else p
  if version in SPINS:
    return spin_targets(p, SPINS[version])
  ways = VERSIONS[version]
  lying = wlc.Clip("stand_up").q[0].copy()
  cycle = sum(int(round(t * HZ)) for t, _ in keys(p, 1))
  lead, start, ramp, tail = (
    int(round(p["stagger"] * HZ)),
    int(0.8 * HZ),
    int(0.8 * HZ),
    int(1.6 * HZ),
  )
  # end_lift: after the last stroke each finger rises (as at a cycle's start) before it
  # goes back to lying, so a tip planted at the stroke's end leaves the table first instead
  # of pushing the palm back as it uncurls.
  lift_n = int(round(p["t_lift"] * HZ)) if p.get("end_lift") else 0
  n = start + p["cycles"] * cycle + 2 * lead + tail + lift_n
  q = np.repeat(lying[None], n, 0)
  for j, f in enumerate(FINGERS):
    cols = [I[f + "_mcp"], I[f + "_pip"], I[f + "_dip"]]
    rest = lying[cols]
    t, pts = start + j * lead, [(start + j * lead, rest)]
    if ways[f] == 0:
      continue
    for _ in range(p["cycles"]):
      k = {"if": 1 - p.get("bias", 0.0), "mf": 1.0, "rf": 1 + p.get("bias", 0.0)}[f]
      if version in ("left", "right"):
        k = 1.0
      for dur, val in keys(p, ways[f], k):
        t += int(round(dur * HZ))
        pts.append((t, val))
    if lift_n:
      t += lift_n
      pts.append((t, keys(p, ways[f], k)[0][1]))
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
  for name, dv in zip(("th_mcp", "th_ipl"), p.get("thumb", (0.0, 0.0)), strict=True):
    q[:, I[name]] += dv * w
  return q


def simulate(q, kp=1.6, mu=0.6, palm_mu=None):
  """The twin's physics: per frame the palm's x, y, heading, "down" in its axes and height,
  the tilt (deg), the longest any motor stayed at its cap (s), and the palm's share of
  the hand's weight while crawling (its contacts' normal force over m g)."""
  s = twin.Sim(kp=kp, mu=mu)
  if palm_mu is not None:
    for i in range(s.m.ngeom):
      if s.m.geom_contype[i] and i != s.floor and not s.m.geom(i).name.endswith("_tip"):
        s.m.geom_friction[i, 0] = palm_mu
  s.lie(q[0])
  import mujoco

  weight = float(s.m.body_subtreemass[s.palm]) * 9.81
  palm_geoms = {i for i in range(s.m.ngeom) if s.m.geom_bodyid[i] == s.palm}
  half = int(round(0.01 / s.m.opt.timestep))
  prev, rows, sat, worst, loads = q[0], [], np.zeros(16), 0.0, []
  f6 = np.zeros(6)
  for k in range(len(q)):
    for target in (prev, q[k]):
      for _ in range(half):
        s.step(target)
    prev = q[k]
    R = s.d.xmat[s.palm].reshape(3, 3)
    rows.append((*s.pose(), *(-R[2, :]), float(s.d.xpos[s.palm][2])))
    sat = np.where(np.abs(s.tau) > 0.94 * s.cap, sat + 1 / HZ, 0.0)
    worst = max(worst, float(sat.max()))
    load = 0.0
    for c in range(s.d.ncon):
      con = s.d.contact[c]
      if con.geom1 in palm_geoms or con.geom2 in palm_geoms:
        mujoco.mj_contactForce(s.m, s.d, c, f6)
        load += abs(f6[0])
    loads.append(load / weight)
  a = np.array(rows)
  tilt = float(np.degrees(np.arccos(np.clip(-a[:, 5], -1, 1))).max())
  return a, tilt, worst, np.array(loads)


def moved(a):
  """Forward and left (m, the start's axes) and the turn (rad), start to end."""
  c, s = np.cos(a[0, 2]), np.sin(a[0, 2])
  d = a[-1, :2] - a[0, :2]
  return c * d[0] + s * d[1], -s * d[0] + c * d[1], a[-1, 2] - a[0, 2]


def write(version, q):
  p, tilt, sat, _ = simulate(q)
  c, s = np.cos(p[0, 2]), np.sin(p[0, 2])
  d = p[:, :2] - p[0, :2]
  palm = np.stack(
    [c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], p[:, 2] - p[0, 2]], 1
  )
  grav = p[:, 3:6] / np.linalg.norm(p[:, 3:6], axis=1, keepdims=True)
  meta = dict(
    joint_names=list(J), fps=HZ, start="lying", end="lying", title=TITLES[version],
    note=(f"lying, never standing: the fingers plant pressed in and stroke ({version}); "
          f"{PARAMS[version]['cycles']} cycles; planned on the twin (palm friction 1.0, a guess)"),
  )  # fmt: skip
  out = f"{HERE}/clips/crawl_deep_{version}.npz"
  np.savez_compressed(out, q=q.astype(np.float32), grav=grav.astype(np.float32),
                      palm=palm.astype(np.float32), meta=json.dumps(meta))  # fmt: skip
  print(f"{out}: {palm[-1, 0] * 1e3:+.0f} mm forward, {palm[-1, 1] * 1e3:+.0f} mm left, "
        f"{np.degrees(palm[-1, 2]):+.1f} deg, tilt <= {tilt:.1f} deg, at the cap <= {sat:.2f} s")  # fmt: skip


def library() -> None:
  """The four clips end to end as the twin plays them (its own grip, 1.6 N*m/rad; the
  training's stiffer model lifts the palm to 95 mm with the same targets), in mjlab's
  motion format, with the training's model for the bodies' poses."""
  from walking_leap import clips

  q = np.concatenate([wlc.Clip(f"crawl_deep_{v}").q for v in PARAMS])
  s = twin.Sim()
  s.lie(q[0])
  half = int(round(0.01 / s.m.opt.timestep))
  qpos, prev = np.zeros((len(q), s.m.nq)), q[0]
  for i, target in enumerate(q):
    qpos[i] = s.d.qpos
    for tg in (prev, target):
      for _ in range(half):
        s.step(tg)
    prev = target
  hand = clips.RealHand()
  assert hand.model.nq == s.m.nq
  motion = clips.Motion(
    "crawl_deep", qpos, np.zeros((len(q), 4), bool), np.zeros(len(q))
  )
  out = clips.CLIPS_DIR / "crawl_deep.npz"
  clips.save(motion, hand, out)
  z = qpos[:, 2]
  print(f"{out}: {len(q) / HZ:.1f} s, the palm {z.min() * 1e3:.0f}-{z.max() * 1e3:.0f} mm high, "
        f"from ({qpos[0, 0] * 1e3:+.0f}, {qpos[0, 1] * 1e3:+.0f}) to ({qpos[-1, 0] * 1e3:+.0f}, "
        f"{qpos[-1, 1] * 1e3:+.0f}) mm")  # fmt: skip


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument(
    "--check", action="store_true", help="12 simulated hands, nothing written"
  )
  ap.add_argument("--library", action="store_true", help="the tracker's clip library")
  a = ap.parse_args()
  if a.library:
    library()
    return 0
  for version in PARAMS:
    q = targets(version)
    speed = float(np.abs(np.diff(q, axis=0)).max() * HZ)
    print(f"{version}: {len(q) / HZ:.1f} s, fastest joint {speed:.2f} rad/s")
    if not a.check:
      write(version, q)
      continue
    for kp, mu, pm in itertools.product((1.6, 3.0), (0.4, 0.8), (0.3, 0.5, 1.0)):
      p, tilt, sat, load = simulate(q, kp, mu, pm)
      fx, fy, yaw = moved(p)
      print(f"  kp {kp} tip mu {mu} palm mu {pm}: {fx * 1e3:+.0f} mm x, {fy * 1e3:+.0f} mm y, "
            f"{np.degrees(yaw):+.0f} deg, tilt <= {tilt:.0f}, at the cap <= {sat:.2f} s, "
            f"palm load {load.mean():.2f}")  # fmt: skip
  return 0


if __name__ == "__main__":
  sys.exit(main())
