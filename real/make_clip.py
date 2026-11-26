"""A clip in mjlab's motion format (``clips.py``'s output) -> a clip the real loop plays.

  python real/make_clip.py MOTION.npz NAME [--title "Bow"] [--note "..."]

Writes real/clips/NAME.npz: the joint targets (``q``, 50 Hz, the repo's joint order), the
palm's planned "down" in its own axes (``grav``: what the loop compares the IMU with), the
planned palm path (``palm``: x, y in m along the start's axes and the heading in rad, for
the panel's map) and ``meta``. The clip must start and end on the stand. numpy only.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlc  # noqa: E402


def rotation(quat_wxyz: np.ndarray) -> np.ndarray:
  """Body -> world rotation matrices, [N, 3, 3]."""
  w, x, y, z = quat_wxyz.T
  return np.stack(
    [
      np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
      np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
      np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ],
    -2,
  )


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("motion")
  ap.add_argument("name")
  ap.add_argument("--title", default="")
  ap.add_argument("--note", default="")
  a = ap.parse_args()
  m = np.load(a.motion)
  q = m["joint_pos"].astype(float)
  R = rotation(m["body_quat_w"][:, 0].astype(float))  # body 0: the palm
  grav = -R[:, 2, :]  # world (0, 0, -1) in the palm's axes: R^T e_z, negated
  heading = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
  pos = m["body_pos_w"][:, 0].astype(float)
  c, s = np.cos(heading[0]), np.sin(heading[0])
  d = pos[:, :2] - pos[0, :2]
  palm = np.stack(
    [c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], heading - heading[0]], 1
  )
  stand = wlc.Clip("stand_up").q[-1]
  for k, at in ((0, "starts"), (-1, "ends")):
    off = float(np.abs(q[k] - stand).max())
    assert off <= 0.05, f"it {at} {off:.3f} rad from the stand (0.05 at most)"
  meta = dict(
    joint_names=list(wlc.JOINT_NAMES),
    fps=wlc.HZ,
    start="stand",
    end="stand",
    title=a.title or a.name.replace("_", " "),
    note=a.note
    or f"from {os.path.basename(a.motion)}: planned, grav from the plan's palm",
  )
  os.makedirs(f"{HERE}/clips", exist_ok=True)
  out = f"{HERE}/clips/{a.name}.npz"
  np.savez_compressed(
    out, q=q.astype(np.float32), grav=grav.astype(np.float32), palm=palm.astype(np.float32),
    meta=json.dumps(meta),
  )  # fmt: skip
  clip = wlc.Clip(out)
  print(
    f"{out}: {len(clip.q) / wlc.HZ:.1f} s, moves {palm[-1, 0] * 1e3:+.0f} mm forward,"
    f" {palm[-1, 1] * 1e3:+.0f} mm left, turns {np.degrees(palm[-1, 2]):+.1f} deg (planned)"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
