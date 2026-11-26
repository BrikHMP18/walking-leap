"""A trained animation policy -> one .npz that the real loop (real/wl.py) runs in plain numpy.

  uv run python real/export_policy.py CHECKPOINT --motion CLIP_FILE --clip NAME --onnx RUN.onnx

CHECKPOINT  an RSL-RL ``model_<iteration>.pt`` of a Walking-Leap-Motion-* task.
--motion    the clip file the task trained on (its ``--motion-file``, or the task's own in
            src/walking_leap/assets/clips/). The policy keeps that reference.
--clip      the clip of docs/data/real_clips/ it plays along on the hand: same frames, 50 Hz.
--onnx      the run's ONNX, to check the numpy copy against it (do it).
--out       default outputs/real/policies/<checkpoint's folder>.npz

The hand's PC needs no torch: numpy only.
"""

import argparse
import os
import re
import sys

import numpy as np
import torch

from walking_leap import robot
from walking_leap.env_cfg import ACTION_SCALE

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlc  # noqa: E402
from policy import MotionPolicy  # noqa: E402


def gravity_in_palm(quat_wxyz: np.ndarray) -> np.ndarray:
  """World "down" in the palm's axes, per frame. Shape [N, 3]."""
  w, x, y, z = quat_wxyz.T
  # Third row of the rotation matrix (body -> world), negated: R^T (0, 0, -1).
  return -np.stack(
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1
  )


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("checkpoint")
  ap.add_argument("--motion", required=True)
  ap.add_argument("--clip", required=True)
  ap.add_argument("--onnx")
  ap.add_argument("--out")
  a = ap.parse_args()
  name = os.path.basename(os.path.dirname(os.path.abspath(a.checkpoint)))
  out = a.out or f"{wlc.POLICIES}/{name}.npz"
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

  d = torch.load(a.checkpoint, map_location="cpu", weights_only=False)[
    "actor_state_dict"
  ]
  layers = sorted(
    int(m.group(1)) for k in d if (m := re.fullmatch(r"mlp\.(\d+)\.weight", k))
  )
  J = robot.JOINT_NAMES
  assert J == wlc.JOINT_NAMES, "the loop's joint order is the repo's"
  limits = robot.joint_limits(as_built=True)
  stance = [
    next(v for k, v in robot.FOLDED_JOINT_POS.items() if re.fullmatch(k, n)) for n in J
  ]
  m = np.load(a.motion)
  ref_q, ref_dq = m["joint_pos"].astype(float), m["joint_vel"].astype(float)
  ref_g = gravity_in_palm(m["body_quat_w"][:, 0].astype(float))  # body 0: the palm
  clip = wlc.Clip(a.clip)

  def save(eps: float) -> None:
    w = {}
    for i, k in enumerate(layers):
      w[f"w{i}"] = d[f"mlp.{k}.weight"].numpy()
      w[f"b{i}"] = d[f"mlp.{k}.bias"].numpy()
    np.savez(
      out,
      mean=d["obs_normalizer._mean"].numpy()[0],
      std=d["obs_normalizer._std"].numpy()[0] + eps,
      default=stance,
      scale=ACTION_SCALE,
      lo=[limits[n][0] for n in J],
      hi=[limits[n][1] for n in J],
      clip=clip.name,
      ref_q=ref_q,
      ref_dq=ref_dq,
      ref_g=ref_g,
      **w,
    )

  # RSL-RL's normalizer divides by std + eps; which eps, the ONNX says.
  save(1e-2)
  if a.onnx:
    import onnx
    from onnx.reference import ReferenceEvaluator

    R, rng = ReferenceEvaluator(onnx.load(a.onnx)), np.random.default_rng(0)
    for eps in (1e-2, 0.0):
      save(eps)
      P, err = MotionPolicy(out), []
      for _ in range(20):
        o = P.mean + P.std * rng.normal(0, 1.5, P.mean.size)
        feed = {
          "obs": o.astype(np.float32)[None],
          "time_step": np.zeros((1, 1), np.float32),
        }
        act = R.run(["actions"], feed)[0][0]
        err.append(np.abs(act - P.net(o)).max())
      print(f"normalizer eps {eps}: largest difference with the ONNX {max(err):.1e}")
      if max(err) < 1e-4:
        break
    else:
      print("NOT EQUAL to the ONNX: don't use it")
      os.remove(out)
      return 1
  else:
    print("not checked against the ONNX (--onnx): normalizer eps 1e-2 assumed")

  P = MotionPolicy(out)
  n = min(len(ref_q), len(clip.q))
  dq_off = np.abs(ref_q[:n] - clip.q[:n]).max(1)
  g_off = np.degrees(np.arccos(np.clip((ref_g[:n] * clip.grav[:n]).sum(1), -1, 1)))
  print(
    f"{out}: {P.mean.size} inputs ({'with' if P.imu else 'no'} IMU), plays along {clip.name}\n"
    f"  reference {len(ref_q)} frames, the loop's clip {len(clip.q)}\n"
    f"  joints: the reference is {np.median(dq_off):.3f} rad from the loop's targets"
    f" (worst {dq_off.max():.3f}); its down {np.median(g_off):.1f} deg from the clip's"
    f" (worst {g_off.max():.1f})"
  )
  if len(ref_q) != len(clip.q):
    print("  LENGTHS DIFFER: the policy keeps its last frame once its reference ends")
  return 0


if __name__ == "__main__":
  sys.exit(main())
