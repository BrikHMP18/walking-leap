"""A trained animation policy (the Walking-Leap-Motion-* tasks) in plain numpy, for the real loop.

Its inputs, in the task's order (motion_cfg.py, the actor's terms):

  reference_joints    the clip's 16 joints and their 16 speeds at this frame
  reference_gravity   the clip's "down" in the palm's axes (3)
  base_ang_vel        the IMU's gyro, rad/s, palm axes (3)          - not in the NoImu tasks
  projected_gravity   the IMU's "down", unit vector, palm axes (3)  - not in the NoImu tasks
  joint_pos           the encoders minus the stance (16)
  joint_vel           the encoders' speeds (16)
  actions             its last output (16)

89 numbers with the IMU, 83 without. Output: 16 joint targets, stance + scale * action, inside the
joint limits, 50 times a second. Files made by export_policy.py.
"""

import os

import numpy as np

INPUTS = (
  "reference_joints",
  "reference_gravity",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)
IMU_INPUTS = ("base_ang_vel", "projected_gravity")


def elu(x):
  return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)


class MotionPolicy:
  def __init__(self, path):
    d = np.load(path)
    self.name = os.path.basename(path)[:-4]
    n = sum(1 for k in d.files if k[0] == "w" and k[1:].isdigit())
    self.W = [(d[f"w{i}"].astype(float), d[f"b{i}"].astype(float)) for i in range(n)]
    self.mean, self.std = d["mean"].astype(float), d["std"].astype(float)
    self.default, self.scale = d["default"].astype(float), float(d["scale"])
    self.lo, self.hi = d["lo"].astype(float), d["hi"].astype(float)
    self.imu = self.mean.size == 89
    assert self.mean.size in (83, 89), (
      f"{self.mean.size} inputs: not a Walking-Leap-Motion policy"
    )
    # The clip it was trained on: it only runs while that clip plays.
    self.clip = str(d["clip"]) if "clip" in d.files else "hello"
    # The reference it saw in training (from the task's motion file). Without it, the loop's clip.
    self.ref = (
      (d["ref_q"].astype(float), d["ref_dq"].astype(float), d["ref_g"].astype(float))
      if "ref_q" in d.files
      else None
    )
    self.last = np.zeros(16)

  def reset(self, clip):
    self.last = np.zeros(16)
    if self.ref is not None:
      self.q, self.dq, self.g = self.ref
    else:
      self.q = clip.q
      self.dq = np.gradient(self.q, 1 / 50, axis=0)
      self.g = getattr(clip, "ref_grav", clip.grav)

  def obs(self, i, q, dq, gyro, gravity):
    imu = [gyro, gravity] if self.imu else []
    return np.concatenate(
      [self.q[i], self.dq[i], self.g[i], *imu, q - self.default, dq, self.last]
    )

  def net(self, o):
    x = (o - self.mean) / self.std
    for k, (W, b) in enumerate(self.W):
      x = W @ x + b
      if k < len(self.W) - 1:
        x = elu(x)
    return x

  def act(self, i, q, dq, gyro, gravity):
    i = min(i, len(self.q) - 1)
    o = self.obs(
      i, np.asarray(q), np.asarray(dq), np.asarray(gyro), np.asarray(gravity)
    )
    a = self.net(o)
    self.last = a
    return np.clip(self.default + self.scale * a, self.lo, self.hi)
