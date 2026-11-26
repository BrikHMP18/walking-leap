"""The real hand's loop (real/): what it can check without the hand or its driver."""

import os
import sys

import numpy as np

from walking_leap import robot
from walking_leap.env_cfg import ACTION_SCALE, IMU_OBSERVATIONS
from walking_leap.motion_cfg import walking_leap_motion_env_cfg

sys.path.insert(
  0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "real")
)
import policy  # noqa: E402
import wlc  # noqa: E402


def test_joint_order_is_the_repos():
  assert wlc.JOINT_NAMES == robot.JOINT_NAMES
  assert sorted(wlc.MOTOR_OF) == list(range(16))


def test_real_clips_load_and_meet():
  names = sorted(f[:-4] for f in os.listdir(wlc.CLIPS) if f.endswith(".npz"))
  assert {"stand_up", "lie_down", "crawl", "pushups"} <= set(names)
  clips = {n: wlc.Clip(n) for n in names}
  up, down = clips["stand_up"], clips["lie_down"]
  assert (up.start, up.end, down.start, down.end) == (
    "lying",
    "stand",
    "stand",
    "lying",
  )
  for c in clips.values():
    if c.start == "stand":  # what the loop checks before anything moves
      assert np.abs(up.q[-1] - c.q[0]).max() <= 0.05, c.name
    if c.end == "stand":
      assert np.abs(c.q[-1] - down.q[0]).max() <= 0.05, c.name


def test_policy_inputs_are_the_tasks():
  for imu in (True, False):
    cfg = walking_leap_motion_env_cfg(imu=imu, feet=False)
    terms = tuple(cfg.observations["actor"].terms)
    want = tuple(t for t in policy.INPUTS if imu or t not in policy.IMU_INPUTS)
    assert terms == want, terms
  assert set(policy.IMU_INPUTS) == set(IMU_OBSERVATIONS)


def test_policy_reads_its_inputs_in_order(tmp_path):
  """A made-up linear policy that copies one input to the output: each input lands where the task puts it."""
  rng = np.random.default_rng(0)
  for n in (89, 83):
    w = np.zeros((16, n))
    w[:, :16] = np.eye(16)  # the output is the reference's joints
    f = tmp_path / f"p{n}.npz"
    np.savez(
      f, w0=w, b0=np.zeros(16), mean=np.zeros(n), std=np.ones(n), default=np.zeros(16),
      scale=ACTION_SCALE, lo=-np.ones(16) * 9, hi=np.ones(16) * 9, clip="crawl",
      ref_q=rng.normal(size=(5, 16)), ref_dq=np.zeros((5, 16)), ref_g=np.zeros((5, 3)),
    )  # fmt: skip
    P = policy.MotionPolicy(str(f))
    P.reset(None)
    assert P.imu == (n == 89)
    o = P.obs(2, np.ones(16), np.ones(16), np.ones(3), np.ones(3))
    assert o.size == n
    np.testing.assert_allclose(
      P.act(2, np.ones(16), np.ones(16), np.ones(3), np.ones(3)), ACTION_SCALE * P.q[2]
    )
