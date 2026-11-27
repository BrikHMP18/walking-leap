"""Crawling lying down: the deep crawls, the lying pose the tasks start from, and a lying
velocity policy on the loop and the panel (steer_lying: from lying back to lying)."""

import json
import os
import sys
import time

import numpy as np

REAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "real")
sys.path.insert(0, REAL)
sys.path.insert(0, os.path.join(REAL, "panel"))
import server  # noqa: E402
import wlc  # noqa: E402

DEEP = ("crawl_deep_forward", "crawl_deep_back", "crawl_deep_left", "crawl_deep_right")


def test_the_lying_tasks_start_as_the_loop_lies():
  """The tasks' lying pose is the stand-up clip's first frame (to 1 mrad)."""
  from walking_leap.lying_cfg import LYING_JOINT_POS

  lying = wlc.Clip("stand_up").q[0]
  assert list(LYING_JOINT_POS) == list(wlc.JOINT_NAMES)
  assert np.abs(np.array(list(LYING_JOINT_POS.values())) - lying).max() < 1e-3


def test_the_deep_crawls_lie_and_stay_under_3_rad_s():
  """Each deep crawl starts and ends lying, at the lying pose, every joint under 3 rad/s,
  and the panel lists it as a lying emote."""
  lying = wlc.Clip("stand_up").q[0]
  for name in DEEP:
    c = wlc.Clip(name)
    assert (c.start, c.end) == ("lying", "lying"), name
    assert np.abs(c.q[0] - lying).max() < 1e-6 and np.abs(c.q[-1] - lying).max() < 1e-6
    assert np.abs(np.diff(c.q, axis=0)).max() * wlc.HZ <= 3.0, name
    assert name in dict(server.EMOTES), name


def test_the_trackers_library_is_the_four_crawls():
  from walking_leap import clips

  d = np.load(clips.CLIPS_DIR / "crawl_deep.npz")
  assert len(d["joint_pos"]) == sum(len(wlc.Clip(n).q) for n in DEEP)


def test_a_fake_session_crawls_lying_with_a_lying_policy(tmp_path):
  """A lying velocity policy (start "lying") walks from lying on the fake motors: no
  stand-up, the joystick steers it, and the walk ends lying."""
  import policy

  from walking_leap.env_cfg import ACTION_SCALE
  from walking_leap.motion_cfg import STUDENT_HISTORY

  n = STUDENT_HISTORY * sum(policy.VELOCITY_SIZES)
  f = tmp_path / "lying.npz"
  rng = np.random.default_rng(0)
  np.savez(
    f, w0=rng.normal(0, 0.01, (16, n)), b0=np.zeros(16), mean=np.zeros(n), std=np.ones(n),
    default=wlc.Clip("stand_up").q[0], scale=ACTION_SCALE, lo=-np.ones(16) * 9,
    hi=np.ones(16) * 9, kind="velocity", history=STUDENT_HISTORY, start="lying",
    command_names=["forward", "back"], command_values=[[0.006, 0, 0], [-0.008, 0, 0]],
    command_range=[[-0.008, 0.006], [0.0, 0.0], [-0.1, 0.1]],
    source=json.dumps(dict(run="a_run_s42", checkpoint="model_7.pt")),
  )  # fmt: skip
  assert policy.VelocityPolicy(str(f)).start == "lying"
  pol = server.catalog({"lying": str(f)})["policies"]["lying"]
  assert pol["start"] == "lying"

  def wait(cond, s=40):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < s:
      time.sleep(0.05)
    assert cond()

  s = server.Session("fake", (1200, 0, 600), {"lying": str(f)})
  assert s.start()[0]
  wait(lambda: s.state == "on")
  assert s.set_policy("lying", True)[0]
  assert s.status()["where"] == "lying"
  assert s.walk_stick(0.0, 1.0, True)[0]  # up: forward, from lying
  wait(lambda: s.status()["walking"], 5)
  time.sleep(0.3)
  assert s.status()["walk_vec"][0] > 0
  assert s.end_walk()[0]
  wait(lambda: not s.status()["walking"] and s.status()["where"] == "lying", 10)
  s.stop()
