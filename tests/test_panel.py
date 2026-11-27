"""The control panel (real/panel/): its clips, its 3D model, a session on the twin."""

import json
import os
import sys
import time

import mujoco
import numpy as np
from mjlab.entity import Entity

from walking_leap import robot

REAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "real")
sys.path.insert(0, REAL)
sys.path.insert(0, os.path.join(REAL, "panel"))
import server  # noqa: E402
import wlc  # noqa: E402


def test_moves_and_emotes_chain_from_the_stand():
  """Every move and emote starts and ends in one pose: the stand, or lying (as a session
  starts), within 0.05 rad, so they chain."""
  pose = {"stand": wlc.Clip("stand_up").q[-1], "lying": wlc.Clip("stand_up").q[0]}
  names = list(server.MOVES.values()) + [e for e, _ in server.EMOTES]
  for name in names:
    c = wlc.Clip(name)
    assert c.start == c.end and c.start in pose, name
    assert c.start == "stand" or name not in server.MOVES.values(), (
      name
    )  # moves: standing
    assert np.abs(c.q[0] - pose[c.start]).max() <= 0.05, name
    assert np.abs(c.q[-1] - pose[c.start]).max() <= 0.05, name


def test_the_dead_crawl_lies_and_stays_safe():
  """The dead crawl never stands: lying from start to end, its plan tilts the palm less
  than 30 deg, no joint faster than 3 rad/s, the thumb as lying."""
  c, lying = wlc.Clip("crawl_dead"), wlc.Clip("stand_up").q[0]
  assert (c.start, c.end) == ("lying", "lying")
  tilt = np.degrees(np.arccos(np.clip(-c.grav[:, 2], -1, 1)))
  assert tilt.max() < 30
  assert np.abs(np.diff(c.q, axis=0)).max() * wlc.HZ <= 3.0
  thumb = [wlc.JOINT_NAMES.index(n) for n in ("th_cmc", "th_axl", "th_mcp", "th_ipl")]
  assert np.abs(c.q[:, thumb] - lying[thumb]).max() < 1e-6


def test_moves_go_where_their_keys_say():
  """W forward, S back, A left (thumb side), D right, Q turns left, E right (planned)."""
  want = {
    "w": (1, 0, 0),
    "s": (-1, 0, 0),
    "a": (0, 1, 0),
    "d": (0, -1, 0),
    "q": (0, 0, 1),
    "e": (0, 0, -1),
  }
  for key, sign in want.items():
    end = wlc.Clip(server.MOVES[key]).palm[-1]
    got = (end[0] * 1e3, end[1] * 1e3, np.degrees(end[2]))  # mm, mm, degrees
    for value, s in zip(got, sign, strict=True):
      if s:
        assert value * s > 5, (key, got)
    if not sign[2]:
      assert abs(got[2]) < 1, (key, got)


def test_catalog_lists_every_key():
  info = server.catalog({})
  assert set(info["moves"]) == set("wasdqe")
  assert {e["name"] for e in info["emotes"]} >= {
    "bow",
    "crawl_dead",
    "crawl_happy",
    "pushups",
  }


def test_3d_model_is_the_repos():
  """hand.json was made from the hand as it is now (make_model.py): same bodies, same frames."""
  with open(os.path.join(REAL, "panel", "static", "hand.json")) as f:
    saved = json.load(f)
  m = Entity(robot.get_leap_hand_cfg(as_built=True, feet=False)).spec.compile()
  assert [b["name"] for b in saved["bodies"]] == [
    m.body(i).name for i in range(1, m.nbody)
  ]
  for i, b in enumerate(saved["bodies"]):
    np.testing.assert_allclose(b["pos"], m.body_pos[i + 1], atol=1e-6)
  d = mujoco.MjData(m)
  check = saved["checks"][0]
  d.qpos[:3], d.qpos[3:7] = check["root_pos"], check["root_quat"]
  for i, n in enumerate(robot.JOINT_NAMES):
    d.qpos[m.joint(n).qposadr[0]] = check["q"][i]
  mujoco.mj_kinematics(m, d)
  np.testing.assert_allclose(d.xpos[1:], check["xpos"], atol=1e-6)


def test_a_twin_session_connects_and_stops():
  s = server.Session("twin", (1200, 0, 600), {})
  assert s.start()[0]
  t0 = time.monotonic()
  while s.state != "on" and time.monotonic() - t0 < 60:
    time.sleep(0.1)
  assert s.state == "on" and s.where == "lying"
  time.sleep(0.5)
  assert s.status()["q"] is not None
  assert s.ask("clip", "step_forward")[0]  # queued; lying, so it gets skipped
  time.sleep(0.3)
  assert s.stop()[0]
  s.thread.join(20)
  assert s.state == "off"
  texts = [e[2] for e in s.events]
  assert any("skipped" in t for t in texts) and texts[-1] == "torque off"


def test_keys_and_the_joystick_follow_the_policy():
  old = server.walk_keys({"forward": 0, "back": 0, "turn_left": 0, "turn_right": 0})
  assert old == {"w": "forward", "s": "back", "q": "turn_left", "e": "turn_right"}
  carrot = server.walk_keys(
    {"forward": 0, "back": 0, "forward_left": 0, "forward_right": 0}
  )
  assert carrot == {
    "w": "forward",
    "s": "back",
    "q": "forward_left",
    "e": "forward_right",
  }
  r = np.array(
    [[-0.02, 0.03], [0.0, 0.0], [-0.3, 0.3]]
  )  # back 20, forward 30 mm/s, 0.3 rad/s
  np.testing.assert_allclose(server.stick_command(0, 1, r), [0.03, 0, 0])
  np.testing.assert_allclose(server.stick_command(0, -0.5, r), [-0.01, 0, 0])
  np.testing.assert_allclose(server.stick_command(-1, 1, r), [0.03, 0, 0.3])  # up-left
  np.testing.assert_allclose(
    server.stick_command(0.5, 0, r), [0, 0, -0.15]
  )  # right: turn right
  np.testing.assert_allclose(
    server.stick_command(9, 9, r), [0.03, 0, -0.3]
  )  # never out of it


def test_a_fake_session_walks_with_a_velocity_policy(tmp_path):
  """A velocity policy on the fake motors, live: a held key walks (the policy's targets
  are sent), released it stands still, the joystick steers inside the policy's range, and
  the walk ends back on the stance."""
  import policy

  from walking_leap.env_cfg import ACTION_SCALE
  from walking_leap.motion_cfg import STUDENT_HISTORY

  n = STUDENT_HISTORY * sum(policy.VELOCITY_SIZES)
  f = tmp_path / "student.npz"
  rng = np.random.default_rng(0)
  np.savez(
    f, w0=rng.normal(0, 0.01, (16, n)), b0=np.zeros(16), mean=np.zeros(n), std=np.ones(n),
    default=wlc.Clip("lie_down").q[0], scale=ACTION_SCALE, lo=-np.ones(16) * 9,
    hi=np.ones(16) * 9, kind="velocity", history=STUDENT_HISTORY,
    command_names=["forward", "forward_left"], command_values=[[0.03, 0, 0], [0.03, 0, 0.3]],
    command_range=[[-0.02, 0.03], [0.0, 0.0], [-0.3, 0.3]],
  )  # fmt: skip
  pol = server.catalog({"student": str(f)})["policies"]["student"]
  assert pol["kind"] == "velocity" and pol["commands"]["forward"] == [30.0, 0.0, 0.0]
  assert pol["keys"] == {"w": "forward", "q": "forward_left"}
  assert pol["range"] == [[-20.0, 30.0], [0.0, 0.0], [-17.2, 17.2]]

  def wait(cond, s=40):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < s:
      time.sleep(0.05)
    assert cond()

  s = server.Session("fake", (1200, 0, 600), {"student": str(f)})
  assert s.start()[0]
  wait(lambda: s.state == "on")
  assert not s.walk("forward", True)[0]  # no velocity policy yet
  assert s.ask("clip", "stand_up")[0]
  wait(lambda: s.status()["where"] == "stand" and s.clip is None)
  assert s.set_policy("student", True)[0]
  assert not s.walk("turn_left", True)[0]  # not one of its commands
  assert s.walk("forward")[0] and not s.walk_pending  # a repeat alone starts nothing
  assert s.walk("forward", True)[0]
  wait(lambda: s.status()["walk_cmd"] == "forward", 5)
  for _ in range(8):
    time.sleep(0.1)
    s.walk("forward")
  assert s.status()["walking"] and s.status()["policy_on"]
  assert not s.set_policy(None, False)[0]  # not while walking
  time.sleep(server.WALK_ALIVE_S + 0.2)  # the page goes quiet: it stands still
  assert s.status()["walk_cmd"] == "stand"
  assert s.walk("forward_left")[0]  # Q on this policy: curve left
  time.sleep(0.1)
  assert s.status()["walk_vec"] == [30.0, 0.0, 17.2]
  assert s.walk_stick(0.5, 1.0)[0]  # the joystick: up and a bit right
  time.sleep(0.1)
  assert s.status()["walk_cmd"] == "joystick"
  assert s.status()["walk_vec"] == [30.0, 0.0, -8.6]
  assert s.walk_stick(0.0, 0.0)[0]  # back in the middle: stand still
  time.sleep(0.1)
  assert s.status()["walk_cmd"] == "stand"
  assert s.end_walk()[0]
  wait(lambda: not s.walking, 10)
  assert s.status()["where"] == "stand"
  assert s.stop()[0]
  s.thread.join(40)
  texts = [e[2] for e in s.events]
  assert "steer: forward" in texts and any(t.startswith("steer: done") for t in texts)
  assert texts[-1] == "torque off"
