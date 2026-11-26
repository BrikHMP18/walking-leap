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
  stand = wlc.Clip("stand_up").q[-1]
  names = list(server.MOVES.values()) + [e for e, _ in server.EMOTES]
  for name in names:
    c = wlc.Clip(name)
    assert (c.start, c.end) == ("stand", "stand"), name
    assert np.abs(c.q[0] - stand).max() <= 0.05, name
    assert np.abs(c.q[-1] - stand).max() <= 0.05, name


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
