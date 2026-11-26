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


def test_velocity_policy_inputs_are_the_students():
  from walking_leap.motion_cfg import walking_leap_gait_student_env_cfg

  terms = walking_leap_gait_student_env_cfg().observations["student"].terms
  assert tuple(terms) == policy.VELOCITY_INPUTS


def test_velocity_policy_builds_the_students_observation(tmp_path):
  """Driving the student task with a made-up numpy policy: its 285 inputs (5 steps of
  each term, oldest first, the first step repeated after a reset) are the task's."""
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  from walking_leap.motion_cfg import STUDENT_HISTORY

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg("Walking-Leap-Motion-Gait-Flat-Student-v0", play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    obs, _ = env.reset()
    robot_data = env.scene["robot"].data
    default = robot_data.default_joint_pos[0].cpu().numpy()
    n = STUDENT_HISTORY * sum(policy.VELOCITY_SIZES)
    rng = np.random.default_rng(0)
    f = tmp_path / "student.npz"
    np.savez(
      f, w0=rng.normal(0, 0.02, (16, n)), b0=np.zeros(16), mean=np.zeros(n), std=np.ones(n),
      default=default, scale=ACTION_SCALE, lo=-np.ones(16) * 9, hi=np.ones(16) * 9,
      kind="velocity", history=STUDENT_HISTORY, command_names=["forward"],
      command_values=[[0.03, 0.0, 0.0]],
    )  # fmt: skip
    P = policy.VelocityPolicy(str(f))
    terms = cfg.observations["student"].terms

    def now(name):
      t = terms[name]
      return t.func(env, **t.params)[0].cpu().numpy().astype(float)

    for _ in range(8):
      o = P.obs(
        now("command"),
        now("joint_pos") + default,
        now("joint_vel"),
        now("base_ang_vel"),
        now("projected_gravity"),
      )
      np.testing.assert_allclose(o, obs["student"][0].cpu().numpy(), atol=1e-5)
      a = P.net(o)
      P.last = a
      obs, *_ = env.step(torch.tensor(a, dtype=torch.float32, device=device)[None])
  finally:
    env.close()


def test_keys_name_the_terminals_keys():
  assert wlc.Keys.parse("\x1b[A") == "forward"
  assert wlc.Keys.parse("\x1b[C\x1b[D") == "left"  # the last one counts
  assert wlc.Keys.parse(",x") == "turn_left"
  assert wlc.Keys.parse("q") == "quit"
  assert wlc.Keys.parse("") is None


class _FakeHand:
  """Joints that go where they are sent, at once; everything else quiet."""

  def __init__(self, q):
    from types import SimpleNamespace

    self.q = np.array(q, dtype=float)
    limits = robot.joint_limits(as_built=True)
    self.lo = np.array([limits[n][0] for n in robot.JOINT_NAMES])
    self.hi = np.array([limits[n][1] for n in robot.JOINT_NAMES])
    self.h = SimpleNamespace(
      stats=SimpleNamespace(hz=100.0, comm_errors=0, bus_retries=0, ticks=0)
    )
    self.State = lambda: SimpleNamespace(
      t=__import__("time").monotonic(), hw_error=np.zeros(16, bool)
    )

  def __enter__(self):
    return self

  def __exit__(self, *e):
    pass

  def read(self):
    z = np.zeros(16)
    return self.State(), self.q.copy(), z, z, self.q.copy()

  def target(self, q):
    self.q = np.clip(q, self.lo, self.hi)
    return self.q

  def hold(self):
    pass

  def temps(self):
    return np.full(16, 30.0)


def test_steer_runs_a_velocity_policy_between_stand_up_and_lie_down(
  tmp_path, monkeypatch
):
  """stand_up, steering (forward, then q), lie_down, on a fake hand and IMU: the
  policy's targets are sent while steering, and the hand ends lying."""
  from walking_leap.motion_cfg import STUDENT_HISTORY

  n = STUDENT_HISTORY * sum(policy.VELOCITY_SIZES)
  stance = wlc.Clip("lie_down").q[0]
  f = tmp_path / "student.npz"
  rng = np.random.default_rng(0)
  np.savez(
    f, w0=rng.normal(0, 0.01, (16, n)), b0=np.zeros(16), mean=np.zeros(n), std=np.ones(n),
    default=stance, scale=ACTION_SCALE, lo=-np.ones(16) * 9, hi=np.ones(16) * 9,
    kind="velocity", history=STUDENT_HISTORY, command_names=["forward"],
    command_values=[[0.03, 0.0, 0.0]],
  )  # fmt: skip

  class ScriptedKeys:
    HELP = "scripted"
    script = ["forward"] + [None] * 30 + ["quit"]

    def __enter__(self):
      return self

    def __exit__(self, *e):
      pass

    def read(self):
      return self.script.pop(0) if self.script else None

  monkeypatch.setattr(wlc, "Keys", ScriptedKeys)
  monkeypatch.setattr(wlc, "LOGS", str(tmp_path / "logs"))
  hand = _FakeHand(wlc.Clip("stand_up").q[0])
  ok, path = wlc.run(
    ["stand_up", "steer", "lie_down"],
    speed=8.0,  # the clips fast: this tests the steps, not the motion
    policy=policy.VelocityPolicy(str(f)),
    live=True,
    devices=(wlc.FakeImu(), hand),
  )
  assert ok
  log = np.load(path)
  meta = __import__("json").loads(str(log["meta"]))
  events = [e[1] for e in meta["events"]]
  assert "steer: forward" in events and meta["where"] == "lying"
  cmd = log["cmd"]
  steering = ~np.isnan(cmd[:, 39 + 3])  # policy_q is set only while steering
  assert steering.sum() > 30
  np.testing.assert_allclose(cmd[steering, 3:19], cmd[steering, 38:54], atol=1e-5)
