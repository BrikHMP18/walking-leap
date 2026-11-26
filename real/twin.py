"""The twin: the real loop (wlc.py), unchanged, driving a simulated hand instead of the motors
and the IMU. To try a clip or a policy, or the panel, with no hardware.

  uv run python real/twin.py stand_up crawl lie_down [--policy NAME [--live]] [--kp 1.6 --mu 0.6]

Plain MuJoCo (CPU), the hand as built on stock fingertips, a table. The servos are a PD
with the 350 mA cap (0.18 N*m), 1.6 N*m/rad by default (the real hand's at 1200/0/600, from
current; the joints follow like 2.2-3.0 under load), sliding friction ``--mu`` at the tips,
no torsional friction. Physics moves 20 ms each time the loop sends a target, in two
10 ms halves, one tick late (like the bus). It runs in real time, as the loop paces it.
"""

import argparse
import os
import sys
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from walking_leap import robot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlc  # noqa: E402

MA_PER_NM = 1 / 0.000517  # the datasheet's stall point: 1 mA is 0.517 mN*m


class Sim:
  """The hand on a table. Joint vectors in the repo's order."""

  def __init__(self, kp=1.6, kd=0.11, cap=0.18, mu=0.6, dt=0.002):
    spec = robot.get_spec(as_built=True, feet=False)
    spec.worldbody.add_geom(
      name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1, 1, 0.1)
    )
    spec.option.timestep = dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    for j in spec.joints:
      if j.type != mujoco.mjtJoint.mjJNT_FREE:
        j.armature, j.damping[:] = 0.001, 0.0
    m = spec.compile()
    floor = m.geom("floor").id
    for i in range(m.ngeom):
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
      if i == floor:
        m.geom_contype[i] = m.geom_conaffinity[i] = 1
        continue
      solid = "_collision" in name or name.endswith("_tip") or name.endswith("_foot")
      m.geom_contype[i] = m.geom_conaffinity[i] = int(solid)
      if solid:
        m.geom_priority[i], m.geom_condim[i] = 1, 3
        if name.endswith("_tip"):
          m.geom_friction[i, 0] = mu
        m.geom_solref[i, 0] = max(m.geom_solref[i, 0], 0.004)
    self.m, self.d, self.floor = m, mujoco.MjData(m), floor
    J = robot.JOINT_NAMES
    self.qi = np.array([m.joint(n).qposadr[0] for n in J])
    self.vi = np.array([m.joint(n).dofadr[0] for n in J])
    self.lo = np.array([m.joint(n).range[0] for n in J])
    self.hi = np.array([m.joint(n).range[1] for n in J])
    self.kp = np.full(16, kp)
    self.kp[[J.index(n) for n in ("if_rot", "mf_rot", "rf_rot")]] *= 0.75
    self.kd, self.cap = kd, cap
    self.palm = m.body(robot.ROOT_BODY).id
    self.tau = np.zeros(16)

  def lowest(self):
    """How far the lowest solid part is above the table, m."""
    return min(
      mujoco.mj_geomDistance(self.m, self.d, i, self.floor, 1.0, None)
      for i in range(self.m.ngeom)
      if i != self.floor and self.m.geom_contype[i]
    )

  def step(self, target):
    d = self.d
    tau = self.kp * (target - d.qpos[self.qi]) - self.kd * d.qvel[self.vi]
    self.tau = np.clip(tau, -self.cap, self.cap)
    d.qfrc_applied[self.vi] = self.tau
    mujoco.mj_step(self.m, d)

  def lie(self, q0, settle=1.0):
    """Palm down on the table, the joints at q0, settled."""
    m, d = self.m, self.d
    mujoco.mj_resetData(m, d)
    d.qpos[self.qi] = np.clip(q0, self.lo + 1e-3, self.hi - 1e-3)
    d.qpos[:7] = [0, 0, 0.2, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    d.qpos[2] -= self.lowest() - 0.002
    mujoco.mj_forward(m, d)
    for _ in range(int(settle / m.opt.timestep)):
      self.step(q0)

  def pose(self):
    """The palm on the table: x, y (m) and heading (rad)."""
    R = self.d.xmat[self.palm].reshape(3, 3)
    x, y = self.d.xpos[self.palm][:2]
    return float(x), float(y), float(np.arctan2(R[1, 0], R[0, 0]))


class SimHand:
  """What the loop expects of wlc.Hand, on the Sim."""

  fake = True

  def __init__(self, sim):
    self.s, self.lock = sim, threading.Lock()
    self.lo, self.hi = sim.lo.copy(), sim.hi.copy()
    for n, v in wlc.SAFE_LOW.items():
      i = wlc.JOINT_NAMES.index(n)
      self.lo[i] = max(self.lo[i], v)
    self.prev, self.t = None, time.monotonic()
    self.cmd = sim.d.qpos[sim.qi].copy()
    self.h = SimpleNamespace(
      hold=lambda: None,
      stats=SimpleNamespace(hz=100.0, comm_errors=0, bus_retries=0, ticks=0),
    )

  def __enter__(self):
    return self

  def __exit__(self, *e):
    pass

  def read(self):
    s = self.s
    with self.lock:
      q, dq = s.d.qpos[s.qi].copy(), s.d.qvel[s.vi].copy()
      cur = s.tau * MA_PER_NM
    return (
      SimpleNamespace(t=self.t, hw_error=np.zeros(16, np.uint8)),
      q,
      dq,
      cur,
      self.cmd.copy(),
    )

  def target(self, q):
    q = np.clip(q, self.lo, self.hi)
    with self.lock:
      if self.prev is None:
        self.prev = q
      for tg in (self.prev, q):  # 10 ms each, one tick late
        for _ in range(5):
          self.s.step(tg)
      self.prev = q
    self.cmd, self.t = q, time.monotonic()
    return q

  def hold(self):
    pass

  def temps(self):
    return np.full(16, 30.0)


class SimImu:
  """What the loop expects of the TM171, read off the Sim's palm."""

  quality, error, lost, count = 5, None, 0, 0

  def __init__(self, hand):
    from tm171 import Sample

    self.hand, self.Sample, self.history = hand, Sample, []

  def __enter__(self):
    return self

  def __exit__(self, *e):
    pass

  def wait(self, timeout=0):
    return self.latest

  @property
  def latest(self):
    s = self.hand.s
    with self.hand.lock:
      R = s.d.xmat[s.palm].reshape(3, 3).copy()
      w = s.d.qvel[3:6].copy()  # a free joint's angular velocity is in the body's axes
    g = tuple(R.T @ np.array([0, 0, -1.0]))
    t = time.monotonic()
    return self.Sample(t, t, tuple(w), g, g, (1.0, 0.0, 0.0, 0.0))

  def fresh(self, limit=0.05):
    return True

  def gyro_mean(self, s=0.01):
    return self.latest.gyro


def devices(kp=1.6, mu=0.6, cap=0.18):
  """A simulated hand lying where the stand-up starts, and its IMU."""
  sim = Sim(kp=kp, mu=mu, cap=cap)
  sim.lie(wlc.Clip("stand_up").q[0])
  hand = SimHand(sim)
  return SimImu(hand), hand


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("clips", nargs="+")
  ap.add_argument("--policy")
  ap.add_argument("--live", action="store_true")
  ap.add_argument("--kp", type=float, default=1.6)
  ap.add_argument("--mu", type=float, default=0.6)
  ap.add_argument("--max-off", type=float, default=15.0)
  a = ap.parse_args()
  imu, hand = devices(a.kp, a.mu)
  pol = None
  if a.policy:
    from policy import MotionPolicy

    path = a.policy if a.policy.endswith(".npz") else f"{wlc.POLICIES}/{a.policy}.npz"
    pol = MotionPolicy(path)
  ok, _ = wlc.run(
    a.clips, devices=(imu, hand), policy=pol, live=a.live, max_off=a.max_off,
    label="twin_" + "_".join(a.clips)[:50],
  )  # fmt: skip
  x, y, yaw = hand.s.pose()
  print(
    f"twin: the palm ends at x {x * 1e3:+.0f} y {y * 1e3:+.0f} mm, heading {np.degrees(yaw):+.1f} deg"
  )
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
