#!/usr/bin/env python
"""Walking Leap on the real hand: play clips from lying, back to lying, everything logged.

  real/wl play stand_up pushups lie_down                  # the scripts, one after the other
  real/wl play stand_up crawl lie_down --policy NAME      # a policy in the shadow (computed, not sent)
  real/wl play stand_up crawl lie_down --policy NAME --live   # the policy drives
  real/wl play ... --fake                       # the driver's simulated motors and a level IMU: no hardware
  real/wl play stand_up steer lie_down --policy NAME [--live]   # a velocity policy, from the keyboard
  real/wl thumb                                 # lying: where does the thumb stop? (one joint at a time)
  real/wl list                                  # the clips (real/clips/, docs/data/real_clips/)
  real/wl scan                                  # read only: do all 16 motors answer? (nothing written)
  touch real/STOP                               # stop: lies down if it can, then torque off

NAME: a file of outputs/real/policies/ (export_policy.py), or a path to one.
Somebody next to it with the power switch. Nothing else on the motors' bus; close wlimu first.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wlc


def thumb(gains):
  """Each thumb joint alone, slowly, towards its lower limit, while lying. Stops at 300 mA."""
  from leap_hand.hand import raise_on_signals
  from tm171 import TM171

  raise_on_signals()
  with TM171() as imu, wlc.Hand(gains, probe=True) as hand:
    imu.wait()
    rec = wlc.Recorder(
      hand,
      imu,
      "thumb_probe",
      dict(
        gains=gains,
        joint_names=wlc.JOINT_NAMES,
        columns=dict(
          motor=wlc.Recorder.MOTOR, imu=wlc.Recorder.IMU, cmd=wlc.Recorder.CMD
        ),
      ),
    )
    ctl = wlc.Control(hand, imu, rec, max_tilt=12)
    try:
      ctl.start()
      base = ctl.q.copy()
      for name, low in (
        ("th_mcp", -1.15),
        ("th_cmc", -0.33),
        ("th_ipl", -1.0),
        ("th_axl", 1.0),
      ):
        j = wlc.JOINT_NAMES.index(name)
        a = base[j]
        n, stopped = int(abs(low - a) / 0.25 * wlc.HZ), None
        for k in range(1, n + 1):
          q = base.copy()
          q[j] = a + (low - a) * k / n
          s, qq, dq, cur, g = ctl.step(q)
          if abs(cur[j]) > 300 and abs(qq[j] - q[j]) > 0.08:
            stopped = qq[j]
            break
        ctl.hold(0.3)
        s, qq, dq, cur, g = ctl.guard(None)
        rec.event(
          f"{name}: asked {q[j]:+.2f}, reached {qq[j]:+.2f}, {cur[j]:+.0f} mA, tilt {np.degrees(np.arccos(min(1, -g[2]))):.1f} deg"
          + (f" -> STOPPED at {stopped:+.2f}" if stopped is not None else "")
        )
        ctl.ramp(base, 0.4)
        ctl.hold(0.3)
    except (wlc.Abort, wlc.SoftStop, KeyboardInterrupt) as e:
      rec.event(f"ABORT: {e}")
      hand.hold()
    finally:
      rec.close()
  print("torque off.")


def main():
  if sys.argv[1:2] == ["scan"]:
    from leap_hand import scan

    return scan.main(sys.argv[2:])
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("what", choices=["play", "thumb", "list", "scan"])
  ap.add_argument("clips", nargs="*")
  ap.add_argument("--gains", default="1200,0,600")
  ap.add_argument("--speed", type=float, default=1.0)
  ap.add_argument("--policy")
  ap.add_argument("--live", action="store_true")
  ap.add_argument("--fake", action="store_true")
  ap.add_argument("--max-off", type=float, default=15.0)
  ap.add_argument("--label", default="")
  a = ap.parse_args()
  gains = tuple(int(x) for x in a.gains.split(","))
  if a.what == "list":
    for f in sorted(os.listdir(wlc.CLIPS)):
      c = wlc.Clip(f[:-4])
      print(
        f"{c.name:12s} {len(c.q) / 50:5.1f} s  {c.start:6s} -> {c.end:6s}  {c.meta.get('note', '')}"
      )
    return 0
  if a.what == "thumb":
    thumb(gains)
    return 0
  pol = None
  if a.policy:
    from policy import MotionPolicy, VelocityPolicy

    p = a.policy if a.policy.endswith(".npz") else f"{wlc.POLICIES}/{a.policy}.npz"
    with np.load(p) as f:
      velocity = "kind" in f.files and str(f["kind"]) == "velocity"
    pol = VelocityPolicy(p) if velocity else MotionPolicy(p)
    if velocity and "steer" not in a.clips:
      raise SystemExit("a velocity policy runs in a steer step: stand_up steer lie_down")
  ok, path = wlc.run(
    a.clips,
    gains=gains,
    fake=a.fake,
    speed=a.speed,
    policy=pol,
    live=a.live,
    label=a.label,
    max_off=a.max_off,
  )
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
