# A keyboard-velocity policy that walks fast, takes pushes and gets up

The goal, agreed on 2026-11-26: one policy on the real hand that

- follows the keyboard's velocity: **W/S** forward/back, **A/D** left/right, **Q/E** turn
  left/right; held keys move, releasing stops; combinations (W+Q: an arc) allowed;
- walks fast: at least 20 mm/s forward (the real crawl and the panel's moves do about
  2.5 mm/s);
- takes pushes: falls less often than a hand holding its stance under the same kicks;
- gets up by itself from a fall on a side, its fingertips or its wrist (not from its
  back: `scripts/getup_check.py` found no open-loop way);
- stays inside the real loop's limits: targets no faster than 6 rad/s between frames,
  350 mA.

Its inputs are the student's (`Walking-Leap-Motion-Gait-Flat-Student-v0`, 285 numbers):
the command (forward m/s, left m/s, turn rad/s), the palm IMU, the encoders and the last
action, over the last 5 steps. The real loop already runs such a policy
(`real/wl play stand_up steer lie_down`, `policy.VelocityPolicy`).

## Where we start (measured)

| Policy | Commands (6 cycles, nominal, `scripts/eval_gait.py`) | Gets up (`eval_survival.py`) | Kicks 0.5 / 0.75 / 1.0 m/s, fell | Real hand |
| --- | --- | --- | --- | --- |
| Crawl-v0 IMU s42 (`brik_crawl_s42`) | one clip, forward | no | not measured | ran live on the panel |
| Gait-v0 | forward 180 of 180 mm; back -36 of -120, left 32 of 90, right -13 of -48, turns +10/-32 of 120 deg | - | - | no: targets too fast |
| Gait-v1 | forward 142; back -20, left 35, right -9, turns +17/-28 deg | 3 % | 19 / 53 / 59 % | no: targets too fast |
| A hand holding its stance | - | 0 % | 22 / 69 / 72 % | - |

Training pushes (0.1 m/s) are too weak to matter: a still hand did not fall from 0.3 m/s
(3 %) or from spins of 8 rad/s about a horizontal axis.

**The planner's jumps.** `gait.py`'s inverse kinematics plants each tip at its mesh's
lowest vertex, which switches vertex as the tip rolls: its targets jumped up to 69 rad/s
between frames (gait.npz: 18 % of frames over 6), and the loop refuses such a clip.
`gait.SMOOTH_IK` (each solve also stays near the previous keyframe) brings every command
under 4.9 rad/s, and all seven commands' clips still stay up on 45 of 45 random simulated
hands (`scripts/gait_clips.py --smooth`). Default 0 keeps gait.npz, so Gait-v0/v1 train
what they trained. `real/clips/gait_forward.npz` (`scripts/gait_real_clip.py`): three
smooth forward cycles from the stand back to the stand, 9.5 s, 86 mm in the replay; 45 of
45 random hands stay up, tilt up to 12.5 deg, the palm's "down" at most 13.2 deg from the
clip's (the loop stops past 15). Six cycles tipped 13 of 45 in the fifth cycle: open loop
drifts, and keeping a long walk up is the policy's job.

## The plan: two teachers, one student, then RL

Every stage ends on a check with the scripts above (metrics independent of the reward);
a stage that misses its check stops the plan there.

1. **Teachers, in parallel.**
   - *Gait-v2* (walking): Gait-v1 with the smooth clip (`gait_smooth.npz`), pushes up to
     0.5-1.0 m/s, a penalty on targets faster than about 4 rad/s, no fallen starts;
     continued from Gait-v1. Check: forward at least 20 mm/s; falls under kicks below the
     still hand; its targets pass the loop's clip check.
   - *Getup-v0* (getting up): always starts fallen (57-115 deg about a random horizontal
     axis, as Gait-v1's fallen starts), rewards an upright palm at the stance's height and
     the stance's pose, sees the IMU and the encoders only. Check: gets up from at least
     70 % of the falls. Fallback: imitate `getup_check.py`'s open-loop solutions as clips.
2. **Student**: distilled from both (the get-up teacher while the hand is down, the
   walking teacher while it stands), on the student's inputs. Check: within reach of each
   teacher on the same tests.
3. **RL on the student's own task** (PPO continuing from it): commands drawn over the
   whole range, combined, changed at random times (every 0.5-3 s, mid-cycle included),
   standing still 10-20 % of the time; rewards for the commanded velocity, an upright
   palm, slow targets and little torque; strong pushes, 20-30 % fallen starts. Distillation
   alone tracks commands worse than RL on the velocity reward (SuperMimic, arXiv
   2610.28480, sec. 4.5). Check: tracks the fixed commands of `scripts/eval_velocity.py`
   better than Walking-Leap-Flat-v3, without losing the get-up and kick results.

## On the real hand

- The panel already walks a velocity student (Mathias, "Walking with a velocity policy"
  in real/README.md): hold W S A D Q E, release to stand still, shadow then live. It sends
  one command at a time, at the speed the policy trained with; combinations (W+Q) need a
  policy trained on mixed commands (stage 3) and the panel summing the held keys.
- The repo's LEAP driver already moves each setpoint at most 0.05 rad a tick (100 Hz:
  5 rad/s), whatever the policy asks.
- Walking and pushes run with today's guards. Getting up needs the 40 deg tilt guard
  raised for that test (Mathias' call) and a soft surface.
- Order, as always: shadow, then live, Mathias there.
