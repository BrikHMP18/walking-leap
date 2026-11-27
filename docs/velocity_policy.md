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

Tasks (all on Gait-v2's hand: the smooth clip, randomization, cables, pushes up to
0.4 m/s per axis, and `mdp.SlewJointPositionActionCfg`, which moves the servo setpoint at
5 rad/s at most as the real driver does):

| Stage | Task | Learns | From |
| --- | --- | --- | --- |
| 1a | `Walking-Leap-Motion-Gait-Flat-v2` | imitate the smooth clip under strong pushes | Gait-v1 |
| 1b | `Walking-Leap-Motion-Gait-Flat-Getup-v0` | get up (every episode starts fallen; IMU and encoders only) | scratch |
| 2 | `Walking-Leap-Motion-Gait-Flat-Student-v1` | copy Getup while fallen, Gait-v2 otherwise (`runner.SwitchTeacher`; teachers merged by `scripts/merge_teachers.py`), 30 % fallen starts | the merged teachers |
| 3 | `Walking-Leap-Motion-Gait-Flat-Velocity-v0` | random keyboard commands (combined, changed every 0.5-3 s, 15 % standing), getting up, pushes; PPO on the student's network (`runner.StudentPpoRunner`) | Student-v1 |

`real/export_policy.py` exports a Student-v1 or a Velocity-v0 checkpoint as a velocity
policy (285 inputs) for the panel and the loop.

## Results

Stage 1 (Modal, seed 42; nominal hand, no noise; 32 hands per survival test):

| Teacher | Run | Check | Measured |
| --- | --- | --- | --- |
| Getup-v0 | `2026-11-27_01-04-30_..._Getup-v0_s42`, 2000 it, W&B `xu0o7ard` | gets up from >= 70 % | **100 %**, median 0.7 s; standing, kicks of 0.5/0.75/1.0 m/s fell 0/3/25 %, still down 0/0/3 % |
| Gait-v2 | `2026-11-27_01-04-12_..._Gait-Flat-v2_s42`, from Gait-v1, 2000 it, W&B `1krnj3it` | falls under kicks below the still hand | 0/34/66 % against 22/69/72 % |
| Gait-v2 | same | forward >= 20 mm/s | **missed**: 45 of 90 mm in three cycles (about 14 mm/s walking; its clip 87 mm); back +2 of -60, left 14 of 45, right +5 of -24, turns +6/-17 of 60 deg |

Gait-v2 traded speed for robustness. The plan went on: stage 3 rewards the commanded
velocity itself, and the student needs the teachers mainly for getting up and the gait's
shape.

Stage 2, Student-v1 (`2026-11-27_02-11-32_..._Student-v1_s42`, 1000 it, W&B `xuv7rvzn`;
behaviour loss 0.099 -> 0.015): gets up from **97 %** of falls (median 0.7 s); kicks of
0.5/0.75/1.0 m/s fell 0/38/62 %, still down at the end 0/31/25 % (Gait-v2: 0/28/66 %).
Commands (three cycles, the clip's planned velocity as its command): the right signs but
small, forward 15 mm (Gait-v2 45), back -6, turns +5/-9 deg, and a drift of 16 mm to the
left even standing. As SuperMimic found, distillation passes on getting up and pushes
better than command tracking; stage 3 is for that.

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

Stage 3, Velocity-v0 (`2026-11-27_02-50-07_..._Velocity-v0_s42`, from Student-v1, 1500
it; `scripts/eval_velocity.py` at model_1200): it kept getting up and taking pushes, and
turned (left +3.1 of 19 deg/s), but barely translated: forward 1.7 of 30 mm/s, back 0,
sideways under 0.4. Its single velocity term paid half for a zero turn while standing,
and its translation scale (0.01 m/s) gave a standing hand no slope towards walking.
Velocity-v1 (the term split, Flat-v3's scales, from v0's end) reached forward 4.3 mm/s
after 500 iterations: better, still standing too much (noise 0.1, learning rate 3e-4).
Velocity-v2 (translation weight 3, noise 0.3, learning rate 1e-3) runs from v1's
model_2000 and, as a check on what the teachers pass on, from scratch.
