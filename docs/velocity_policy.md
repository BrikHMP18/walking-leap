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

- The panel already walks a velocity student ("Walking with a velocity policy" in
  real/README.md): hold a key, release to stand still, shadow then live. What reaches the
  policy depends on its exported file:
  - **Named commands** (`command_names`, `command_values`): one at a time, at its speeds.
    W `forward`, S `back`, A/D `left`/`right`, Q/E `forward_left`/`forward_right` (curving
    while walking) if it has them, else `turn_left`/`turn_right`. Keys stack: the last one
    still held wins (W, add Q: it curves; let go of Q: straight again).
  - **A command range** (`command_range`, 3 x 2: forward m/s, left m/s, turn rad/s; min,
    max): a joystick on the panel (on screen, or a gamepad's left stick), up/down the
    forward speed and left/right the turn rate at the same time, never outside the range.
    `real/export_policy.py <run>/model_<it>.pt` keeps the ranges the run trained on (its
    `params/env.yaml`): stage 3 and later get the joystick by themselves, the students
    (one command at a time, no ranges) keys only. `--command-range` overrides them,
    `--keys-only` drops them.
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

Velocity-v3 (Gait-v1's 0.1 m/s pushes, entropy 0.01) and Velocity-v4 (a diagnosis: no
pushes, cables, random physics or sensor noise, as Walking-Leap-Flat-v3 trained, the one
policy that walked about 20 of 30 mm/s) both continue from Velocity-v2's model_2300.
Comparing them tells whether the perturbations held the walking back.

Velocity-v3 at model_2700 still barely translated (forward 5.2 of 30 mm/s) and the user,
watching it, called it: no progress. All runs stopped. Two followed: Velocity-v5, the
last velocity checkpoint (v3 model_2800) taught by imitation (the command is the clip's
planned velocity, the rewards are the clip's: following the palm's path makes it
translate as the clip does), and Gait-v3, a safe fallback (the smooth clip tracked with
Gait-v1's light pushes, from Gait-v1) for the panel's gait_forward emote.

### The night of 2026-11-27: a curriculum, then a gait clock

`scripts/eval_keys.py` (each key held 8 s after 2 s standing, 8 hands, nominal hand) is
the panel's use; `eval_gait.py` (the clip's command, three cycles) and
`eval_survival.py` as before.

| Policy | Forward, key held (cmd 27.7 mm/s) | Back / left / right (cmd -19.6 / 14.7 / -7.8) | Turns (cmd 18.5 deg/s) | Gets up | 0.5/0.75/1.0 m/s kicks: fell, still down |
| --- | --- | --- | --- | --- | --- |
| Velocity-v5 (clip rewards, no clock), model_1999 | 5.7 | -1.0 / 0.5 / -1.7 | +1.2 / -1.6 | 100 % | 38/62/88 %, 3/25/28 % |
| Student-v2 (of Gait-v3, no clock) | 4.0 | -1.2 / 0.8 / -1.2 | +0.6 / -0.6 | - | - |
| **Student-v3 (of Gait-v3, gait clock)** | **32.2** | -5.7 / 3.5 / -3.6 | +2.6 / -3.3 | **100 %** (0.7 s) | 41/59/78 %, **0/16/28 %** |
| a hand holding its stance | 0.3 | - | - | 0 % | 22/69/72 %, 22/69/72 % |

Gait-v3 (the smooth clip, Gait-v1's light pushes, from Gait-v1): forward 93 of 90 mm in
three cycles, back -31, turns +14/-22 deg (as its clip); kicks fell 16/41/62 %. Gait-v4
(0.25 m/s pushes, from Gait-v3): forward 57, kicks 3/34/62 %. Both are exported as
trackers of the six `real/clips/gait_*` clips (`brik_gait_v3`, `brik_gait_v4`).

Without the clip a student cannot tell which digit's turn it is: the fast teacher alone
(Student-v2) changed nothing. With the gait clock (the clip's phase in training; on the
hand the numpy policy's own clock, 1.04 s cycles while a key is held) the student walks
forward at the commanded speed. The exported numpy policy, its clock running, drives the
simulated hand forward at 33.2 mm/s. Back, sideways and turning stay at about the
clip's own (weak) reach; Velocity-v6 (PPO on the clip's rewards with the clock, from
Student-v3) and Student-v4 (the clock student of Gait-v4 under moderate pushes) run.

Later that night (same tests):

| Policy | Forward | Back / left / right | Turns | Gets up | Kicks: fell, still down |
| --- | --- | --- | --- | --- | --- |
| Student-v4 (clock, of Gait-v4, 0.25 m/s pushes) | 22.0 | -4.1 / 1.6 / -3.1 | +2.2 / -2.4 | 100 % | 19/50/88 %, 3/16/38 % |
| Velocity-v6 (PPO from Student-v3, clip rewards) | 26.1 (drifts -10.5 left) | -3.9 / 3.9 / -2.5 | +3.6 / -4.7 | 100 % | 34/53/78 %, 6/25/47 % |
| Velocity-v7 (v6, planned velocity x3) | 28.6 | -3.3 / 2.0 / -1.9 | +2.4 / -6.6 | 94 % | 25/62/75 %, 3/22/28 % |

PPO after the clock student did not beat it. Turning is limited by the clip: its turn
cycles turn about a fifth of their plan. gait.FAST_TURN_COMMANDS (left 60, right 40 deg
a cycle) turns the replay about twice as far; Gait-v5 and Student-v5 train on it.

**For the demo (2026-11-27):** `brik_keys_student_v3` (Hugging Face, its run's `real/`),
a keyboard velocity policy on the panel: hold W to walk forward about 30 mm/s in
simulation; it gets up by itself in simulation (the real loop stops at 40 deg of tilt).
`brik_gait_v3` / `brik_gait_v4`: trackers of the six fast gait clips on the move keys.
None of them has run on the real hand yet: shadow first, then live.

### Where the night ended (2026-11-27, 06:30)

| Policy (key held 8 s) | Forward (cmd ~27.6) | Turns (cmd +33 / -26 deg/s) | Back / left / right | Gets up | Kicks 0.5/0.75/1.0 m/s, still down |
| --- | --- | --- | --- | --- | --- |
| Gait-v5 (tracker of gait_smooth2, from Gait-v3) | 99 mm in 3 cycles | +38 / -36 deg in 3 cycles | as its clip | - | - |
| Student-v5, model_999 (clock student of Gait-v5) | 27.9 (straight) | +5.9 / -3.2 | -4.8 / 3.0 / -3.9 | 100 % | 0/22/28 % |
| **Student-v5, model_1998 (`brik_keys_v5`)** | **31.2** | +5.5 / -3.1 | -5.7 / 3.4 / -2.5 | **100 %** | **0/12/34 %** |
| Velocity-v8 (PPO from Student-v5, gait_smooth2) | 27.1 | +5.5 / -3.1 | -0.7 / 1.0 / -1.5 | 94 % | 12/28/41 % |

The clock runs at each key's own cycle (gait_smooth2's turns: 1.83 s left, 1.54 s
right; the rest 1.02-1.08 s); with one 1.04 s period the turns were the same. The
students turn well short of their teacher (+38 deg in three cycles is about 9 deg/s); PPO
on top of a student never helped (Velocity-v6, v7, v8).

**The demo policy: `brik_keys_v5`** (Student-v5, model_1998; Hugging Face
`runs/2026-11-27_09-59-38_Walking-Leap-Motion-Gait-Flat-Student-v5_s42/real/brik_keys_v5.npz`,
sha256 a219f38fc9d7...). Its numpy export, its own clock running, drives the simulated
hand forward at 33.3 mm/s and turns it +5.9/-3.4 deg/s; it ran on the real loop with a
fake hand and IMU (stand_up, steer, lie_down). On the hand's PC:

```sh
cd ~/walking-leap/repo && git pull
wget -O outputs/real/policies/brik_keys_v5.npz \
  https://huggingface.co/Autobrik/walking-leap/resolve/main/runs/2026-11-27_09-59-38_Walking-Leap-Motion-Gait-Flat-Student-v5_s42/real/brik_keys_v5.npz
real/panel/run        # Connect, Stand up, Policy: brik_keys_v5, Shadow, then Live; hold W A S D Q E
```

In simulation: `uv run python scripts/steer_student.py outputs/hf/runs/2026-11-27_09-59-38_Walking-Leap-Motion-Gait-Flat-Student-v5_s42 --checkpoint model_1998.pt --fallen`.

Another 1000 distillation iterations (model_2997, run
`2026-11-27_11-12-30_Walking-Leap-Motion-Gait-Flat-Student-v5_s42`): forward 31.2 mm/s,
turns +5.3/-4.1 deg/s, back -6.5, right -4.7; gets up from 100 %; still down after
0.5/0.75/1.0 m/s kicks 0/12/25 %. Its numpy export drives the simulated hand at
32.2 mm/s, turns +5.2/-4.6 deg/s. **`brik_keys_v5b`** replaces `brik_keys_v5` for the
demo (Hugging Face `runs/2026-11-27_11-12-30_Walking-Leap-Motion-Gait-Flat-Student-v5_s42/real/brik_keys_v5b.npz`,
sha256 b68645a466a8...): the same steps, that file instead.
