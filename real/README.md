# The real hand: control loop and deployment

The loop that has run the real hand: scripted clips and trained animation policies, with
its guards and full logs. Also the IMU's reader and the tool that turns a trained policy
into a file the loop runs. What it has done so far: [`docs/real_hand.md`](../docs/real_hand.md).

```text
real/
  wl                 the command (a launcher for wl.py)
  wl.py              play clips, run a policy in the shadow or live, probe the thumb, list clips
  wlc.py             the loop: Hand (the only class that talks to the motors), Recorder, Clip, Control, run()
  policy.py          an animation policy in plain numpy (89 inputs, or 83 without IMU)
  export_policy.py   a checkpoint (+ its ONNX) -> outputs/real/policies/<run>.npz
  imu/               the TM171's reader (tm171.py, standard library only) and `wlimu`, see imu/README.md
docs/data/real_clips/  the clips the loop plays (joint targets at 50 Hz, what the sim's IMU read)
outputs/real/          policies/, logs/, imu/ (git-ignored)
```

## The bench

- Right LEAP, 16 Dynamixel XC330-M288-T (IDs 0-15) on a U2D2 at 4 Mbps, its own power
  supply. The motors run in current-based position mode with a **350 mA** limit (about
  0.18 N·m): the as-built model's actuator cap.
- The TM171 on the back of the hand (`hardware/imu_mount/`), on USB, 400 readings a second,
  read in the palm's axes (the simulation's: x to the fingertips, y to the thumb, z up).
- A Linux PC next to the hand runs the loop. The motors go through **dexhw**, the LEAP
  driver installed on that PC (Mathias'; not in this repo): it sets the mode, the 350 mA,
  the joint limits and a 0.5 s watchdog at every connect. The loop uses its conda env
  (`WL_PYTHON` to choose another Python). Gains 1200/0/600 by default.
- The policy runs in numpy on that PC's CPU, 50 times a second. No torch, no GPU there.

## Rules, every run

1. **Mathias' go, and somebody at the power switch.** The hand is shared with another
   project: it is set up and left as he says.
2. **One program on the motors' bus.** Nothing else may hold the U2D2's port; close
   `wlimu` before the loop opens the IMU.
3. **It starts lying flat, palm down** (it refuses over 10°), the IMU plugged in for half
   a minute (it refuses under quality 4).
4. **The 350 mA stays,** and so do the gains and the driver's settings.
5. **A policy runs in the shadow first** (computed and logged, the clip is what's sent),
   then live, on a short clip first.

It stops by itself: 40° of tilt; 15° between the IMU's "down" and the one the sim read for
that frame; a motor error; the IMU or the motors silent; a motor at its current limit for
1.5 s (it holds 0.3 s, then torque off); `touch real/STOP` or a motor over 60 °C (it lies
down first). Clips that follow each other must meet within 0.05 rad, checked before
anything moves. Torque is off whenever it returns.

## Scripts

```sh
real/wl list                                # the clips, their length, where they start and end
real/wl play stand_up pushups lie_down      # from lying, back to lying, everything logged
real/wl play stand_up sway taps lie_down
real/wl play stand_up crawl6 lie_down       # six crawl cycles, about 14 cm in the sim
real/wl play ... --fake                     # the driver's simulated motors and a level IMU
```

## Deploying a trained crawl

For the `Walking-Leap-Motion-Crawl-Flat-v0` and `-NoImu-v0` runs on Hugging Face: they
imitate `crawl.npz`, which is the real hand's `crawl` clip replayed on its model, so the
loop's `crawl` plays along frame for frame.

**1. Export it,** on any machine with the repo (`uv sync` done):

```sh
R=2026-11-26_01-55-17_Walking-Leap-Motion-Crawl-Flat-v0_s42
uvx --from huggingface_hub hf download Autobrik/walking-leap --include "runs/$R/*" --local-dir outputs/hf
uv run python real/export_policy.py outputs/hf/runs/$R/model_2999.pt \
  --motion src/walking_leap/assets/clips/crawl.npz --clip crawl --onnx outputs/hf/runs/$R/$R.onnx
```

It writes `outputs/real/policies/$R.npz` (weights, normalizer, stance, action scale,
limits and the reference it trained on). It checks the numpy copy against the ONNX and
refuses to keep it if they differ; it also says how far its reference is from the loop's
clip. For the seed-42 runs, with and without IMU: equal to the ONNX within 1e-6, 1,404
frames both, joints 0.033 rad apart (median; the replay's sag), "down" within 0.8°.

**2. Copy that file** to `outputs/real/policies/` of the repo's clone on the hand's PC.

**3. Shadow, then live** (Mathias there, the rules above):

```sh
real/wl play stand_up crawl lie_down --policy $R            # computed and logged, the clip is sent
real/wl play stand_up crawl lie_down --policy $R --live     # the policy drives the crawl
```

`stand_up` and `lie_down` stay scripted; the policy runs only during the clip it was
trained on (`clip` in its file). At the end of each clip the loop prints the policy's mean
and worst distance from the clip, the peak current, the worst gap between the real "down"
and the sim's, and the motors' temperatures.

**Before the hand,** the same loop code can drive a simulated hand instead of the motors
(the "twin": plain MuJoCo, the hand as built, servo 1.6 N·m/rad, 350 mA, one 10 ms tick
of lag; not in this repo yet). The seed-42 crawls, `stand_up crawl lie_down`:

| | Finished | Policy from the clip, mean (worst) | IMU's "down" from the sim's | Peak current |
| --- | --- | --- | --- | --- |
| with IMU, shadow | yes | 0.021 rad (0.10) | 1.3° | 279 mA |
| with IMU, live | yes | 0.025 rad (0.10) | 2.0° | 285 mA |
| no IMU, live | yes | 0.070 rad (0.48) | 6.9° | 348 mA (the cap) |

So: the with-IMU run first. Through the driver's simulated motors (`--fake`) the same
command runs clean too (0.021 rad in the shadow). None of this has been on the real hand.

## Other policies

- **Another animation** (a `Walking-Leap-Motion-*` task): the loop needs that clip in
  `docs/data/real_clips/` in its own format (`q`, frames × 16 at 50 Hz in the repo's
  order; `grav`, the sim IMU's "down" per frame; `meta`: `joint_names`, `fps`, `start` and
  `end` among `lying`, `stand`), starting and ending on the stand within 0.05 rad. Export
  with `--clip <it>`. A converter from mjlab's motion files to this format isn't written.
- **A walking policy** (`Walking-Leap-Flat-v*`, 57 inputs with a velocity command) doesn't
  run here yet. It needs a policy class with those inputs and a mode that sends its targets
  with no clip; the guard that compares the IMU with the clip's "down" has nothing to
  compare then, so only the tilt guard is left. Train it on the real hand's model
  (`-Flat-v3`): the walking tasks' nominal hand pushes 0.5 N·m, about three times the cap.

## Logs

`outputs/real/logs/<date>_<clips>.npz`, every run, cut or not:

| Array | Rate | Columns |
| --- | --- | --- |
| `motor` | 100 Hz | t, q ×16, dq ×16, current mA ×16, setpoint ×16 |
| `imu` | 400 Hz | t, device time, gyro ×3, acc ×3, gravity ×3, quaternion ×4 |
| `cmd` | 50 Hz | t, clip, frame, sent ×16, clip ×16, sim gravity ×3, policy ×16, loop late ms |
| `meta` | | json: clips and their hashes, gains, policy, events, bus statistics, temperatures |

All joints in the repo's order (`wlc.JOINT_NAMES`); the motors' IDs are
`wlc.MOTOR_OF`. Six of these logs are in `docs/data/real_runs/`, with a test that replays
one in the simulator (`tests/test_real_runs.py`).

## Another driver

`Hand` in `wlc.py` is the only place that knows the driver. Another one (for instance on
LEAP's public `LEAP_Hand_API`) fits if it keeps `Hand`'s methods (`with`, `read`,
`target`, `hold`, `temps`, `lo`, `hi`, `h.stats`) and, before torque, puts every motor in
current-based position mode (5) with a 350 mA limit. LEAP's own example uses 550 mA for
this motor: not here.
