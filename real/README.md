# The real hand: control loop and deployment

The loop that has run the real hand: scripted clips and trained animation policies, with
its guards and full logs; a control panel on top of it (a web page: keys to move, emotes,
a live 3D view); the IMU's reader; the tool that turns a trained policy into a file the
loop runs; a simulated hand to try all of it without hardware. What the real hand has done
so far: [`docs/real_hand.md`](../docs/real_hand.md).

```text
real/
  setup              once per PC: the loop's own small Python, real/.venv (requirements.txt)
  wl                 the command (a launcher for wl.py)
  wl.py              play clips, run a policy in the shadow or live, steer one, probe the thumb, list clips, scan
  wlc.py             the loop: Hand (the only class that talks to the motors), Recorder, Clip, Control, run()
  policy.py          an animation policy in plain numpy (89 inputs, or 83 without IMU), and a velocity one
  export_policy.py   a checkpoint (+ its ONNX) -> outputs/real/policies/<run>.npz
  panel/             the control panel: server.py, run, static/ (the page, the hand's 3D model)
  twin.py            the loop driving a simulated hand instead of the motors (MuJoCo, CPU)
  make_clip.py       a clips.py motion file -> a clip the loop plays (real/clips/)
  clips/             the panel's moves (one cycle each) and emotes: planned, not yet played on the hand
  leap_hand/         the LEAP hand's driver: the motors' loop, its fake, the rig file, a read-only scan
  imu/               the TM171's reader (tm171.py, standard library only) and `wlimu`, see imu/README.md
docs/data/real_clips/  the clips the real hand played (joint targets at 50 Hz, what the sim's IMU read)
outputs/real/          policies/, logs/, imu/ (git-ignored)
```

## The bench

- Right LEAP, 16 Dynamixel XC330-M288-T (IDs 0-15) on a U2D2 at 4 Mbps, its own power
  supply. The motors run in current-based position mode with a **450 mA** limit (about
  0.23 N·m; 350 mA, about 0.18 N·m, the as-built model's actuator cap, until 2026-11-27).
- The TM171 on the back of the hand (`hardware/imu_mount/`), on USB, 400 readings a second,
  read in the palm's axes (the simulation's: x to the fingertips, y to the thumb, z up).
- A Linux PC next to the hand runs the loop. The motors go through the driver in
  `leap_hand/`: at every connect it sets the mode, the 450 mA, the gains (1200/0/600 by
  default), the joint limits and a 0.5 s watchdog, and it turns torque off on every way
  out. The loop runs in its own small Python, `real/.venv` (`real/setup`; `WL_PYTHON` to
  choose another one).
- The policy runs in numpy on that PC's CPU, 50 times a second. No torch, no GPU there.

## Running it on another PC (a laptop)

Any Linux x86-64 PC with Python 3.10 or newer, no GPU: everything is in this repo. The
hand's PC runs it this way; a laptop is set up the same.

1. The repo as usual (`uv sync`): it's what `export_policy.py` and the tests run in.
2. The loop's Python, once: `real/setup` (real/.venv: numpy, pyyaml, pyserial,
   dynamixel-sdk; with uv if you have it).
3. Without the hand: `real/wl play stand_up crawl lie_down --fake` must end with
   `torque off. OK` (add `--policy <run>` for a policy).
4. The ports, once. Your user in `dialout` (the U2D2 and the IMU), and the U2D2's USB
   latency at 1 ms instead of the kernel's 16 (the driver reads 16 motors 100 times a
   second):

   ```sh
   sudo usermod -aG dialout $USER        # then log out and in
   echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1"'      | sudo tee /etc/udev/rules.d/99-leap-u2d2.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger
   cat /sys/bus/usb-serial/devices/ttyUSB*/latency_timer   # with the U2D2 plugged: 1
   ```

   The driver finds the U2D2 by itself (the one FTDI adapter under `/dev/serial/by-id/`;
   `WL_LEAP_PORT` to name it), and `tm171.py` finds the IMU by its name, so nothing changes
   from one PC to another.
5. The first time on the hand, with Mathias, nothing moving: the U2D2 and the IMU
   unplugged from the other PC and into yours, the hand powered, then `real/wl scan`
   (read only: all 16 answer, torque off) and `real/imu/wlimu check` (400 readings a
   second, none lost; close it after). Then the scripts or a policy as below. At the end
   of every run the loop says how late it ran: on the hand's PC typically under 1 ms; keep
   the laptop on its charger.

## The control panel

A page in the browser that drives the hand through this loop, with every guard below.

```sh
real/panel/run                               # the real hand (on the hand's PC, Mathias there)
real/panel/run --fake                        # the driver's fake motors and a level IMU
uv run python real/panel/server.py --twin    # a simulated hand, anywhere the repo runs
```

Then open http://localhost:8765 (from a laptop: `ssh -L 8765:localhost:8765 <hand PC>`
first, **run on the laptop**; it listens on that computer only). The page loads three.js
from a CDN. The same tunnel started on the hand's PC takes port 8765 from the panel ("port
8765 is taken"; `ss -ltnp | grep 8765` says which program has it).

To keep the panel running after the ssh session closes (closing it otherwise lies the
hand down and turns the torque off), start it in tmux on the hand's PC:

```sh
tmux new -d -s panel -c ~/walking-leap/repo     # a session that outlives ssh
tmux send-keys -t panel 'real/panel/run' Enter
tmux attach -t panel                            # see it; Ctrl+b then d leaves it running
```

Ctrl+C in it stops the panel (lies down, then torque off; it takes a few seconds). After a
`git pull`, restart it: the header shows the commit it runs and turns orange ("restart the
panel") when the repo on disk has moved on.

- **Connect** turns the torque on, lying; **Stand up**, then the moves and emotes, then
  **Lie down** or **Disconnect** (lies down if needed, then torque off).
- **Move, W A S D Q E** (the keyboard or the buttons). Without a policy, or with an
  animation policy: one crawl cycle each, about 10 s. W and S forward and back (23 mm
  planned), A and D sideways (8 mm), Q and E turn on the spot (10°). Up to 4 wait in a
  queue; the hand holds its stance between them. **With a velocity policy: hold to walk**
  (below).
- **Emotes:** bow, push-ups, sway, taps, hello, the happy crawl, three crawl cycles, and
  the fast scripted crawl (`gait_forward`, three cycles of `gait.py`'s smooth planner,
  9.5 s; `scripts/gait_real_clip.py`; not run on the real hand yet): each starts and ends
  on the stand. **The dead crawl is played lying** (offered only while the hand lies, "·
  lying" on its button): it never stands up; index, middle and ring raise, straighten in
  the air, plant at full reach and curl, dragging the palm, four times (13.8 s); the thumb
  stays folded. Made by `make_crawl_dead.py` in the twin's physics. There it barely moves
  forward and wobbles (from -14 to +28 mm, up to 42 mm sideways and 23 deg over 12
  simulated hands; the palm's friction on the table, never measured, decides much of it),
  front heaving up to 29 deg, no motor at its cap for more than 0.9 s.
- **Policy:** the dropdown lists `outputs/real/policies/`. A green dot marks the moves and
  emotes it drives; the rest play as written. Shadow (computed and logged) or Live (its
  targets go to the motors; on the real hand it asks first). Picking one starts it in
  Shadow; the dropdown always shows the policy the server holds (a refused pick goes
  back), and under it the run and checkpoint it was exported from.
- **Live view:** the hand as measured (encoders and the IMU's tilt) and, translucent, as
  the clip says it should be, each standing on the floor. Below it: tilt, the IMU against
  the plan, the policy against the clip, how late the loop runs, motor temperatures.
- **Planned path:** where the hand should be on the table, from the moves' planned
  paths (no sensor measures it), restarting where it stood up; the gyro's heading in
  orange; with the twin, where the simulated hand really is (dashed).
- **STOP** or **Esc:** lies down (if it can), then torque off. The power switch is still
  the emergency stop.
- **When a guard stops the hand** (below, "It stops by itself"), a red banner under the
  header says which one and what it means (e.g. "rf_mcp at its current limit for 1.5 s:
  pushed too hard, or blocked"), until OK or the next Connect.

**What a move does depends on the policy.** In the twin, open loop (no policy), W goes
31 mm where 23 are planned, S 10 of 23, and the turns lean right; all six stay up on 45
random simulated hands. The steering policy (W S Q E) corrected that in the earlier tests;
A and D aren't in any trained policy yet. None of the moves or the new emotes has run on
the real hand.

### The fast gait on the keys

`real/clips/gait_*.npz` (`scripts/gait_real_clip.py`): for each of W S A D Q E, three
cycles of `gait.py`'s smooth crawl, stand to stand, about 9.5 s (forward 86 mm in the
replay, turns 7-22 deg). With a policy that tracks them selected (`brik_gait_v3`:
Walking-Leap-Motion-Gait-Flat-v3, on Hugging Face under its run's `real/`), the keys play
those clips, driven by the policy, instead of the scripted steps. In simulation Gait-v3
walks them as planned (forward 93 of 90 mm in three cycles, about 28 mm/s while walking)
and falls under kicks less than a still hand. None has run on the real hand yet: shadow
first.

### Your own policy on the panel

1. **Export it,** on any machine with the repo (`uv sync` done). One clip: `--motion` and
   `--clip`, as in "Deploying a trained crawl" below. Several clips, one policy (a
   tracker), here the four steering moves:

   ```sh
   uv run python real/export_policy.py <run>/model_<iteration>.pt --onnx <run>/<run>.onnx \
     --tracker --clips step_forward step_back turn_left turn_right --out outputs/real/policies/<name>.npz
   ```

   It refuses to keep the file if the numpy copy differs from the ONNX.
2. **Copy the file** into `outputs/real/policies/` of the repo's clone on the hand's PC:
   `scp outputs/real/policies/<name>.npz <hand PC>:<the clone>/outputs/real/policies/`.
3. **Reload the page:** the panel reads that folder again at every page load (a file it
   can't read is skipped, and its log says why). To try it without the hand first:
   `real/panel/run --fake`, or `--twin` anywhere the repo runs.
4. **Pick it in the dropdown.** The green dots show the moves and emotes it drives (the
   clips it was trained on, by name); the others play as written. Shadow first, then
   Live, with Mathias there.

### Walking with a velocity policy

A velocity student (`kind` "velocity", exported as in "Other policies") is in the dropdown
under "Walking". Standing, **hold** a key: the policy gets that key's command every tick,
at the velocity it trained with for it (the key shows it); keys stack (hold W, add Q: it
curves; let go of Q: straight again). **Release and it stands still**, the policy still
balancing.

| Key | Command (the first one the policy has) |
| --- | --- |
| W, S | `forward`, `back` |
| Q, E | `forward_left`, `forward_right` (curve while walking), else `turn_left`, `turn_right` (on the spot) |
| A, D | `left`, `right` (sideways) |

**The joystick:** a policy trained on *ranges* of commands (any forward speed and turn rate
inside them, at the same time) gets one, on screen (drag it) or a gamepad's left stick
(the browser's Gamepad API; plug it in and press a button once). Up and down set the
forward speed (to its fastest forward and back), left and right the turn rate; sideways
stays 0. The command never leaves the range in the file. The keys become points of it
(W full forward, Q full forward and full left turn, A turning on the spot...). The export
keeps the ranges the run trained on (its `params/env.yaml`, `commands.twist.ranges`), so
a Velocity-v0 or later checkpoint gets the joystick by itself:

```sh
uv run python real/export_policy.py <run>/model_<it>.pt    # -> outputs/real/policies/<task>_s<seed>_it<it>.npz
```

It prints the range it kept. `--command-range FWD_MIN FWD_MAX LEFT_MIN LEFT_MAX TURN_MIN
TURN_MAX` (m/s, m/s, rad/s) sets another one, `--keys-only` none.

Its named commands (`command_names`, `command_values`, from the training's gait) stay as
they are for the keys. The page repeats a held key every
0.15 s; if the panel hears nothing for 0.4 s (key up, the tab hidden, the network gone) it
stands still. Only a key going down starts a walk. **Stop walking**, an emote, Lie down or
Disconnect ends it: a second standing still, then back to the stance, then the next thing.
STOP as always.

- **Shadow:** the hand holds its stance; the policy is computed and logged (the HUD's
  "policy vs clip" is then its distance from the stance). **Live:** it walks.
- Same loop as `real/wl play stand_up steer lie_down` (`Control.steer`): the guards stay
  except the one comparing the IMU with a clip's "down" (there is no clip).
- The map's planned path is the command's velocity, integrated (live only): a plan,
  not a measure.
- No range in its file (the gait students, which have seen one command at a time at fixed
  speeds; or a file exported before the export kept ranges): keys only, at those speeds.

The moves were written with clip functions not in `main` yet (`crawl`,
`backwards`, `turn`, and a sideways crawl:
the turn's cycle with a shift instead of a turn, the leading finger first, 8 mm a cycle;
at 10 mm the step right leaves under 3 mm of balance margin), then `make_clip.py`. The 3D
model is `panel/make_model.py`'s (re-run it if the hand's model changes; a test checks).

## Rules, every run

1. **Mathias' go, and somebody at the power switch.** The hand is shared with another
   project: it is set up and left as he says.
2. **One program on the motors' bus.** Nothing else may hold the U2D2's port; close
   `wlimu` before the loop opens the IMU.
3. **It starts lying flat, palm down** (it refuses over 10°), the IMU plugged in for half
   a minute (it refuses under quality 4).
4. **The 450 mA stays** (`CURRENT_LIMIT_MA` in `wlc.py`, the rig file), and so do the gains
   and the driver's settings.
5. **A policy runs in the shadow first** (computed and logged, the clip is what's sent),
   then live, on a short clip first.

It stops by itself: 70° of tilt; 25° between the IMU's "down" and the one the sim read for
that frame, for 0.3 s; a motor error; the IMU or the motors silent; a motor over 430 mA for
3 s (it holds 0.3 s, then torque off; the guards in `wlc.py`, loosened on 2026-11-27 from
40°, 15° and 330 mA for 1.5 s so a policy can react before the loop gives up); `touch real/STOP` or a motor over 60 °C (it lies
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

It writes `outputs/real/policies/Motion-Crawl-Flat-v0_s42_it2999.npz` (the run's task, seed
and iteration; weights, normalizer, stance, action scale, limits, the reference it trained
on and where it came from). It checks the numpy copy against the ONNX and
refuses to keep it if they differ; it also says how far its reference is from the loop's
clip. For the seed-42 runs, with and without IMU: equal to the ONNX within 1e-6, 1,404
frames both, joints 0.033 rad apart (median; the replay's sag), "down" within 0.8°.

**2. Copy that file** to `outputs/real/policies/` of the repo's clone on the hand's PC.

**3. Shadow, then live** (Mathias there, the rules above):

```sh
P=Motion-Crawl-Flat-v0_s42_it2999
real/wl play stand_up crawl lie_down --policy $P            # computed and logged, the clip is sent
real/wl play stand_up crawl lie_down --policy $P --live     # the policy drives the crawl
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
- **A velocity policy from the keyboard** (the student of
  `Walking-Leap-Motion-Gait-Flat-Student-v0`, 285 inputs: the command, the IMU, the
  encoders and its last action over 5 steps): a `steer` step between `stand_up` and
  `lie_down`.

  ```sh
  uv run python real/export_policy.py logs/rsl_rl/walking_leap_motion/<student run>/model_<it>.pt
  real/wl play stand_up steer lie_down --policy <student run>          # shadow: the stance is held
  real/wl play stand_up steer lie_down --policy <student run> --live   # the policy walks
  ```

  Arrows move (forward, back, left, right), `,` and `.` turn, the same key again or space
  stands still, `q` ends: a second standing still, an easy return to the standing pose,
  then `lie_down`. Each key sends the velocity the student was trained with for that
  command (in the exported file: forward 28 mm/s, back 20, left 15, right 8, turns
  19 deg/s). No clip, so the guard comparing the IMU with the clip's "down" is off; the
  others stay (70° of tilt, current, temperatures, silence, `real/STOP`). The export is
  checked against rsl-rl's student, the steps on a fake hand (`tests/test_real_loop.py`).
- **A walking policy** (`Walking-Leap-Flat-v*`, 57 inputs) doesn't run here: its inputs
  are not the student's. Train it on the real hand's model (`-Flat-v3`): the walking
  tasks' nominal hand pushes 0.5 N·m, about three times the cap.

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

## The driver

`leap_hand/` is the only code that drives the motors, and `Hand` in `wlc.py` the only place
in the loop that uses it. At connect: torque off, current-based position mode (5), the
gains (the side-to-side knuckles at 0.75 of them), Goal Current 350 mA, the goal set to
where the hand is, then torque on, so connecting doesn't move it. Then a 100 Hz loop: the
setpoint moves at most 0.05 rad a tick, a 0.5 s watchdog holds when no target comes, the
motors' error flags are read every 10 ticks, five failed ticks in a row stop it. Torque
goes off on disconnect, at the end of a `with`, at exit and on a signal. Every number is
in `leap_hand/leap_right.yaml`; `tests/test_leap_hand.py` runs the whole loop on the fake
servos. `leap_hand/third_party/` is CMU's LEAP_Hand_API, unmodified (CC BY-NC 4.0). LEAP's
own example uses 550 mA for this motor: not here.
