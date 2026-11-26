# Start here: the branch `mathias/feet-imu-animation`

Everything this branch adds to `main`, how to run it, what to train next, and the
hardware. Brik trains from here on; Mathias keeps the real hand, its numbers and the
parts. The details live in [real_hand.md](real_hand.md) (what ran on the real hand) and
[next_training.md](next_training.md) (what the next policy should change).

## In short

- **The hand as it is built, in the simulator:** TPU feet, the TM171 IMU on its holder,
  LEAP's own thumb limits, the 350 mA torque cap (`as_built=True`).
- **Animations:** written as keyframes (`clips.py`), learned by imitation
  (`Walking-Leap-Motion-Flat-*`). One clip, `hello` (22 s), is learned and holds up to
  pushes, extra weight, softer motors and cable pull, in simulation.
- **The real hand, scripts only:** it stands up by itself, does push-ups, lifts one finger
  at a time and crawls forward (past a 12 cm object in six cycles). The simulator, given
  the same commands, has the palm's tilt within about 1.7° of the real IMU's. No policy
  has driven the real hand yet; the `hello` policy ran in the shadow (computing from the
  real sensors, sending nothing) and agreed with simulation.
- **For the next policy:** train on the hand as it is (stock fingertips, no socks yet),
  imitate the scripted crawl rather than reward walking from scratch, turn on torsional
  friction, center the motors on 2.2-3.0 N·m/rad ([next_training.md](next_training.md)).
- **Before merging into `main`:** a few files that both sides changed (below).

## What the branch adds

| Area | Files | What |
| --- | --- | --- |
| The built hand | `robot.py` (`as_built=True`, `folded_thumb`, `actuator_cfg(effort_limit)`), `assets/parts/` | four TPU socks over the stock fingertips, the IMU (19 g) under its printed cover and pad with its USB plug (58 g in all), LEAP's own thumb limits (Menagerie's are shifted by one joint), the folded-thumb stance, the 350 mA cap (0.181 N·m) |
| Animations | `clips.py`, `assets/clips/hello.npz` | a palm path plus finger gestures, every planted foot solved to stay put; `hello`: push-ups, a stretch, hip circles, a look around, a wave, finger drumming, a bow |
| Imitation | `motion_cfg.py`, `agent_cfg.py`, `mdp.py` | mjlab's motion tracking at this hand's scale; the actor sees only encoders and the IMU (89 numbers). `-v1` adds kicks, sensor noise, up to 20 ms of delay, random grip, balance, motors and a steady cable pull |
| Tests | `tests/test_as_built.py`, `test_motion.py`, `test_real_runs.py` | 53 tests in all on `main` with this, passing on GPU and CPU |
| Real hand | `docs/real_hand.md`, `docs/data/real_clips/`, `docs/data/real_runs/`, `docs/media/real_*.png` | the runs, sim against real, and the exact joint targets the real hand played |
| Next training | `docs/next_training.md` | what to change and why, each number marked measured, sim or guess |
| Hardware | `hardware/` | the latest printable files: IMU mount v4 (cover, pad), foot sock v4 |
| Docs | `README.md`, `PLAN.md`, `AGENTS.md` | the sections above, verified progress, one line on clips |

The walking tasks this branch had for the built hand (`Flat-v1` and `-v2` with gait terms)
were tests and are removed; their results stay in README "First runs". The built hand's
walking comes next from imitating the scripted crawl.

## Tasks

| Task ID (this branch) | What | Train | State |
| --- | --- | --- | --- |
| `Walking-Leap-Flat-v0` | Brik's walking task, nominal hand | as in README | unchanged |
| `Walking-Leap-Motion-Flat-v0` | imitate `hello`, nothing random | `uv run train Walking-Leap-Motion-Flat-v0 --env.scene.num-envs 4096` | learned, within 2.3 mm (one seed) |
| `Walking-Leap-Motion-Flat-v1` | imitate `hello`, randomized and cables | same with `-v1` (3,000 iterations, about 40 min on an RTX 4090) | learned, within 1.8 mm; the one to use |

```sh
uv sync && uv run pytest                    # 53 tests; CUDA_VISIBLE_DEVICES="" for CPU only
uv run play Walking-Leap-Motion-Flat-v1 --viewer viser \
  --motion-file src/walking_leap/assets/clips/hello.npz \
  --checkpoint-file logs/rsl_rl/walking_leap_motion/<run>/model_2999.pt
uv run python -m walking_leap.clips         # rebuild assets/clips/*.npz after editing clips.py
```

Trained checkpoints are not in git (`.gitignore`); Mathias has the `hello` ones
(`model_2999.pt` and its ONNX, 3 MB) and sends them on request.

**A new animation:** write it in `clips.py` like `hello` (palm moves with `to`, `dip`,
`sway`, `hold`; finger gestures with `Gesture`), add it to `CLIPS`, rebuild, look at the
`report` line it prints (balance margin, joint speed, limit margin), then register a task
with `walking_leap_motion_env_cfg(clip="<name>")` next to the `hello` ones in
`__init__.py`, train and play it as above.

## What to train next

The full reasoning and numbers: [next_training.md](next_training.md). In order, each step
needing the one before:

1. **The hand as it is:** an option to build the as-built hand without the socks (stock
   fingertips), the fingertip friction randomized on the tips. Its test.
2. **Torsional friction:** `condim=4` on the hand's contacts, tips 0.001-0.005 m, other
   contacts 0.01-0.03 m, randomized. It is what makes the simulated crawl turn like the
   real one.
3. **Motors:** stiffness 2.2-3.0 N·m/rad at the real gains (1200 / 0 / 600); keep the cap
   at 0.181 N·m ±20 % until the torque per amp is measured.
4. **The crawl as a motion file:** `docs/data/real_clips/crawl.npz` holds the exact joint
   targets the real hand played (stock fingertips); it needs the body poses of the model
   above to become an mjlab motion file (`clips.save` writes that format).
5. **Train the crawl's imitation**, randomized as above, then the randomized test table of
   README "First runs" on the new model.
6. **Then the real hand:** the policy in the shadow first, then driving, with somebody at
   the power switch. Mathias runs that part.

## The real hand

- What ran and what it showed: [real_hand.md](real_hand.md). The joint targets of every
  What the hand recorded in six runs (stand up, a script, lie down): `docs/data/real_runs/*.npz`,
  motors at 100 Hz (`q`, `current` in mA), the IMU at 100 Hz (`gravity`, `gyro`, palm axes), every
  target sent at 50 Hz (`sent`) and which clip played (`clip`); `tests/test_real_runs.py` replays
  one in the simulator and checks the palm's tilt stays within 3° of the real IMU on average
  (1.5° now). Run it after any change to the model's contacts or motors.
  script the hand played: `docs/data/real_clips/*.npz`, each with `q` (frames × 16, 50 Hz,
  the repo's joint order, rad), `grav` (what the simulation's IMU read for them, in the
  palm's axes) and `meta` (JSON: start and end pose, how it was made).
  `stand_up` starts lying palm down; `lie_down` ends there; every other clip starts and
  ends in the folded-thumb stance.
- **Joint order on the hardware:** the LEAP API orders each finger (side, forward, pip,
  dip); the repo (mcp, rot, pip, dip). Motor ID of each repo joint, in order:
  `1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15`. Checked on the hand.
- **The IMU is turned 90° on the hand:** palm x = −IMU y, palm y = +IMU x, palm z = IMU z
  (the model's `IMU_QUAT`). Checked on the hand for x and y.
- The loop that drives the hand is in [`real/`](../real/README.md); the motor driver it
  uses (dexhw) isn't public. Ask Mathias to run a policy on the hand.

## Hardware

What is printed for the hand, the latest versions: [hardware/README.md](../hardware/README.md).

| Part | Version | State | In the simulator |
| --- | --- | --- | --- |
| IMU mount: cover and pad | v4 | v3 printed and fits; v4 adds room for the 5-pin header soldered on the IMU | yes, v4 (`assets/parts/imu_*.stl`) |
| Foot sock, ×4 | v4 | v2 printed, a bit too tight; v3 went too low; v4 is a short cap over the last 16 mm, not printed yet. The real hand has stock fingertips for now | the full v2/v3 sock (`assets/parts/foot_sock_*.obj`): the same end as v4, 1.6 g heavier |

## Merging into `main`

`main` has moved on since this branch started (seven commits, up to the omnidirectional
and IMU-free tasks). A trial merge conflicts in four files only, all of them lists both
sides added to:

- `src/walking_leap/__init__.py`: keep both sets of tasks (`main`'s `Flat-v1` and
  `Flat-NoImu-v1`, this branch's `Motion-Flat-v0` and `-v1`). No task ID is shared any
  more: this branch's test walking tasks are gone.
- `README.md`, `PLAN.md`, `AGENTS.md`: both sides' sections and lines.
- `env_cfg.py` and `mdp.py` merge on their own. `main`'s fingertip-sliding penalty is close
  to the torsion finding above.
- The printed parts live in `hardware/`, the real hand's data in `docs/data/`.

## Licences

Code: MIT, as the repository. The foot socks follow LEAP's fingertip and are CC BY-SA 4.0
(`assets/parts/LICENSE`, `hardware/foot_sock/LICENSE`). The IMU mount only bolts to the
hand: MIT.
