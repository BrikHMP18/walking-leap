# PLAN — Walking LEAP

Roadmap and progress tracking. A box is checked only when the item is
implemented **and** verified (test, measurement or smoke run named next to it).
No time estimates.

Scope of the current milestone: **the foundation only** (phases 1–4). No policy
training beyond PPO smoke tests and no hardware work until phase 5 is approved.

---

## Milestone A — Foundation (current)

### Phase 1 — Project infrastructure
- [x] Repository initialization (GitHub: `BrikHMP18/walking-leap`, branch `main`)
- [x] Dependency management
  - [x] `uv` project, Python pinned via `.python-version` (3.12)
  - [x] `mjlab==1.6.1` pinned exactly; full lock in `uv.lock`
  - [x] Dev tools: `pytest`, `ruff`
- [x] Native mjlab integration
  - [x] `mjlab.tasks` entry point in `pyproject.toml`
  - [x] `Walking-Leap-Flat-v0` registered (visible in `uv run list-envs`)
  - [x] mjlab's own `train` / `play` CLIs used unchanged
- [x] Documentation (see phase 4)

### Phase 2 — Robot integration
- [x] MJCF integration
  - [x] Menagerie `leap_hand/right_hand.xml` + meshes copied unmodified (sha256 test)
  - [x] Upstream revision recorded (`robot.MENAGERIE_COMMIT`), MIT license kept
- [x] Floating base: one free joint on `palm`, no welds/equalities/mocap/gravcomp,
      no actuator on the base (`test_floating_base`)
- [x] Reference frames: palm-down = identity, +x forward, +y index side, +z dorsal
      (`test_palm_frame_conventions`, projected gravity checked in `test_reset_to_nominal_pose`)
- [x] Actuators
  - [x] 16 joints, one actuator each, MJCF order (`test_sixteen_actuated_finger_joints`)
  - [x] XC330-M288-T DC-motor position servo from datasheet values
        (`test_actuator_model`: 3 N·m/rad, back-EMF 0.11 N·m·s/rad, 0.5 N·m cap)
  - [x] Joint speed bounded by the 8.48 rad/s no-load speed (`test_joint_speed_within_datasheet`)
  - [x] Sources of every parameter documented (datasheet / LEAP sim / assumed)
- [x] Initial pose
  - [x] Kinematic search: four-tip support, CoM margin 29 mm, joints ≥ 0.25 rad from limits
  - [x] No penetration or self-contact at reset (`test_initial_pose_rests_on_four_digit_tips`)
- [x] Contact configuration: all palm/digit collision primitives vs. ground,
      self-collision between digits (`test_self_collision_between_fingers`)
- [x] Physics validation
  - [x] Gravity on the floating base (`test_gravity_acts_on_floating_base`)
  - [x] Static support with zero action, < 25 % of torque limit (`test_zero_action_static_support`)
  - [x] Individual joint actuation (`test_individual_joint_actuation`)
  - [x] Numerical stability under random targets, MuJoCo warning counters clean
  - [x] Solver settings checked for creep (elliptic cones need converged solves)

### Phase 3 — Reinforcement learning environment (scaffold, no training)
- [x] Observations: actor 57 (proprioception + IMU-type signals), critic 78
      (+ privileged); separation tested (`test_spaces_and_observation_separation`)
- [x] Actions: 16 joint offsets around the nominal pose, clipped to joint limits
- [x] Rewards: velocity/yaw tracking, palm height floor, upright; torque,
      action-rate and palm-contact penalties (signs checked in `test_zero_action_stands`)
- [x] Terminations: timeout (truncation), NaN state, tilt > 70°, palm contact > 1 s
      (`test_time_limit_is_truncation`, `test_independent_reset_and_physical_termination`,
      `test_palm_contact_penalty_and_termination`)
- [x] Commands: constant 0.05 m/s forward in the palm frame
- [x] Metrics independent of reward: forward velocity, net displacement, palm
      contact, actuator saturation, mechanical power, velocity tracking error
- [x] Gait metrics: ground-contact fraction per digit, fingertip slip speed
      (`test_gait_metrics`)
- [x] PPO configuration (RSL-RL via mjlab; actor/critic 256-128-64, mjlab defaults)
- [x] Smoke tests
  - [x] Env init/reset/step, zero and random actions, finite values, GPU and CPU
  - [x] `train` CLI: 10 iterations × 512 envs, finite losses, checkpoints written
  - [x] `play` CLI: checkpoint and `--agent zero` with the Viser viewer
  - [x] 4096 envs fit on an 8 GB GPU (~51k env-steps/s, RTX 5060 Laptop)

### Phase 4 — Documentation of the foundation
- [x] README media rendered reproducibly (`docs/render_media.py`)
  - [x] Nominal pose, three views, palm frame and contact points
  - [x] Zero-action standing GIF
  - [x] Joint-sweep GIF (actuation)
  - [x] Random-action GIF (untrained behavior)
- [x] README: objective, status, stack, architecture, install, commands, tests, limitations
- [x] AGENTS.md: conventions and validation rules for future agents
- [x] PLAN.md (this file)
- [x] Final technical self-review
- [x] Commit and push to the remote

---

## Milestone B — Learning in simulation (not started; needs approval to train)

### Phase 5 — Standing and balance
- [x] Cluster training on Alliance Canada (Rorqual) with `cluster/train.sbatch`:
      the CUDA 13.2 torch build runs on the H100 nodes; 1.16 s per iteration at
      4096 envs (interactive job 27622927); offline W&B runs synced and runs
      uploaded to Hugging Face after job 27623314
- [ ] Run the test suite on a Rorqual GPU node (run in job 27622927, result not
      recorded)
- [x] Baseline run with the current config: job 27623314, `Walking-Leap-Flat-v0`,
      3000 iterations, seeds 42 and 43 complete, seed 44 stopped at 2700 (time
      limit); results in README "Status"
- [ ] Investigate fingertip contact flicker at rest in MuJoCo Warp (~3 % of
      policy steps without contact; absent in CPU MuJoCo); decide whether contact
      observations and metrics need a substep history or a detection margin.
      The standing hand also vibrates in place: over 64 envs, palm speed up to
      6.2 mm/s and yaw rate up to 0.016 rad/s, drift <= 1.9 mm in 5 s
- [x] Decide whether a standing-only command (`vx = 0`) phase is needed first:
      not needed, v0 learned to walk directly (job 27623314)
- [ ] Evaluate alternative support poses (lower stance, different thumb placement)
- [ ] Confirm static support margins against the identified actuator limits
- [ ] Stability evaluation: tilt, palm contact frequency, time without falling

### Phase 6 — Forward locomotion
- [x] Train forward walking at 0.05 m/s; several seeds: seeds 42 and 43 reach
      0.049 m/s from iteration ~300 on, no falls, no palm contact (job 27623314)
- [ ] Analyze the emergent gait (contact patterns per digit, slip, palm contact).
      First look from the training metrics: tripod on index, ring and thumb,
      middle finger on the ground 9–19 % of the time, fingertips sliding at
      1.3–1.9 cm/s while in contact. Still missing: stride frequency, slip per
      stance, rollout videos
- [ ] Tune reward weights, action scale, termination thresholds
- [ ] Physical plausibility audit (saturation, penetration, impacts, reward hacking).
      Saturation 0 % in v0; fingertip sliding is the main open question
- [x] Locomotion metrics reported separately from reward (`Episode_Metrics/*`,
      used to evaluate job 27623314)

### Phase 7 — Directional locomotion
- [ ] Variable forward speed
- [ ] Lateral velocity
- [ ] Turning (yaw rate / heading)
- [ ] Standing commands
- [ ] Omnidirectional tracking
- [ ] `Walking-Leap-Flat-v1`: vx ±0.05, vy ±0.03 m/s, yaw ±0.3 rad/s, resampled
      every 3–5 s, 15 % zero (implemented; `test_omni_commands_cover_all_directions`,
      `test_tracking_error_metrics`); train 3 seeds. A policy has to beat the
      stand-still errors of 0.027 m/s and 0.13 rad/s
- [ ] Stop the fingertips dragging (in v0 the front digits step but the thumb,
      the only rear support, is dragged): v1 adds the `tip_slip` penalty and
      per-tip sliding metrics (`test_gait_metrics`); compare `tip_slip_rate_th`
      and `digit_contact_th` with v0. If the thumb still drags, revisit the
      support pose (phase 5) rather than raising the penalty
- [ ] IMU ablation: `Walking-Leap-Flat-NoImu-v1` (actor without palm IMU, 51
      dims; `test_no_imu_actor_keeps_privileged_critic`) against v1, 3 seeds
      each: tracking errors, falls, slip
- [ ] Keyboard teleoperation of the velocity command in `play`

### Phase 8 — Robustness
- [ ] External pushes
- [ ] Fall recovery
- [ ] Randomized initial configurations
- [ ] Ground friction / surface variation
- [ ] Robustness evaluation protocol

---

## Milestone C — Real hardware (not started)

### Phase 9 — Sim-to-real preparation
- [ ] Measure real hand mass and inertia (with IMU, cables, mounts)
- [ ] Actuator identification: torque at the configured current limit, gains, armature, friction
- [ ] Reconcile joint conventions: hardware order (side, forward, pip, dip) vs. MJCF
      (mcp, rot, pip, dip); thumb limits in `LEAPsim_limits` vs. Menagerie
- [ ] IMU choice and mounting; sensor noise models (only if the NoImu-v1
      ablation shows the IMU is needed)
- [ ] Communication latency (Dynamixel bus) and observation/action delay
- [ ] Domain randomization (only after nominal physics is validated)
- [ ] Policy export (ONNX) and inference timing

### Phase 10 — Hardware deployment
- [ ] Hardware control loop on the LEAP Hand API
- [ ] Joint mapping and calibration
- [ ] IMU integration
- [ ] Safety limits and emergency stop
- [ ] Supported (suspended) first experiments
- [ ] Incremental free-standing tests
- [ ] Real-world locomotion

## Milestone D — Release

### Phase 11 — Evaluation and open-source release
- [ ] Quantitative evaluation (distance, velocity error, drift, heading error,
      falls, palm contact, saturation, mechanical energy and cost of transport)
- [ ] Baselines and ablations; multiple seeds
- [ ] Released checkpoints
- [ ] Demonstration videos
- [ ] Reproducibility documentation
- [ ] Public release
