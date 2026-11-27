# PLAN — Walking LEAP

Roadmap and progress tracking. A box is checked only when the item is
implemented **and** verified (test, measurement or smoke run named next to it).
No time estimates.

Scope of the current milestone: **the foundation only** (phases 1–4). No policy
training beyond PPO smoke tests and no hardware work until phase 5 is approved.
The trainings below that are checked come from a branch, not from this milestone.

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
- [x] README media rendered reproducibly (`scripts/render_media.py`)
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
      limit); results in README "Status". An earlier single run (seed 42):
      README "First runs"
- [ ] Investigate fingertip contact flicker at rest in MuJoCo Warp (~3 % of
      policy steps without contact; absent in CPU MuJoCo); decide whether contact
      observations and metrics need a substep history or a detection margin.
      The standing hand also vibrates in place: over 64 envs, palm speed up to
      6.2 mm/s and yaw rate up to 0.016 rad/s, drift <= 1.9 mm in 5 s
- [x] Decide whether a standing-only command (`vx = 0`) phase is needed first:
      not needed, v0 learned to walk directly (job 27623314)
- [ ] Evaluate alternative support poses (lower stance, different thumb placement)
  - [x] Folded-thumb stance on the as-built hand: holds 4 s (`test_stands_on_its_feet`),
        palm range on planted feet (README "As-built hand")
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
- [ ] Rewards for the as-built hand: the base ones are met by holding a foot up for
      good; three sets of gait terms did not change that (README "First runs"). Those test
      tasks are removed; next: imitate the scripted crawl (docs/next_training.md)
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
- [ ] Real hand, rewards only (`Walking-Leap-Flat-v3`, `-NoImu-v3`): v1 on the real
      hand's model with slower commands and `long_air_time` (no digit up for more
      than 1 s), both penalties bounded
      (`test_long_air_time_counts_only_beyond_the_threshold`, `test_real_hand_tasks`).
      v2 had v1's unbounded sliding penalty and learned to fall within 0.3 s (local
      200-iteration run); its two cluster jobs were cancelled. Train 2 seeds each, 6000 iterations; compare with and
      without IMU: tracking errors, falls, thumb sliding (`tip_slip_rate_th`), and
      survival under pushes as in Bhatt et al., "Blind Dexterity" (arXiv:2608.29487)
- [ ] Imitate the real crawl (`Walking-Leap-Motion-Crawl-Flat-v0`, `-NoImu-v0`):
      docs/next_training.md steps 4–5. The crawl as a motion file
      (`clips.replay`, `test_crawl_file_is_up_to_date`); train 2 seeds each,
      3000 iterations, nominal first, then randomized; then the real hand
      (shadow, then driving)
  - [x] Nominal, 2 seeds each, 3000 iterations: Rorqual jobs 27676904 (IMU) and
        27676905 (no IMU); in the training logs the palm stays 5.8–6.3 mm and
        3.0–3.3° from the clip in all four runs (README "Checkpoints", on Hugging Face)
  - [ ] Evaluated outside training (rollouts watched, tests of docs/next_training.md)
  - [ ] Randomized
  - [ ] Real hand: shadow, then driving
- [ ] Steerable imitation: turning crawl clips (and one where the thumb steps,
      as in `taps`), one policy for all of them, the keyboard picking the next cycle
  - [x] A scripted crawl, one cycle per command (forward, back, left, right, turns,
        stand), cycles chaining in any order (`gait.py`); each command's clip and a
        12-command mix stayed up on 45 of 45 random simulated hands
        (`scripts/gait_clips.py`); planned steps checked by
        `test_gait_commands_move_the_plan`
  - [x] `Walking-Leap-Motion-Gait-Flat-v0`: imitate 150 random cycles of it, with the
        IMU, nothing random (`test_gait_file_is_up_to_date`,
        `test_crawl_as_targets_stays_close[...Gait...]`); 10-iteration smoke run
  - [x] Trained, one seed, 3000 iterations on Modal (W&B `pmh0h1vl`, Hugging Face
        `runs/2026-11-26_20-44-02_Walking-Leap-Motion-Gait-Flat-v0_s42`). Training log,
        iterations 2958-2999: palm 8.5 mm and 4.1° from the clip, episodes 499.4 of
        500 steps, no palm-height or tilt terminations, saturation 0.5 %
  - [x] Evaluate v0 outside training (`scripts/eval_gait.py`, 6 cycles per command, 4
        hands, nominal; outputs/eval/): forward 180 mm of 180 planned, straighter than
        its clip; the rest follow the clip's shortfall, or worse: back -36 mm of -120
        (clip -52), left 32 of 90 (48), right -13 of -48 (-28), turns +10/-32 deg of
        120 (clip +23/-42); no falls. The reference, not the learning, limits it
  - [ ] Keyboard demo: the planner generates the reference live from the keys, the
        policy follows it (checks it obeys commands in orders it never saw)
  - [x] Getting up is possible open loop from a side, the fingertips or the wrist
        (57-115° tilt; best scores 0.96-1.00), not found from the back (164°, 0.51:
        it rolls onto a side) (`scripts/getup_check.py`)
  - [x] `Walking-Leap-Motion-Gait-Flat-v1`: randomization, pushes, cables, falls that
        do not end the episode (`mdp.GaitMotionCommand`: the clip waits, resumes
        where the hand gets up), a fifth of the episodes started fallen, rewards for
        an upright palm and the planned velocity; same observations as v0
        (`test_gait_v1_keeps_v0_observations`,
        `test_gait_v1_clip_waits_while_fallen_and_resumes_where_the_hand_gets_up`);
        10-iteration smoke run, and continuing a v0 checkpoint loads
  - [x] Train v1 from v0's checkpoint (Modal, W&B `82n1zwib`, iterations 3000-5998);
        measured (`scripts/eval_gait.py`, `scripts/eval_survival.py`): commands no better
        than v0 (forward 142 of 180 mm), gets up from 3 % of falls, kicks of 0.5/0.75/1.0
        m/s fell 19/53/59 % (a still hand 22/69/72 %)
  - [x] The planner's targets jumped up to 69 rad/s between frames (a tip's lowest vertex
        switches as it rolls; the real loop refuses 6): `gait.SMOOTH_IK` keeps every
        command under 4.9 rad/s, 45 of 45 random hands up (`scripts/gait_clips.py --smooth`)
  - [x] `real/clips/gait_forward.npz` (`scripts/gait_real_clip.py`): three smooth forward
        cycles, stand to stand, on the panel; 45 of 45 random hands up, at most 13.2 deg
        from the clip's "down" (the loop stops past 15). Not run on the real hand yet
  - [ ] A keyboard-velocity policy that walks fast, takes pushes and gets up
        (docs/velocity_policy.md): teachers Gait-v2 and Getup-v0, a student of both, PPO
        on the student's own task; the panel's velocity mode (W A S D Q E held)
    - [x] The four tasks, the driver's setpoint limit in simulation
          (`mdp.SlewJointPositionActionCfg`), gait_smooth.npz; tests
          `test_keyboard_velocity_tasks_fit_together`,
          `test_slew_action_moves_the_setpoint_at_the_drivers_rate`,
          `test_gait_file_is_up_to_date[gait_smooth]`; 10-iteration smoke runs of each
          stage chained (teachers -> student -> PPO -> export)
    - [x] Stage 1 on Modal (docs/velocity_policy.md, "Results"): Getup-v0 gets up from
          100 % of falls (median 0.7 s); Gait-v2 falls under kicks far less than a still
          hand (0/34/66 % against 22/69/72 %) but walks forward about 14 mm/s (check:
          20; missed)
    - [x] Stage 2, Student-v1: gets up from 97 % of falls, takes kicks as Gait-v2 and
          recovers more (still down 0/31/25 %); commands right in sign but small (forward
          15 mm in three cycles against Gait-v2's 45)
    - [x] Stage 3 (Velocity-v0..v4): rewarded on velocity alone it barely translated
          (forward at most 5.2 of 30 mm/s); Velocity-v5, rewarded on the clip, 5.7 mm/s
          with the key held
    - [x] A curriculum: Gait-v3 (smooth clip, light pushes) tracks at the planned speed
          (forward 93 of 90 mm in three cycles); Gait-v4 adds 0.25 m/s pushes
    - [x] A gait clock for the student (`mdp.gait_clock`; real/policy.py runs its own):
          Student-v3 walks forward 32.2 of 27.7 mm/s with the key held, gets up from
          100 % of falls (`scripts/eval_keys.py`, `eval_survival.py`); its numpy export
          drives the simulated hand at 33.2 mm/s
    - [ ] Turns, backwards and sideways: about the clip's own reach (turns 2-7 deg/s of
          18.5); larger turns in the clip (gait_smooth2, Gait-v5, Student-v5) running
    - [ ] On the real hand: brik_keys_student_v3 on the panel, shadow then live
  - [ ] A velocity-command student: distill v1 (teacher, given the planner's
        reference for a commanded velocity) into a policy that sees only the command,
        encoders and IMU, then fine-tune it with PPO on velocity tracking
    - [x] `Walking-Leap-Motion-Gait-Flat-Student-v0`: rsl-rl's `Distillation` through
          `runner.GaitStudentRunner`; the student sees the plan's velocity, IMU,
          encoders and last action over 5 steps; Huber loss (the teacher's raw actions
          reach 78, far past the joint limits) (`test_gait_student_sees_a_command_not_the_clip`,
          `test_gait_student_runner_takes_a_ppo_teacher_and_plays_the_student`); 40
          local iterations with v0 as teacher: loss 3.4 -> 1.2, episodes 13 -> 571 steps
    - [ ] Distill from the v1 checkpoint; evaluate with `scripts/eval_gait.py`
    - [x] Keyboard: `scripts/steer_student.py` (viewer); the real loop's `steer` step
          (`real/wl play stand_up steer lie_down`), `policy.VelocityPolicy` and its export
          (`test_velocity_policy_builds_the_students_observation`,
          `test_steer_runs_a_velocity_policy_between_stand_up_and_lie_down`)
    - [ ] On the real hand: shadow, then live
    - [ ] PPO fine-tuning of the student on velocity tracking; compare with Flat-v3
          (`scripts/eval_velocity.py`)
  - [ ] The physics replay moves less than planned: right about 2 mm a cycle (8
        planned), back 7 (20), turns 4-7 deg (20); forward drifts left 8 mm a cycle
- [ ] Keyboard teleoperation of the velocity command in `play`

### Phase 8 — Robustness
- [ ] External pushes
- [ ] Fall recovery
- [ ] Randomized initial configurations
- [ ] Ground friction / surface variation
- [ ] Robustness evaluation protocol

### Animations (motion imitation)
- [x] Clips written as keyframes, solved with planted feet (`tests/test_motion.py`)
- [x] Imitation environment whose actor sees encoders and IMU only
      (`test_actor_sees_only_encoders_and_imu`)
- [x] `hello` imitated in simulation, nominal and randomized, with and without the
      cables' pull (`test_cables_pull_and_change`): README "First runs"
- [ ] Measure the cables' pull on the bench, set `CABLE_FORCE` from it
- [ ] The cables' pull in the walking tasks (a dragged cable is not a steady force)
- [ ] Several seeds; several clips with one policy
- [ ] The same clips with a learned world model
- [ ] Clips recorded from a human hand

---

## Milestone C — Real hardware (started: scripted motions, see docs/real_hand.md)

### Phase 9 — Sim-to-real preparation
- [x] The real hand's set-up in the model: IMU and mount, stock fingertips
      (`feet=False`), LEAP's thumb limits, 350 mA cap, measured stiffness 2.6
      N·m/rad and fitted torsional friction (`test_stock_tips_keep_the_imu_and_thumb_limits`,
      `test_torsion_needs_condim_4_and_leaves_sliding_alone`,
      `test_stands_on_stock_tips_folded`)
- [ ] Measure real hand mass and inertia (with IMU, cables, mounts)
  - [x] Feet and IMU in the model, from CAD and datasheets (`tests/test_as_built.py`);
        weights, grip and fit not measured
- [ ] Actuator identification: torque at the configured current limit, gains, armature, friction
  - [x] Stiffness at gains 1200 / 0 / 600 from five scripted runs replayed in simulation:
        2.2-3.0 N·m/rad (docs/real_hand.md, "Simulation against the real hand"); torque per
        amp not measured
  - [x] Contact friction against the same runs: the heading gap is torsional friction the
        model leaves out (`condim=3`); 0.003 m at the tips and 0.03 m elsewhere halve it
        (docs/next_training.md). Not in the model yet
- [ ] Reconcile joint conventions: hardware order (side, forward, pip, dip) vs. MJCF
      (mcp, rot, pip, dip); thumb limits in `LEAPsim_limits` vs. Menagerie
  - [x] Thumb limits: the as-built hand uses LEAP's own
        (`test_thumb_limits_are_leaps_own`); the walking tasks still use Menagerie's
  - [x] Joint order on the hardware: each joint moved alone, and every scripted clip in
        docs/real_hand.md played as in simulation
- [ ] IMU choice and mounting; sensor noise models
  - [x] TM171 on the back of the hand; gravity x and y and gyro x and y checked against the
        simulation on the hand (docs/real_hand.md); yaw axis and noise model not yet
  - [ ] Whether walking needs it: the `Walking-Leap-Flat-NoImu-v1` ablation (phase 7)
- [ ] Communication latency (Dynamixel bus) and observation/action delay
- [ ] Domain randomization (only after nominal physics is validated)
  - [ ] Ranges centered on the real hand's numbers (docs/next_training.md)
- [ ] Policy export (ONNX) and inference timing
  - [x] The animation policy in plain numpy, equal to its ONNX export within 3e-7; runs in
        the 50 Hz loop with the loop under 1 ms late typically

### Phase 10 — Hardware deployment
- [ ] Hardware control loop on the LEAP Hand API
  - [x] A 50 Hz loop on our own driver (not public): scripted clips and the animation
        policy in the shadow, guards and full logs (docs/real_hand.md, "How it was run")
- [ ] Joint mapping and calibration
- [ ] IMU integration
- [ ] Safety limits and emergency stop
  - [x] Stops at 40° of tilt, 15° from the simulation's prediction, a motor error, a silent
        sensor, a motor at its limit for 1.5 s, or on request (lies down first)
- [ ] Supported (suspended) first experiments
- [ ] Incremental free-standing tests
  - [x] Scripted: stand up, push-ups, sway, finger taps (docs/real_hand.md)
- [ ] Real-world locomotion
  - [x] A scripted, quasi-static crawl: 12+ cm in six cycles (one measurement)

## Milestone D — Release

### Phase 11 — Evaluation and open-source release
- [ ] Quantitative evaluation (distance, velocity error, drift, heading error,
      falls, palm contact, saturation, mechanical energy and cost of transport)
- [ ] Baselines and ablations; multiple seeds
- [ ] Released checkpoints
- [ ] Demonstration videos
- [ ] Reproducibility documentation
- [ ] Public release
