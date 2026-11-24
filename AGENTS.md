# AGENTS.md — instructions for coding agents and contributors

## Objective

Emergent finger locomotion for the LEAP Hand V1 (16 DOF, right hand) trained
with RL in mjlab, progressing toward sim-to-real deployment. Priorities, in
order: **physical validity, experimental correctness, reproducibility,
simplicity.** A smaller, verifiable change always beats a broader speculative one.

The current milestone is the foundation (see `PLAN.md`). Do not launch training
runs beyond smoke tests (a few iterations) or start hardware work without the
maintainer's explicit approval.

## Architecture conventions

- This is an **external mjlab task package**, not a framework. Use mjlab's
  managers, terms, runner, `train`/`play` CLIs, viewers and logging as they are.
  No custom training loops, simulation loops, CLI wrappers or base classes
  unless a concrete technical need is demonstrated (and written down).
- One responsibility per file:
  - `robot.py` — MJCF loading and every modification of it, actuators,
    collisions, nominal pose. The asset XML itself is never edited.
  - `env_cfg.py` — the environment as configuration.
  - `agent_cfg.py` — PPO / networks.
  - `mdp.py` — only MDP terms mjlab does not have. Reuse
    `mjlab.envs.mdp` and `mjlab.tasks.velocity.mdp` first. Split `mdp.py`
    into modules only when its size justifies it.
- Configuration-driven: thresholds, weights and scales are config values, not
  constants buried in functions.
- Register new variants with `register_mjlab_task` in `__init__.py`; task IDs
  follow `Walking-Leap-<Terrain>-v<N>`.
- Style: ruff, 2-space indentation (mjlab's convention), type hints, short
  docstrings that state units and frames.

## Physics rules

- `assets/leap_hand/right_hand.xml` is Menagerie's file, byte-identical
  (`test_asset_matches_upstream`). Apply changes in `robot.get_spec()` and
  document them there. When updating the asset, update `MENAGERIE_COMMIT` and
  the hash together.
- Do not strengthen actuators, raise friction, add hidden supports or forces,
  or change physics options to make a behavior appear. Every actuator value is
  labeled as datasheet, LEAP-simulator or assumed; keep that labeling.
- Frames: palm frame +x forward (wrist → fingertips), +y thumb side, +z dorsal;
  palm-down is the identity orientation. Commands are in the palm frame.
- Keep actor observations deployable: anything not measurable on the hand
  (encoders) or by a palm IMU belongs to the critic, and the README table must
  say which sensor provides each actor term.
- No domain randomization, observation noise or curricula until the nominal
  behavior they would perturb has been validated.

## Known pitfalls (mjlab 1.6.1, MuJoCo / MuJoCo Warp 3.11)

- mjlab ignores entity `<option>`s; set physics options in `MujocoCfg`.
- `impratio=100` produced NaN contact-sensor forces in MuJoCo Warp; keep 1.
- Elliptic cones need converged solves: 10 / 20 iterations made the standing
  hand creep. Keep the defaults (100 / 50).
- Size `nconmax` / `njmax` explicitly; automatic sizing ran out of memory at
  4096 envs on an 8 GB GPU. `qpos0` must not start inside the ground.
- mjlab's velocity-command joystick (Viser) cannot represent ranges below 0.1;
  `mdp.VelocityCommand` disables it.
- CPU MuJoCo silently resets the state after a divergence; physics tests must
  check `MjData.warning` counters, not only finiteness.
- Many mjlab APIs changed since older examples (e.g. `RslRlModelCfg`,
  asset loading). Read the installed version's source before using an API.

## Dependencies

- Managed with `uv`; `uv.lock` is committed. Never `pip install` into `.venv`.
- `mjlab` is pinned exactly. To upgrade: read mjlab's changelog, bump the pin,
  `uv lock`, run the full test suite and a training smoke test, and update the
  README version table.
- Add a dependency only if mjlab does not already provide the capability.

## Testing and validation

Before every commit:

```sh
uv run ruff check src tests docs && uv run ruff format --check src tests docs
uv run pytest
uv run train Walking-Leap-Flat-v0 --env.scene.num-envs 512 --agent.max-iterations 10  # if env/agent changed
```

- New physics or model changes need a test in `tests/test_robot.py` (plain
  MuJoCo, CPU). New MDP terms need coverage in `tests/test_env.py`.
- Tests must pass on CPU (`CUDA_VISIBLE_DEVICES=""`), not only on a GPU.
- If a change alters what the hand looks like or does at rest, regenerate the
  README media with `uv run python docs/render_media.py`.

## Experimental validation

- Report locomotion with metrics that are independent of the reward
  (`Episode_Metrics/*`, velocity-tracking errors, displacement, palm contact,
  saturation, mechanical power), across several seeds.
- Look actively for reward hacking: palm dragging, penetration exploits,
  bouncing, oscillation without displacement, actuator saturation, reset
  exploitation. Watch rollouts, not just curves.
- Never report results, checkpoints or test outcomes that were not actually
  produced; quote the command and log location.

## Updating PLAN.md

- Check a box only when the item is implemented **and** verified; name the
  test, measurement or run that verifies it.
- Add newly discovered tasks under the right phase; do not add dates or time
  estimates.
- Keep `README.md` "Status" and "Known limitations" consistent with `PLAN.md`.
