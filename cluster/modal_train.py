"""Train a walking-leap task on a Modal GPU: the same mjlab `train` as train.sbatch.

  modal run cluster/modal_train.py::check          # GPU, driver, torch, secrets' keys
  modal run cluster/modal_train.py --iterations 10 --no-wandb --no-upload --wait
                                                   # smoke, waiting for its end
  modal run --detach cluster/modal_train.py --task Walking-Leap-Motion-Gait-Flat-v0 \
    --seed 42                                      # a full run; the laptop may close
  modal run --detach cluster/modal_train.py --task Walking-Leap-Motion-Gait-Flat-v1 \
    --resume <run folder> --checkpoint model_2999.pt --iterations 3000
                                                   # continue a run's policy

Without --wait the run is spawned and the command returns: with --detach it goes
on with the laptop off (a call that waits for its result dies with the laptop: a
first run stopped at iteration 1221 that way). The image installs uv and the
repository (pyproject.toml, uv.lock, src/) with `uv sync --frozen`, as on Rorqual.
Modal has internet, so W&B logs online (entity from WANDB_USERNAME, as in
cluster/train.sbatch). Runs are written straight into the Modal volume
`walking-leap-runs` (logs/rsl_rl/<experiment>/<run>), committed every
COMMIT_EVERY_S, so an interrupted run keeps its checkpoints and can be resumed; at
the end the run folder goes to Hugging Face at runs/<run>. Secrets (created in the Modal workspace): `wandb-mg` (WANDB_API_KEY of
an account in the mcgill-projects team; the workspace's `wandb-secret` is another
team's) and `huggingface-secret` (HF_TOKEN).
"""

import os
import subprocess
import threading
from pathlib import Path

import modal

ROOT = Path(__file__).parents[1]
REPO = "/repo"
WANDB_ENTITY = "mcgill-projects"
HF_REPO = "Autobrik/walking-leap"
COMMIT_EVERY_S = 300  # s between volume commits during training

image = (
  modal.Image.debian_slim(python_version="3.12")
  .apt_install("git", "libgl1", "libegl1", "libglib2.0-0")
  .pip_install("uv")
  .add_local_file(ROOT / "pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
  .add_local_file(ROOT / "uv.lock", f"{REPO}/uv.lock", copy=True)
  .add_local_file(ROOT / ".python-version", f"{REPO}/.python-version", copy=True)
  .add_local_file(ROOT / "README.md", f"{REPO}/README.md", copy=True)
  .add_local_file(ROOT / "LICENSE", f"{REPO}/LICENSE", copy=True)
  .add_local_dir(ROOT / "src", f"{REPO}/src", copy=True)
  .run_commands(f"cd {REPO} && uv sync --frozen --no-dev")
)
app = modal.App("walking-leap", image=image)
runs = modal.Volume.from_name("walking-leap-runs", create_if_missing=True)
secrets = [
  modal.Secret.from_name("wandb-mg"),
  modal.Secret.from_name("huggingface-secret"),
]


def sh(*cmd: str, **env: str) -> None:
  print("$", " ".join(cmd), flush=True)
  subprocess.run(cmd, cwd=REPO, check=True, env={**os.environ, **env})


@app.function(gpu="H100", timeout=15 * 60, secrets=secrets)
def check() -> None:
  """The GPU and its driver, torch on CUDA, MuJoCo Warp, and which secret keys exist
  (names only)."""
  sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv")
  sh(
    "uv",
    "run",
    "--frozen",
    "python",
    "-c",
    "import torch, mujoco, mujoco_warp, mjlab; "
    "print('torch', torch.__version__, 'cuda', torch.version.cuda, "
    "'available', torch.cuda.is_available()); "
    "print('mujoco', mujoco.__version__)",
  )
  for key in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    print(key, "set" if os.environ.get(key) else "missing")


@app.function(timeout=5 * 60, secrets=secrets)
def check_wandb() -> None:
  """Whether the W&B key reaches the project the runs go to (no GPU)."""
  sh(
    "uv",
    "run",
    "--frozen",
    "python",
    "-c",
    "import wandb; api = wandb.Api(); "
    f"p = api.project('walking-leap', entity='{WANDB_ENTITY}'); "
    "print('W&B user', api.viewer.username, '- project', p.entity + '/' + p.name)",
  )


@app.function(gpu="H100", timeout=5 * 60 * 60, secrets=secrets, volumes={"/runs": runs})
def train(
  task: str,
  seed: int,
  iterations: int,
  num_envs: int,
  wandb: bool,
  upload: bool,
  commit: str,
  resume: str = "",
  checkpoint: str = "",
) -> str:
  """One training run; returns the run folder's name. ``commit``: the local
  checkout the image was built from, written to the run folder."""
  print(f"commit {commit}, task {task}, seed {seed}", flush=True)
  name = f"{task}_s{seed}"
  cmd = ["uv", "run", "--frozen", "train", task]
  cmd += ["--env.scene.num-envs", str(num_envs), "--agent.seed", str(seed)]
  cmd += ["--agent.run-name", name]
  if iterations:
    cmd += ["--agent.max-iterations", str(iterations)]
  if wandb:
    cmd += ["--agent.logger", "wandb"]
  # mjlab writes logs/ under the working directory: make that the volume.
  Path("/runs/logs").mkdir(exist_ok=True)
  Path(REPO, "logs").symlink_to("/runs/logs")
  if resume:  # a run folder of the volume; older runs sit at its top level
    experiment = Path("/runs/logs/rsl_rl/walking_leap_motion")
    if not (experiment / resume).exists():
      experiment.mkdir(parents=True, exist_ok=True)
      sh("cp", "-r", f"/runs/{resume}", str(experiment / resume))
    cmd += ["--agent.resume", "True", "--agent.load-run", resume]
    cmd += ["--agent.load-checkpoint", checkpoint or "model_.*.pt"]
  done = threading.Event()

  def keep_committing() -> None:
    while not done.wait(COMMIT_EVERY_S):
      runs.commit()

  threading.Thread(target=keep_committing, daemon=True).start()
  try:
    sh(*cmd, WANDB_USERNAME=WANDB_ENTITY, PYTHONUNBUFFERED="1", WANDB_CONSOLE="off")
  finally:
    done.set()
    runs.commit()
  run = max(Path("/runs/logs/rsl_rl").glob(f"*/*_{name}"), key=os.path.getmtime)
  (run / "commit.txt").write_text(commit + "\n")
  runs.commit()
  if upload:
    sh(
      "uvx",
      "--from",
      "huggingface_hub",
      "hf",
      "upload",
      HF_REPO,
      str(run),
      f"runs/{run.name}",
    )
  return run.name


@app.local_entrypoint()
def main(
  task: str = "Walking-Leap-Motion-Gait-Flat-v0",
  seed: int = 42,
  iterations: int = 0,
  num_envs: int = 4096,
  wandb: bool = True,
  upload: bool = True,
  resume: str = "",
  checkpoint: str = "",
  wait: bool = False,
) -> None:
  head = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
  ).stdout.strip()
  dirty = subprocess.run(
    ["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True
  ).stdout.strip()
  commit = head + (" (src has uncommitted changes)" if dirty else "")
  args = (task, seed, iterations, num_envs, wandb, upload, commit, resume, checkpoint)
  if wait:
    print(train.remote(*args))
  else:
    call = train.spawn(*args)
    print(f"spawned {call.object_id}: {task} seed {seed} ({commit})")
