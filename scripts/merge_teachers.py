"""Two PPO checkpoints -> one teacher checkpoint for Walking-Leap-Motion-Gait-Flat-Student-v1.

runner.SwitchTeacher holds the walking teacher (Gait-v2) as ``walk`` and the getting-up
teacher (Getup) as ``getup``; this writes their actors under those prefixes, as a PPO
checkpoint would hold one actor, so that train's ``--agent.resume`` loads it as the
teacher.

  uv run python scripts/merge_teachers.py WALK.pt GETUP.pt OUT_DIR/model_0.pt
"""

import argparse
from pathlib import Path

import torch


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("walk", type=Path)
  parser.add_argument("getup", type=Path)
  parser.add_argument("out", type=Path)
  args = parser.parse_args()
  actor = {}
  for prefix, path in (("walk", args.walk), ("getup", args.getup)):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in saved["actor_state_dict"].items():
      if key in ("std", "log_std"):  # rsl-rl 4 names, as mjlab's runner migrates them
        key = f"distribution.{key}_param"
      actor[f"{prefix}.{key}"] = value
  args.out.parent.mkdir(parents=True, exist_ok=True)
  sources = {"walk": str(args.walk), "getup": str(args.getup)}
  torch.save(
    {"actor_state_dict": actor, "iter": 0, "infos": {"teachers": sources}}, args.out
  )
  print(f"wrote {args.out}: {len(actor)} tensors from {sources}")


if __name__ == "__main__":
  main()
