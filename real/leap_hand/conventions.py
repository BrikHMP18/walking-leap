"""The LEAP hand's joint space in bus order: names, units, limits, conversions.

- Bus ID == index in every array (`MOTOR_IDS[i] == i`). The repo's own joint order and its
  map to the bus are in `wlc.py` (`JOINT_NAMES`, `MOTOR_OF`).
- Canonical unit: "LEAPsim" radians, 0 = flat, positive = curl. The servo's "LEAPhand" unit
  (3.14159 = flat) appears only inside `hand.py`, through `to_servo` and `from_servo`,
  which call the vendored SDK helpers (the offset is not re-derived here).
- Limits default to the SDK's `LEAPsim_limits()`; a rig file may tighten them, never widen.

The names come from watching each motor move alone on the real (right) hand. For the
index, middle and ring finger the four IDs are the side-to-side knuckle motor (`abd`: positive swings the finger toward the little-finger
side), the knuckle flexion motor (`mcp`), the middle joint (`pip`) and the tip joint
(`dip`); a positive value closes a bending joint toward the palm. The thumb's four IDs run
from its base to its tip. "ring" is the SDK's and the sim's name for the third finger.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .third_party.leap_hand_utils import leap_hand_utils as _lhu

NUM_JOINTS = 16
MOTOR_IDS = tuple(range(NUM_JOINTS))  # bus ID == index in every array
FINGERS = ("index", "middle", "ring", "thumb")
JOINT_NAMES = (  # observed on the hand: see the module docstring
  "index_abd",
  "index_mcp",
  "index_pip",
  "index_dip",
  "middle_abd",
  "middle_mcp",
  "middle_pip",
  "middle_dip",
  "ring_abd",
  "ring_mcp",
  "ring_pip",
  "ring_dip",
  "thumb_cmc",
  "thumb_axl",
  "thumb_mcp",
  "thumb_ipl",
)
ABDUCTION_IDS = (0, 4, 8)  # the side-to-side motors; the SDK gives these 0.75x gains
UNITS = "leapsim_rad"  # canonical: radians, 0 = flat, + = curl


def sdk_limits() -> tuple[np.ndarray, np.ndarray]:
  """The SDK's joint limits in canonical radians, as (lower, upper) float64 arrays."""
  lower, upper = _lhu.LEAPsim_limits()
  return np.asarray(lower, dtype=float).copy(), np.asarray(upper, dtype=float).copy()


def resolve_limits(
  rig_limits: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Limits to enforce: the SDK defaults, tightened by `rig_limits` if given.

  `rig_limits` is 16 (lower, upper) pairs in canonical radians. A pair that widens the
  SDK range or is not ordered raises ValueError: a rig may tighten, never widen.
  """
  lower, upper = sdk_limits()
  if rig_limits is None:
    return lower, upper
  arr = np.asarray(rig_limits, dtype=float)
  if arr.shape != (NUM_JOINTS, 2):
    raise ValueError(
      f"limits must be {NUM_JOINTS} (lower, upper) pairs, got {arr.shape}"
    )
  bad = [
    f"{JOINT_NAMES[i]}: {tuple(arr[i])} outside SDK {(lower[i], upper[i])}"
    for i in range(NUM_JOINTS)
    if arr[i, 0] < lower[i] or arr[i, 1] > upper[i] or arr[i, 0] >= arr[i, 1]
  ]
  if bad:
    raise ValueError("rig limits may only tighten the SDK limits: " + "; ".join(bad))
  return arr[:, 0].copy(), arr[:, 1].copy()


def to_servo(q_sim: np.ndarray) -> np.ndarray:
  """Canonical (LEAPsim) radians to the servo's LEAPhand radians (SDK helper)."""
  return np.asarray(_lhu.LEAPsim_to_LEAPhand(q_sim), dtype=float)


def from_servo(q_hand: np.ndarray) -> np.ndarray:
  """The servo's LEAPhand radians to canonical (LEAPsim) radians (SDK helper)."""
  return np.asarray(_lhu.LEAPhand_to_LEAPsim(q_hand), dtype=float)
