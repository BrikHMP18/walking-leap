# Vendored: LEAP Hand API (CMU)

Copied verbatim. Don't edit these files: a change goes in `leap_hand/hand.py`, which
subclasses or wraps them.

| Field | Value |
|---|---|
| Upstream | https://github.com/leap-hand/LEAP_Hand_API, `python/leap_hand_utils/` |
| Upstream commit checked | `b0d00c8` |
| Licence | CC BY-NC 4.0, `LICENSE_LEAP_HAND_API.txt` (identical to upstream `LICENSE`). Non-commercial use, with attribution. The rest of `real/` is under the repo's licence. |

Git blob hashes (`git hash-object`), identical to the upstream blobs; `tests/test_leap_hand.py`
recomputes them, so an accidental edit fails the tests:

```
e69de29bb2d1d6434b8b29ae775ad8c2e48c5391  leap_hand_utils/__init__.py
fa0136e2e3e3c10219f85a95008c9f25b194cabc  leap_hand_utils/dynamixel_client.py
2d81737aa5f9c2b6e8341c14a1bc6311229829a0  leap_hand_utils/leap_hand_utils.py
9b3eb50970ab214f4054dbfc9377789c2b9834e4  LICENSE_LEAP_HAND_API.txt
```

## What the driver uses

- `dynamixel_client.py`: `DynamixelClient` (port, sync read and write, torque) and
  `DynamixelReader`. It imports `dynamixel_sdk` only when a client is made, so the fake
  hand and the tests run without it.
- `leap_hand_utils.py`: `LEAPsim_limits`, `LEAPsim_to_LEAPhand`, `LEAPhand_to_LEAPsim`
  (`leap_hand/conventions.py`).

## Known upstream quirks (left as is)

- `DynamixelReader.read()` does not raise on a failed transfer: it logs and returns the
  previous values. `leap_hand/hand.py` counts failures through `handle_packet_result`.
- Importing `dynamixel_client` registers an `atexit` handler that disables torque on every
  open client (one attempt, no retries).
- LEAP's own example (`python/main.py` upstream) enables torque before writing the gains
  and writes a fixed open pose as the goal. `leap_hand/hand.py` enables torque last, with
  the goal set to the present position, so connecting does not move the hand.
