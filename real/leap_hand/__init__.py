"""The LEAP hand's driver: 16 Dynamixel servos on a U2D2 adapter, one control loop.

The only code in the repo that talks to the motors.

- `hand.LeapHand` opens the port, puts every motor in current-based position mode with the
  rig's current limit (350 mA) and gains, turns torque on at the present pose (connecting
  does not move the hand), then runs a 100 Hz loop: a per-tick step limit on the setpoint,
  a watchdog that holds when no target comes, hardware-error checks, and torque off on
  every way out (`disconnect`, `with`, `atexit`, `raise_on_signals`, too many bus errors).
- `fake.FakeLeapHand` is the same class on simulated servos (`--fake`, the tests).
- `leap_right.yaml` holds every number that reaches the motors; `config.py` loads it and
  finds the U2D2 (`find_port`).
- `scan.py` reads the bus and writes nothing (`real/wl scan`).
- `third_party/` is CMU's LEAP_Hand_API, unmodified (CC BY-NC 4.0, its README).

Needs numpy and pyyaml; the real motors also need pyserial and dynamixel-sdk
(`real/requirements.txt`, installed by `real/setup`).
"""
