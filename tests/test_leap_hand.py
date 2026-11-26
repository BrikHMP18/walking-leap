"""The LEAP hand's driver (real/leap_hand/): its rig, limits, port, and its control loop on the
fake servos (the fake overrides only the bus layer, so the loop under test is the real one)."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REAL = Path(__file__).resolve().parents[1] / "real"
sys.path.insert(0, str(REAL))
import wlc  # noqa: E402
from leap_hand import config  # noqa: E402
from leap_hand import conventions as conv  # noqa: E402
from leap_hand import hand as hand_module  # noqa: E402
from leap_hand.config import (  # noqa: E402
  LeapHandConfig,
  find_port,
  load_rig,
  rig_from_dict,
)
from leap_hand.fake import FakeLeapHand  # noqa: E402
from leap_hand.hand import MODE_CURRENT_BASED_POSITION, HandError  # noqa: E402
from leap_hand.rate import Rate  # noqa: E402
from leap_hand.state import HandState  # noqa: E402
from leap_hand.third_party.leap_hand_utils import leap_hand_utils as lhu  # noqa: E402

HZ = 200.0
WATCHDOG_S = 0.15
STEP = 0.05
N = conv.NUM_JOINTS


def fast_config(**overrides):
  kw = dict(
    control_hz=HZ,
    watchdog_s=WATCHDOG_S,
    max_step_rad=STEP,
    hw_error_check_every=4,
    comm_error_limit=5,
  )
  kw.update(overrides)
  return LeapHandConfig(**kw)


def wait_until(pred, timeout_s=3.0, what="condition"):
  deadline = time.monotonic() + timeout_s
  while not pred():
    if time.monotonic() > deadline:
      pytest.fail(f"timed out waiting for {what}")
    time.sleep(0.005)


def torque_is_off(h):
  """All simulated servos off, the last torque write was 'off', the port is closed."""
  return (not h.sim_torque.any()) and h.torque_writes[-1] is False and not h.sim_open


@pytest.fixture
def hand():
  h = FakeLeapHand(fast_config())
  h.connect()
  yield h
  h.disconnect()


# ------------------------------------------------------------------- CMU's files, untouched

UPSTREAM_BLOBS = {  # git blob hashes of leap-hand/LEAP_Hand_API at b0d00c8
  "leap_hand_utils/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
  "leap_hand_utils/dynamixel_client.py": "fa0136e2e3e3c10219f85a95008c9f25b194cabc",
  "leap_hand_utils/leap_hand_utils.py": "2d81737aa5f9c2b6e8341c14a1bc6311229829a0",
  "LICENSE_LEAP_HAND_API.txt": "9b3eb50970ab214f4054dbfc9377789c2b9834e4",
}


@pytest.mark.parametrize("rel_path", sorted(UPSTREAM_BLOBS))
def test_vendored_file_is_verbatim(rel_path):
  data = (REAL / "leap_hand" / "third_party" / rel_path).read_bytes()
  blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
  assert blob == UPSTREAM_BLOBS[rel_path], f"{rel_path} differs from CMU's"


# ------------------------------------------------------------------- the rig and the port


def test_the_rig_loads_with_350_ma_and_finds_its_port():
  rig = load_rig()
  assert rig.name == "leap_right" and rig.path.endswith("leap_right.yaml")
  h = rig.hand
  assert h.port == "auto" and h.baudrate == 4_000_000 and h.motor_ids == conv.MOTOR_IDS
  assert (h.kp, h.ki, h.kd, h.abduction_gain_scale) == (600, 0, 200, 0.75)
  assert h.current_limit_ma == 350 and h.limits_rad is None
  assert (h.watchdog_s, h.control_hz, h.return_delay_time) == (0.5, 100.0, 0)
  assert (
    wlc.RIG == str(REAL / "leap_hand" / "leap_right.yaml") or "WL_RIG" in os.environ
  )


def test_unknown_keys_are_errors(tmp_path):
  rig_file = tmp_path / "bad.yaml"
  rig_file.write_text("name: x\nhand:\n  kP: 600\n")
  with pytest.raises(ValueError, match=r"unknown keys \['kP'\]"):
    load_rig(rig_file)
  with pytest.raises(ValueError, match="unknown top-level keys"):
    rig_from_dict({"name": "x", "host": {}})
  with pytest.raises(ValueError, match="missing 'name'"):
    rig_from_dict({"hand": {}})
  rig_file.write_text("- not\n- a mapping\n")
  with pytest.raises(TypeError, match="mapping"):
    load_rig(rig_file)
  assert rig_from_dict({"name": "bare"}).hand == LeapHandConfig()


@pytest.mark.parametrize(
  "field, value",
  [
    ("kp", -1), ("kp", 600.5), ("kp", 20000), ("ki", True), ("kd", "200"),
    ("abduction_gain_scale", 0.0), ("abduction_gain_scale", 1.5),
    ("control_hz", 0), ("max_step_rad", -0.1), ("watchdog_s", 0),
    ("current_limit_ma", 0), ("comm_error_limit", 0), ("hw_error_check_every", 0),
    ("motor_ids", (0, 1, 2)), ("return_delay_time", 300), ("return_delay_time", 1.5),
    ("packet_timeout_ms", 0), ("packet_timeout_ms", -2.0),
  ],
)  # fmt: skip
def test_invalid_values_are_rejected(field, value):
  with pytest.raises((ValueError, TypeError)):
    LeapHandConfig(**{field: value})


def test_find_port(tmp_path, monkeypatch):
  monkeypatch.delenv("WL_LEAP_PORT", raising=False)
  assert find_port("/dev/ttyUSB3") == "/dev/ttyUSB3"
  monkeypatch.setattr(config, "PORT_GLOB", str(tmp_path / "usb-FTDI_*-if00-port0"))
  with pytest.raises(OSError, match="no U2D2"):
    find_port()
  one = tmp_path / "usb-FTDI_USB__-__Serial_Converter_X1-if00-port0"
  one.touch()
  assert find_port() == str(one)
  (tmp_path / "usb-FTDI_USB__-__Serial_Converter_X2-if00-port0").touch()
  with pytest.raises(OSError, match="several"):
    find_port("auto")
  monkeypatch.setenv("WL_LEAP_PORT", "/dev/ttyUSB7")
  assert find_port("auto") == "/dev/ttyUSB7"


# ------------------------------------------------------------------- joints and limits


def test_sixteen_joints_in_bus_order():
  assert conv.MOTOR_IDS == tuple(range(16)) and len(set(conv.JOINT_NAMES)) == 16
  for i, name in enumerate(conv.JOINT_NAMES):
    assert name.startswith(conv.FINGERS[i // 4] + "_"), (i, name)
  assert all(i in conv.MOTOR_IDS for i in conv.ABDUCTION_IDS)


def test_limits_are_the_sdks_and_may_only_tighten():
  lower, upper = conv.sdk_limits()
  sdk_lower, sdk_upper = lhu.LEAPsim_limits()
  np.testing.assert_array_equal(lower, sdk_lower)
  np.testing.assert_array_equal(upper, sdk_upper)
  assert (lower < upper).all() and (lower <= 0).all() and (upper >= 0).all()
  tight = [(lo + 0.1, hi - 0.1) for lo, hi in zip(lower, upper, strict=True)]
  lo2, hi2 = conv.resolve_limits(tight)
  np.testing.assert_allclose(lo2, lower + 0.1)
  np.testing.assert_allclose(hi2, upper - 0.1)
  wide = list(tight)
  wide[3] = (lower[3] - 0.01, upper[3])
  with pytest.raises(ValueError, match="index_dip"):
    conv.resolve_limits(wide)
  inverted = list(tight)
  inverted[0] = (0.5, -0.5)
  with pytest.raises(ValueError, match="tighten"):
    conv.resolve_limits(inverted)
  with pytest.raises(ValueError, match="16"):
    conv.resolve_limits(tight[:15])
  wide12 = list(tight)
  wide12[12] = (float(lower[12]), float(upper[12]) + 0.5)
  with pytest.raises(ValueError, match="thumb_cmc"):
    LeapHandConfig(limits_rad=wide12)


def test_servo_conversion_is_the_sdks():
  q = np.linspace(-1.0, 2.0, 16)
  np.testing.assert_array_equal(conv.to_servo(q), lhu.LEAPsim_to_LEAPhand(q))
  np.testing.assert_array_equal(conv.from_servo(q), lhu.LEAPhand_to_LEAPsim(q))
  np.testing.assert_allclose(conv.from_servo(conv.to_servo(q)), q, atol=1e-12)
  np.testing.assert_allclose(
    conv.to_servo(np.zeros(16)), 3.14159
  )  # the SDK's flat, not pi


# ------------------------------------------------------------------- the rate loop


def test_rate_keeps_the_period_and_counts_overruns():
  rate = Rate(200.0)
  t0 = time.monotonic()
  for _ in range(40):
    rate.sleep()
  assert 0.17 < time.monotonic() - t0 < 0.35
  st = rate.stats
  assert st.ticks == 40 and 150 < st.hz < 250 and st.overruns == 0
  assert st.jitter_p50_ms <= st.jitter_p99_ms < 20
  late = Rate(100.0)
  time.sleep(0.05)  # five periods late
  late.sleep()
  assert (
    late.stats.overruns == 1 and late.remaining() > 0
  )  # reset, not run back to back
  assert Rate(50.0).stats.hz == 0.0
  with pytest.raises(ValueError):
    Rate(0.0)


# ------------------------------------------------------------------- connect, configure


def test_connect_configures_with_torque_off_and_holds_present_pose():
  q0 = np.full(N, 0.3)
  h = FakeLeapHand(fast_config(), q0=q0)
  try:
    h.connect()
    assert h.is_connected
    assert h.eeprom_writes_with_torque == 0  # operating mode written with torque off
    assert (h.sim_mode == MODE_CURRENT_BASED_POSITION).all()
    assert (h.sim_goal_current == 350).all()
    assert h.sim_torque.all()
    assert h.torque_writes[:2] == [False, True]  # off before configuring, on last
    np.testing.assert_allclose(h.sim_goal, h.sim_q)  # goal = present pose: no motion
    s = h.read()
    np.testing.assert_allclose(s.q, q0, atol=1e-6)
    np.testing.assert_allclose(s.q_cmd, q0, atol=1e-6)
    np.testing.assert_allclose(s.q_target, q0, atol=1e-6)
    assert s.torque_on and not s.watchdog_active and not s.hw_error.any()
  finally:
    h.disconnect()
  assert torque_is_off(h) and not h.is_connected


def test_connect_writes_gains_with_abduction_scale(hand):
  p, i, d = hand.sim_gains["p"], hand.sim_gains["i"], hand.sim_gains["d"]
  assert (p[1], i[1], d[1]) == (600, 0, 200)
  for j in conv.ABDUCTION_IDS:
    assert (p[j], d[j]) == (450, 150)
  assert hand.gains == (600, 0, 200)


def test_connect_fails_when_a_motor_is_missing():
  h = FakeLeapHand(fast_config())
  h.missing_ids = (7,)
  with pytest.raises(HandError, match=r"IDs \[7\] did not answer"):
    h.connect()
  assert not h.is_connected and not h.sim_open and not h.sim_torque.any()
  h.missing_ids = ()
  h.connect()  # recoverable
  assert h.is_connected
  h.disconnect()
  assert torque_is_off(h)


def test_connect_without_a_u2d2_fails_cleanly(tmp_path, monkeypatch):
  monkeypatch.delenv("WL_LEAP_PORT", raising=False)
  monkeypatch.setattr(config, "PORT_GLOB", str(tmp_path / "none-*"))
  h = hand_module.LeapHand(fast_config())
  with pytest.raises(HandError, match="no U2D2"):
    h.connect()
  assert not h.is_connected


def test_disconnect_is_idempotent_and_reconnect_works():
  h = FakeLeapHand(fast_config())
  h.disconnect()  # never connected: no-op
  h.connect()
  h.disconnect()
  h.disconnect()
  assert torque_is_off(h)
  h.connect()
  assert h.is_connected and h.sim_torque.all()
  with pytest.raises(HandError, match="already connected"):
    h.connect()
  h.disconnect()
  assert torque_is_off(h)


def test_connect_registers_atexit_torque_off(monkeypatch):
  calls = []

  class FakeAtexit:
    def register(self, fn):
      calls.append(("register", fn))
      return fn

    def unregister(self, fn):
      calls.append(("unregister", fn))

  monkeypatch.setattr(hand_module, "atexit", FakeAtexit())
  h = FakeLeapHand(fast_config())
  h.connect()
  assert calls == [("register", h.disconnect)]
  h.disconnect()
  assert calls[-1] == ("unregister", h.disconnect)


def test_loop_runs_at_the_configured_rate(hand):
  time.sleep(0.5)
  st = hand.stats
  assert st.ticks >= 60
  assert abs(st.hz - HZ) / HZ < 0.2
  assert st.comm_errors == 0 and st.watchdog_hits == 0 and st.hw_error_events == 0


# ------------------------------------------------------------------- torque off paths


def test_context_manager_turns_torque_off_on_exception():
  h = FakeLeapHand(fast_config())
  with pytest.raises(ValueError, match="inside the block"), h as hh:
    assert hh is h and h.sim_torque.all()
    raise ValueError("inside the block")
  assert torque_is_off(h)


PUBLIC_CALLS = {
  "read": lambda h: h.read(),
  "set_target": lambda h: h.set_target(np.zeros(N)),
  "move_to": lambda h: h.move_to(np.zeros(N), 0.1),
  "hold": lambda h: h.hold(),
  "release": lambda h: h.release(),
  "engage": lambda h: h.engage(),
  "set_gains": lambda h: h.set_gains(600, 0, 200),
}


@pytest.mark.parametrize("name", sorted(PUBLIC_CALLS))
def test_public_calls_raise_and_torque_is_off_after_loop_exception(hand, name):
  hand.raise_on_next_read = RuntimeError("simulated driver crash")
  wait_until(lambda: not hand.sim_open, what="the loop to stop and close the port")
  assert torque_is_off(hand) and not hand.is_connected
  with pytest.raises(HandError, match="loop stopped"):
    PUBLIC_CALLS[name](hand)
  assert torque_is_off(hand)
  assert hand.stats.ticks > 0  # still readable
  hand.disconnect()  # idempotent after a crash
  assert torque_is_off(hand)


def test_public_calls_raise_when_not_connected():
  h = FakeLeapHand(fast_config())
  for call in PUBLIC_CALLS.values():
    with pytest.raises(HandError, match="not connected"):
      call(h)


def test_comm_errors_below_the_limit_are_tolerated(hand):
  hand.fail_next_reads = 4
  wait_until(lambda: hand.stats.comm_errors == 4, what="four counted errors")
  wait_until(lambda: hand.stats.comm_error_streak == 0, what="the streak to reset")
  assert hand.is_connected and hand.sim_torque.all()
  hand.read()


def test_comm_error_limit_stops_loop_with_torque_off(hand):
  hand.fail_next_reads = 5
  wait_until(lambda: not hand.sim_open, what="the loop to stop")
  assert torque_is_off(hand)
  assert hand.stats.comm_errors == 5
  with pytest.raises(HandError, match="5 consecutive bus errors"):
    hand.read()


# ------------------------------------------------------------------- targets and motion


def test_set_target_clips_to_limits(hand):
  clipped = hand.set_target(hand.upper + 1.0)
  np.testing.assert_allclose(clipped, hand.upper)
  wait_until(lambda: np.allclose(hand.read().q_target, hand.upper), what="the target")
  assert hand.move_to(hand.lower - 1.0, duration_s=0.5)
  wait_until(lambda: np.allclose(hand.read().q, hand.lower, atol=0.02), what="arrival")
  np.testing.assert_allclose(hand.read().q_target, hand.lower)


def test_set_target_validates_input(hand):
  with pytest.raises(ValueError, match="shape"):
    hand.set_target(np.zeros(15))
  q = np.zeros(N)
  q[4] = np.nan
  with pytest.raises(ValueError, match="NaN"):
    hand.set_target(q)
  with pytest.raises(ValueError, match="duration_s"):
    hand.set_target(np.zeros(N), duration_s=-1.0)
  with pytest.raises(ValueError, match="duration_s"):
    hand.set_target(np.zeros(N), duration_s=float("inf"))


def test_setpoint_is_rate_limited(hand):
  q = np.zeros(N)
  q[2] = 1.0
  ticks_before = hand.stats.ticks
  hand.set_target(q)
  time.sleep(0.02)
  s = hand.read()
  moved = abs(s.q_cmd[2])
  assert 0 < moved <= STEP * (hand.stats.ticks - ticks_before + 1) + 1e-9
  assert hand.move_to(q, duration_s=0.1)


def test_duration_slows_the_move(hand):
  q = np.zeros(N)
  q[1] = 0.5  # 10 ticks at max_step_rad; 80 ticks with duration_s=0.4
  t0 = time.monotonic()
  assert hand.move_to(q, duration_s=0.4)
  assert 0.3 <= time.monotonic() - t0 <= 1.0
  np.testing.assert_allclose(hand.read().q, q, atol=0.02)


def test_move_to_times_out_when_the_hand_cannot_follow():
  h = FakeLeapHand(fast_config(), v_max=0.1)  # a very slow servo
  with h:
    q = np.zeros(N)
    q[6] = 1.0
    assert h.move_to(q, duration_s=0.05, timeout_s=0.2) is False
  assert torque_is_off(h)


def test_watchdog_freezes_setpoint_but_keeps_torque():
  h = FakeLeapHand(fast_config(max_step_rad=0.005))  # 1.5 rad takes 1.5 s
  with h:
    q = np.zeros(N)
    q[3] = 1.5
    h.set_target(q)
    wait_until(lambda: h.read().watchdog_active, what="the watchdog")
    s = h.read()
    assert h.stats.watchdog_hits == 1
    assert s.torque_on and h.sim_torque.all()
    np.testing.assert_allclose(s.q_target, s.q_cmd)
    assert 0 < s.q_cmd[3] < 1.5
    time.sleep(0.05)
    assert h.read().q_cmd[3] == s.q_cmd[3]  # frozen
    h.set_target(q)  # a new target re-arms the watchdog
    wait_until(lambda: not h.read().watchdog_active, what="the watchdog to re-arm")
    wait_until(lambda: h.read().q_cmd[3] > s.q_cmd[3], what="motion to resume")
  assert torque_is_off(h)


def test_hold_freezes_the_setpoint_and_clears_the_watchdog_flag():
  h = FakeLeapHand(fast_config(max_step_rad=0.005))
  with h:
    q = np.zeros(N)
    q[3] = 1.5
    h.set_target(q)
    wait_until(lambda: h.read().watchdog_active, what="the watchdog")
    h.hold()
    wait_until(lambda: not h.read().watchdog_active, what="hold to clear the flag")
    np.testing.assert_allclose(h.read().q_target, h.read().q_cmd)
  h = FakeLeapHand(fast_config(max_step_rad=0.005, watchdog_s=5.0))
  with h:
    q = np.zeros(N)
    q[9] = 1.0
    h.set_target(q)
    wait_until(lambda: h.read().q_cmd[9] > 0.02, what="motion")
    h.hold()
    wait_until(lambda: np.array_equal(h.read().q_target, h.read().q_cmd), what="hold")
    s = h.read()
    assert 0.02 < s.q_cmd[9] < 1.0  # stopped on the way
    time.sleep(0.05)
    assert h.read().q_cmd[9] == s.q_cmd[9] and not h.read().watchdog_active
  assert torque_is_off(h)


def test_release_and_engage(hand):
  hand.release()
  assert not hand.sim_torque.any() and not hand.read().torque_on
  with pytest.raises(HandError, match="torque is off"):
    hand.set_target(np.zeros(N))
  q_new = np.full(N, 0.2)
  hand.move_sim(q_new)  # the released fingers are moved by hand
  wait_until(lambda: np.allclose(hand.read().q, q_new), what="the new pose to be read")
  np.testing.assert_allclose(
    hand.read().q_cmd, q_new, atol=1e-6
  )  # the setpoint follows
  hand.engage()
  assert hand.sim_torque.all()
  s = hand.read()
  assert s.torque_on and not s.watchdog_active
  np.testing.assert_allclose(s.q_cmd, q_new, atol=1e-6)
  np.testing.assert_allclose(hand.sim_goal, conv.to_servo(q_new), atol=1e-6)  # no jump
  assert hand.move_to(np.zeros(N), duration_s=0.1)


def test_set_gains_rewrites_the_servos(hand):
  hand.set_gains(800, 10, 100)
  p, i, d = hand.sim_gains["p"], hand.sim_gains["i"], hand.sim_gains["d"]
  assert (p[1], i[1], d[1]) == (800, 10, 100)
  assert (p[0], d[0]) == (600, 75)
  assert hand.gains == (800, 10, 100)
  for bad in [(-1, 0, 0), (600.5, 0, 0), (20000, 0, 0)]:
    with pytest.raises(ValueError):
      hand.set_gains(*bad)
  assert hand.gains == (800, 10, 100)


def test_hardware_error_holds_and_refuses_targets(hand):
  hand.inject_hw_error(5)
  wait_until(lambda: hand.read().hw_error[5] != 0, what="the error to be read")
  s = hand.read()
  assert hand.stats.hw_error_events == 1
  assert s.torque_on  # the driver does not drop torque by itself
  assert hand.sim_torque[[i for i in range(N) if i != 5]].all()
  np.testing.assert_allclose(s.q_target, s.q_cmd)
  with pytest.raises(HandError, match="hardware error on IDs \\[5\\]"):
    hand.set_target(np.zeros(N))
  time.sleep(0.05)
  assert hand.stats.hw_error_events == 1  # logged once, not every check
  hand.release()
  with pytest.raises(HandError, match="torque on failed for IDs \\[5\\]"):
    hand.engage()  # the faulty servo refuses torque; the others are turned off again
  assert not hand.sim_torque.any() and not hand.read().torque_on


# ------------------------------------------------------------------- state


def test_read_returns_a_copy_and_states_are_plain(hand):
  s = hand.read()
  s.q[:] = 99.0
  assert not (hand.read().q == 99.0).any()
  s = hand.read()
  d = s.to_dict()
  json.dumps(d)
  s2 = HandState.from_dict(d)
  for name in ("q", "dq", "current_ma", "q_cmd", "q_target"):
    np.testing.assert_array_equal(getattr(s2, name), getattr(s, name))
  assert s2.hw_error.dtype == np.uint8 and (s2.hw_error == s.hw_error).all()
  assert hand.stats.bus_ms_p99 >= hand.stats.bus_ms_p50 >= 0.0


def test_fake_reports_current_while_tracking():
  h = FakeLeapHand(fast_config(), v_max=0.5)
  with h:
    q = np.zeros(N)
    q[8] = 0.8
    h.set_target(q)
    wait_until(lambda: h.read().current_ma[8] > 0, what="current")
    assert h.read().current_ma.max() <= 350
  assert torque_is_off(h)


# ------------------------------------------------------------------- the loop's Hand on it


def test_the_loops_hand_drives_the_fake_in_the_repos_order():
  q0 = np.zeros(16)
  q0[wlc.JOINT_NAMES.index("if_mcp")] = 0.4
  with wlc.Hand((1200, 0, 600), fake=True, q0=q0) as hand:
    assert hand.cfg.current_limit_ma == 350 and hand.h.gains == (1200, 0, 600)
    assert (hand.h.sim_mode == MODE_CURRENT_BASED_POSITION).all()
    assert (hand.h.sim_goal_current == 350).all()
    s, q, dq, cur, qc = hand.read()
    np.testing.assert_allclose(q, q0, atol=1e-6)  # the bus order is undone
    target = q0.copy()
    target[wlc.JOINT_NAMES.index("mf_pip")] = 0.3
    sent = hand.target(target)
    wait_until(
      lambda: abs(hand.read()[1][wlc.JOINT_NAMES.index("mf_pip")] - 0.3) < 0.02
    )
    np.testing.assert_allclose(sent, np.clip(target, hand.lo, hand.hi))
    assert np.isfinite(hand.temps()).all()
  assert torque_is_off(hand.h)
