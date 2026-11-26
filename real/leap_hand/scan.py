"""Read-only bus scan: which IDs answer, their model, present position and error status.

    real/wl scan [--rig FILE] [--port /dev/ttyUSB0] [--id-max 20] [--all-baudrates]

Pings IDs 0 to --id-max-1 at the rig's baudrate (then, with --all-baudrates or when
nothing answered, at the other common ones) and prints a table per responding ID: model
number and name, firmware, torque enable, operating mode, hardware error status, present
position in servo units and in canonical radians (0 = flat). Nothing is written to the
motors: no torque, no motion. Safe with the hand powered; nothing else may use the port.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import conventions as conv
from .config import DEFAULT_RIG, find_port, load_rig
from .hand import ADDR_HARDWARE_ERROR_STATUS, ADDR_OPERATING_MODE, HandError
from .third_party.leap_hand_utils import dynamixel_client as dxc

# The hand ships at 4 Mbps; 57600 is the Dynamixel factory default after a reset.
COMMON_BAUDRATES = (4_000_000, 57_600, 1_000_000, 2_000_000, 3_000_000, 115_200)
MODEL_NAMES = {
    1190: "XL330-M077-T", 1200: "XL330-M288-T", 1210: "XC330-T181-T", 1220: "XC330-T288-T",
    1230: "XC330-M181-T", 1240: "XC330-M288-T", 1020: "XM430-W350", 1060: "XL430-W250",
}  # fmt: skip
ADDR_FIRMWARE_VERSION = 6
OPERATING_MODES = {0: "current", 1: "velocity", 3: "position", 4: "extended position",
                   5: "current-based position", 16: "PWM"}  # fmt: skip


def read(
  client: dxc.DynamixelClient, motor_id: int, address: int, size: int
) -> int | None:
  """One register, or None if the motor did not answer."""
  fn = {1: client.packet_handler.read1ByteTxRx, 2: client.packet_handler.read2ByteTxRx,
          4: client.packet_handler.read4ByteTxRx}[size]  # fmt: skip
  value, comm, err = fn(client.port_handler, motor_id, address)
  if comm != client.dxl.COMM_SUCCESS or err != 0:
    return None
  return dxc.unsigned_to_signed(value, size) if size == 4 else value


def scan(port: str, baud: int, id_max: int) -> list[tuple[int, int]]:
  """(id, model_number) for every ID that answers a ping. Opens and closes the port."""
  client = dxc.DynamixelClient([], port, baud)
  client.connect()
  try:
    found = []
    for motor_id in range(id_max):
      model, comm, _ = client.packet_handler.ping(client.port_handler, motor_id)
      if comm == client.dxl.COMM_SUCCESS:
        found.append((motor_id, model))
    return found
  finally:
    client.port_handler.closePort()  # no torque write: nothing was enabled
    dxc.DynamixelClient.OPEN_CLIENTS.discard(client)


def report(port: str, baud: int, found: list[tuple[int, int]]) -> None:
  client = dxc.DynamixelClient([], port, baud)
  client.connect()
  try:
    print(f"{'ID':>3} {'name':<11} {'model':<13} {'fw':>3} {'torque':>6} "
              f"{'mode':<24} {'hw_err':>6} {'pos_servo':>10} {'pos_rad':>8}")  # fmt: skip
    for motor_id, model in found:
      fw = read(client, motor_id, ADDR_FIRMWARE_VERSION, 1)
      torque = read(client, motor_id, dxc.ADDR_TORQUE_ENABLE, 1)
      mode = read(client, motor_id, ADDR_OPERATING_MODE, 1)
      hw = read(client, motor_id, ADDR_HARDWARE_ERROR_STATUS, 1)
      pos = read(client, motor_id, dxc.ADDR_PRESENT_POSITION, 4)
      name = conv.JOINT_NAMES[motor_id] if motor_id < conv.NUM_JOINTS else "?"
      pos_servo = "?" if pos is None else f"{pos * dxc.DEFAULT_POS_SCALE:.3f}"
      pos_rad = (
        "?" if pos is None else f"{conv.from_servo(pos * dxc.DEFAULT_POS_SCALE):+.3f}"
      )
      print(f"{motor_id:>3} {name:<11} {MODEL_NAMES.get(model, str(model)):<13} "
                  f"{fw if fw is not None else '?':>3} {'on' if torque else 'off':>6} "
                  f"{OPERATING_MODES.get(mode, str(mode)):<24} "
                  f"{hw if hw is not None else '?':>6} {pos_servo:>10} {pos_rad:>8}")  # fmt: skip
  finally:
    client.port_handler.closePort()
    dxc.DynamixelClient.OPEN_CLIENTS.discard(client)


def _main(argv: list[str] | None) -> int:
  parser = argparse.ArgumentParser(
    prog="real/wl scan", description=__doc__.split("\n\n")[0]
  )
  parser.add_argument("--rig", default=os.environ.get("WL_RIG") or str(DEFAULT_RIG))
  parser.add_argument(
    "--port", default=None, help="serial port (default: the rig's, auto)"
  )
  parser.add_argument("--id-max", type=int, default=20, help="scan IDs 0 .. id-max-1")
  parser.add_argument("--all-baudrates", action="store_true",
                        help="also scan the other common baudrates")  # fmt: skip
  args = parser.parse_args(argv)
  rig = load_rig(args.rig)
  port = find_port(args.port or rig.hand.port)
  bauds = [rig.hand.baudrate] + [b for b in COMMON_BAUDRATES if b != rig.hand.baudrate]
  print(f"port {port}")
  expected = set(conv.MOTOR_IDS)
  for baud in bauds:
    try:
      found = scan(port, baud, args.id_max)
    except OSError as exc:
      print(f"baud {baud}: could not open the port ({exc})")
      return 2
    if found:
      ids = [i for i, _ in found]
      print(f"baud {baud}: IDs {ids}")
      report(port, baud, found)
      missing = sorted(expected - set(ids))
      print(f"missing IDs: {missing}" if missing else "all 16 IDs answered")
      if not args.all_baudrates:
        return 0 if not missing else 1
    else:
      print(f"baud {baud}: no response")
      if not args.all_baudrates and baud != rig.hand.baudrate:
        continue
  print("\nIf nothing answered: check the USB cable, the hand's power supply, the port "
          "(`ls /dev/serial/by-id/`, `dmesg | tail`) and the baudrate in the rig file.")  # fmt: skip
  return 1


def main(argv: list[str] | None = None) -> int:
  """The scan, with one line for the expected failures and 130 on Ctrl-C."""
  try:
    return _main(argv)
  except KeyboardInterrupt:
    print("interrupted", file=sys.stderr)
    return 130
  except (HandError, OSError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1
