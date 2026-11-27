"""A trained animation policy (the Walking-Leap-Motion-* tasks) in plain numpy, for the real loop.

Its inputs, in the task's order (motion_cfg.py, the actor's terms):

  reference_joints    the clip's 16 joints and their 16 speeds at this frame
  reference_gravity   the clip's "down" in the palm's axes (3)
  base_ang_vel        the IMU's gyro, rad/s, palm axes (3)          - not in the NoImu tasks
  projected_gravity   the IMU's "down", unit vector, palm axes (3)  - not in the NoImu tasks
  joint_pos           the encoders minus the stance (16)
  joint_vel           the encoders' speeds (16)
  actions             its last output (16)

89 numbers with the IMU, 83 without. Output: 16 joint targets, stance + scale * action, inside the
joint limits, 50 times a second. Files made by export_policy.py.
"""

import os

import numpy as np

INPUTS = (
  "reference_joints",
  "reference_gravity",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)
IMU_INPUTS = ("base_ang_vel", "projected_gravity")


def elu(x):
  return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)


class MotionPolicy:
  def __init__(self, path):
    d = np.load(path)
    self.name = os.path.basename(path)[:-4]
    n = sum(1 for k in d.files if k[0] == "w" and k[1:].isdigit())
    self.W = [(d[f"w{i}"].astype(float), d[f"b{i}"].astype(float)) for i in range(n)]
    self.mean, self.std = d["mean"].astype(float), d["std"].astype(float)
    self.default, self.scale = d["default"].astype(float), float(d["scale"])
    self.lo, self.hi = d["lo"].astype(float), d["hi"].astype(float)
    self.imu = self.mean.size == 89
    assert self.mean.size in (83, 89), (
      f"{self.mean.size} inputs: not a Walking-Leap-Motion policy"
    )
    # The clips it may play: it runs only while one of them plays. Older files name one
    # clip, which covers the clips whose names start with it.
    self.clips = [str(c) for c in d["clips"]] if "clips" in d.files else None
    self.clip = str(d["clip"]) if "clip" in d.files else "hello"
    # The reference it saw in training (from the task's motion file). Without it, the loop's clip.
    self.ref = (
      (d["ref_q"].astype(float), d["ref_dq"].astype(float), d["ref_g"].astype(float))
      if "ref_q" in d.files
      else None
    )
    self.last = np.zeros(16)

  def covers(self, name):
    return name in self.clips if self.clips is not None else name.startswith(self.clip)

  def reset(self, clip, keep_last=False):
    """At a clip's start. ``keep_last``: clips back to back, as in a training library."""
    if not keep_last:
      self.last = np.zeros(16)
    if self.ref is not None:
      self.q, self.dq, self.g = self.ref
    else:
      self.q = clip.q
      self.dq = np.gradient(self.q, 1 / 50, axis=0)
      self.g = getattr(clip, "ref_grav", clip.grav)

  def obs(self, i, q, dq, gyro, gravity):
    imu = [gyro, gravity] if self.imu else []
    return np.concatenate(
      [self.q[i], self.dq[i], self.g[i], *imu, q - self.default, dq, self.last]
    )

  def net(self, o):
    x = (o - self.mean) / self.std
    for k, (W, b) in enumerate(self.W):
      x = W @ x + b
      if k < len(self.W) - 1:
        x = elu(x)
    return x

  def act(self, i, q, dq, gyro, gravity):
    i = min(i, len(self.q) - 1)
    o = self.obs(
      i, np.asarray(q), np.asarray(dq), np.asarray(gyro), np.asarray(gravity)
    )
    a = self.net(o)
    self.last = a
    return np.clip(self.default + self.scale * a, self.lo, self.hi)


# A velocity-command policy: the student of Walking-Leap-Motion-Gait-Flat-Student-v0. It never
# sees a clip; its inputs (motion_cfg.py, the student's terms), each over the last HISTORY steps,
# oldest first, term after term (mjlab's history; after a reset every step is the first one):
VELOCITY_INPUTS = (
  "command",  # forward m/s, left m/s, turn rad/s (palm heading): the keyboard (3)
  "base_ang_vel",  # the IMU's gyro (3)
  "projected_gravity",  # the IMU's "down" (3)
  "joint_pos",  # the encoders minus the stance (16)
  "joint_vel",  # the encoders' speeds (16)
  "actions",  # its last output (16)
)
VELOCITY_SIZES = (3, 3, 3, 16, 16, 16)
# Clock students (Student-v3, Velocity-v6) also read a gait clock right after the
# command: (sin, cos) of 2 pi times the phase of a cycle of ``clock_period`` s, which the
# policy runs itself while a command is given and holds at 0 while standing (zero
# command), as the clip's phase does in training.
CLOCK_INPUTS = ("command", "clock") + VELOCITY_INPUTS[1:]
CLOCK_SIZES = (3, 2) + VELOCITY_SIZES[1:]


class VelocityPolicy(MotionPolicy):
  """``kind`` "velocity" files of export_policy.py. ``commands``: the velocity of each
  command it was trained on (name -> 3 numbers), for the keyboard."""

  def __init__(self, path):
    d = np.load(path)
    assert "kind" in d.files and str(d["kind"]) == "velocity", (
      f"{path}: not a velocity policy"
    )
    self.name = os.path.basename(path)[:-4]
    n = sum(1 for k in d.files if k[0] == "w" and k[1:].isdigit())
    self.W = [(d[f"w{i}"].astype(float), d[f"b{i}"].astype(float)) for i in range(n)]
    self.mean, self.std = d["mean"].astype(float), d["std"].astype(float)
    self.default, self.scale = d["default"].astype(float), float(d["scale"])
    self.lo, self.hi = d["lo"].astype(float), d["hi"].astype(float)
    self.history = int(d["history"])
    self.clock_period = float(d["clock_period"]) if "clock_period" in d.files else None
    # Per command (aligned with command_names), when the clip's cycles differ by command:
    # the clock runs at the period of the key whose velocity the command is closest to.
    self.clock_periods = (
      d["clock_periods"].astype(float) if "clock_periods" in d.files else None
    )
    sizes = VELOCITY_SIZES if self.clock_period is None else CLOCK_SIZES
    assert self.mean.size == self.history * sum(sizes), "inputs do not add up"
    self.clip = None  # no clip: the loop's steer step
    self.commands = {
      str(k): d["command_values"][i].astype(float)
      for i, k in enumerate(d["command_names"] if "command_names" in d.files else [])
    }
    # (forward m/s, left m/s, turn rad/s) x (min, max) it trained on, if it took any
    # command in a range (then the panel offers a joystick); None: only its named ones.
    self.command_range = (
      d["command_range"].astype(float).reshape(3, 2)
      if "command_range" in d.files
      else None
    )
    self.reset()

  def reset(self, clip=None):
    self.last = np.zeros(16)
    self.past = None  # per input: HISTORY frames, oldest first
    self.ticks = 0  # 50 Hz ticks of the gait clock since the command started
    self.phase = 0.0  # the clock's phase, 0 to 1

  def period(self, command):
    """The clock's period for ``command``: the closest key's (unit vectors, scaled to
    each axis' largest key) or the single ``clock_period``."""
    if self.clock_periods is None:
      return self.clock_period
    keys = np.array(list(self.commands.values()))
    scale = np.abs(keys).max(0) + 1e-9
    k, c = keys / scale, np.asarray(command, float) / scale
    sim = (k @ c) / (np.linalg.norm(k, axis=1) * np.linalg.norm(c) + 1e-12)
    return float(self.clock_periods[int(np.argmax(sim))])

  def clock(self, command):
    """The gait clock's (sin, cos): runs while ``command`` is not zero, else 0. The
    phase advances by one tick over the command's period (``period``)."""
    if not np.any(command):
      self.ticks, self.phase = 0, 0.0
      return np.array([0.0, 1.0])
    if self.ticks == 0:
      self.phase = 0.0
    angle = 2.0 * np.pi * self.phase
    self.phase = (self.phase + 1.0 / 50.0 / self.period(command)) % 1.0
    self.ticks += 1
    return np.array([np.sin(angle), np.cos(angle)])

  def obs(self, command, q, dq, gyro, gravity, clock=None):
    now = [command, gyro, gravity, q - self.default, dq, self.last]
    if self.clock_period is not None:
      now.insert(1, self.clock(command) if clock is None else np.asarray(clock, float))
    if self.past is None:
      self.past = [[x.copy() for _ in range(self.history)] for x in now]
    else:
      for frames, x in zip(self.past, now, strict=True):
        frames.pop(0)
        frames.append(x.copy())
    return np.concatenate([np.concatenate(frames) for frames in self.past])

  def act(self, command, q, dq, gyro, gravity):
    o = self.obs(*(np.asarray(x, dtype=float) for x in (command, q, dq, gyro, gravity)))
    a = self.net(o)
    self.last = a
    return np.clip(self.default + self.scale * a, self.lo, self.hi)
