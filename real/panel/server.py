"""Walking Leap's control panel: a web page that drives the hand through the real loop.

  real/panel/run                          # the real hand (its driver's env; Mathias there)
  real/panel/run --fake                   # the driver's fake motors and a level IMU
  uv run python real/panel/server.py --twin   # a simulated hand (the repo's env)

Then open http://localhost:8765. From a laptop: ``ssh -L 8765:localhost:8765 <hand PC>``
first. It listens on this computer only (``--host`` to change it).

The page sends commands; one session thread runs the loop (wlc.Control) with all its
guards, and plays clips one after the other: stand up, the moves (W S A D Q E), the
emotes, lie down. Between clips it holds the pose, still checking. STOP (or Esc) lies
the hand down, then torque off. Everything is logged as with ``wl`` (outputs/real/logs).

With a velocity policy (``kind`` "velocity", a gait student) the same keys walk instead:
held, the policy gets that command every tick (Control.steer, as ``wl play ... steer``);
released, or the page silent for WALK_ALIVE_S, it stands still. A clip asked for ends the
walk first (a second standing still, then back to the stance).
"""

import argparse
import collections
import json
import mimetypes
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import wlc  # noqa: E402
from policy import MotionPolicy, VelocityPolicy  # noqa: E402

STATIC = os.path.join(HERE, "static")
MOVES = {  # key: clip (real/clips/)
  "w": "step_forward",
  "s": "step_back",
  "a": "step_left",
  "d": "step_right",
  "q": "turn_left",
  "e": "turn_right",
}
EMOTES = (  # clip, icon: the ones that start and end standing
  ("bow", "🙇"),
  ("pushups", "💪"),
  ("sway", "🌊"),
  ("taps", "🥁"),
  ("hello_start", "👋"),
  ("crawl_dead", "🧟"),
  ("crawl_happy", "🎉"),
  ("crawl", "🐾"),
)
TITLES = {
  "pushups": "Push-ups",
  "sway": "Sway",
  "taps": "Taps",
  "hello_start": "Hello",
  "crawl": "Crawl ×3",
}
QUEUE_MAX = 4
WALK = {  # key: a velocity policy's command (walking_leap.clips.GAIT_KEYS)
  "w": "forward",
  "s": "back",
  "a": "left",
  "d": "right",
  "q": "turn_left",
  "e": "turn_right",
}
WALK_ALIVE_S = 0.4  # a held key is repeated by the page; silent this long: stand still


def policy_kind(path):
  with np.load(path) as d:
    return str(d["kind"]) if "kind" in d.files else "motion"


def load_policy(path):
  return VelocityPolicy(path) if policy_kind(path) == "velocity" else MotionPolicy(path)


def covers(policy, name):
  """A motion policy that plays along this clip (a velocity policy plays none)."""
  return policy is not None and policy.clip is not None and policy.covers(name)


class PageKeys:
  """What Control.steer reads, from the page: the held key's command when it changes,
  "stand" when released or when the page went silent, "quit" to end the walk."""

  HELP = "hold W A S D Q E to walk, release to stand still"

  def __init__(self, session):
    self.s, self.last = session, "stand"

  def read(self):
    s = self.s
    if s.walk_quit:
      s.walk_cmd = "stand"
      return "quit"
    fresh = time.monotonic() - s.walk_seen < WALK_ALIVE_S
    want = (s.walk_key if fresh else None) or "stand"
    if want == self.last:
      return None
    self.last = s.walk_cmd = want
    return want


def r(x, n=4):
  return None if x is None else np.round(np.asarray(x, float), n).tolist()


def tilt_of(g):
  return float(np.degrees(np.arccos(np.clip(-g[2], -1, 1))))


class Session:
  """One connection to the hand: torque on from start() to the end of run()."""

  def __init__(self, mode, gains, policies):
    self.mode, self.gains, self.policies = mode, gains, policies
    self.cmds = queue.Queue()
    self.lock = threading.Lock()
    self.events = collections.deque(maxlen=200)
    self.event_id = 0
    self.state = "off"  # off, starting, on, stopping
    self.where, self.clip, self.queue = "lying", None, []
    self.policy_name, self.live, self.policy = None, False, None
    self.ctl = self.thread = None
    self.snap = {}
    self.est = np.zeros(
      3
    )  # supposed palm pose: x, y (m), heading (rad), from the clips' plans
    self.base = np.zeros(3)
    self.yaw_gyro = 0.0
    self.temps = [np.nan, np.nan]
    self.sim, self.sim0 = (
      None,
      None,
    )  # the twin's true palm pose, and where the map started
    self.map_epoch = 0  # the map starts again where the hand stood up
    # Walking with a velocity policy: the held command (from the page) and when it came.
    self.walking, self.walk_pending, self.walk_quit = False, False, False
    self.walk_key, self.walk_seen, self.walk_cmd = None, 0.0, "stand"

  # --------------------------------------------------------------- from the page
  def event(self, text):
    with self.lock:
      self.event_id += 1
      self.events.append((self.event_id, round(time.time(), 1), text))

  def start(self):
    if self.state != "off":
      return False, "already connected"
    self.state, self.where, self.clip, self.queue = "starting", "lying", None, []
    self.thread = threading.Thread(target=self.run, daemon=True)
    self.thread.start()
    return True, "connecting"

  def stop(self, why="STOP button"):
    """Lie down (if standing), then torque off."""
    self.queue.clear()
    if self.ctl is not None and self.state == "on":
      self.ctl.stop_asked = why
      self.state = "stopping"
      return True, "stopping: lying down, then torque off"
    return False, "not connected"

  def ask(self, kind, name=None):
    if self.state != "on":
      return False, "connect first"
    if self.walking:
      self.walk_quit = True  # the walk ends first (a second still, back to the stance)
    if kind == "end":
      self.queue.clear()
      self.cmds.put(("end", None))
      return True, "lying down, then torque off"
    if len(self.queue) >= QUEUE_MAX:
      return False, f"{QUEUE_MAX} already waiting"
    self.queue.append(name)
    self.cmds.put((kind, name))
    return True, f"queued {name}"

  def set_policy(self, name, live):
    if name and name not in self.policies:
      return False, f"no policy {name}"
    if self.walking or self.walk_pending:
      return False, "stop walking first"
    self.policy_name, self.live = name or None, bool(live and name)
    self.policy = load_policy(self.policies[name]) if name else None
    if self.ctl is not None:
      self.ctl.policy, self.ctl.live = self.policy, self.live
    what = (
      "no policy (scripts)"
      if not name
      else f"{name}, {'LIVE' if self.live else 'shadow'}"
    )
    self.event(f"policy: {what}")
    return True, what

  def walk(self, command, start=False):
    """The page's held key as a command name, or None (released); repeated while held.
    Only a key going down (``start``) begins a walk; the repeats only keep it going."""
    if self.state != "on":
      return False, "connect first"
    if not isinstance(self.policy, VelocityPolicy):
      return False, "walking needs a velocity policy"
    if command is not None and command not in self.policy.commands:
      return False, f"{self.policy_name} has no {command}"
    self.walk_key, self.walk_seen = command, time.monotonic()
    if command is not None and start and not (self.walking or self.walk_pending):
      if self.queue:
        return False, "the queue first (or clear it)"
      self.walk_pending = True
      self.cmds.put(("walk", None))
    return True, command or "stand"

  def end_walk(self):
    if not (self.walking or self.walk_pending):
      return False, "not walking"
    self.walk_quit, self.walk_key = True, None
    return True, "standing still, then back to the stance"

  # --------------------------------------------------------------- the session thread
  def devices(self):
    if self.mode == "twin":
      import twin

      imu, hand = twin.devices()
      self.sim = hand.s
      return imu, hand
    if self.mode == "fake":
      return wlc.FakeImu(), wlc.Hand(self.gains, True, wlc.Clip("stand_up").q[0])
    from tm171 import TM171

    return TM171(), wlc.Hand(self.gains)

  def tick(self, t):
    """Every 50 Hz tick, from the loop."""
    ctl, clip = self.ctl, self.clip
    try:
      gz = float(ctl.imu.gyro_mean(0.01)[2])
    except Exception:
      gz = 0.0
    self.yaw_gyro += gz / wlc.HZ
    if clip is not None and clip.palm is not None and t["frame"] >= 0:
      p = clip.palm[min(t["frame"], len(clip.palm) - 1)]
      c, s = np.cos(self.base[2]), np.sin(self.base[2])
      self.est = self.base + np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])
    elif self.walking and self.live:  # the command's velocity, integrated: a plan too
      v = self.policy.commands.get(self.walk_cmd, np.zeros(3))
      c, s = np.cos(self.est[2]), np.sin(self.est[2])
      self.est = (
        self.est + np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]) / wlc.HZ
      )
    pq, cq = t["policy_q"], t["clip_q"]
    snap = dict(
      q=r(t["q"]),
      sent=r(t["sent"]),
      clip_q=r(cq),
      policy_q=r(pq),
      g=r(t["g"]),
      pred=r(t["pred"]),
      cur=r(t["cur"], 0),
      late=round(float(t["late"]), 1),
      frame=int(t["frame"]),
      tilt=round(tilt_of(t["g"]), 1),
      off=None if t["pred"] is None
      else round(float(np.degrees(np.arccos(np.clip(np.dot(t["g"], t["pred"]), -1, 1)))), 1),
      pol=None if pq is None else round(float(np.abs(np.asarray(pq) - cq).mean()), 3),
      est=[round(self.est[0] * 1e3, 1), round(self.est[1] * 1e3, 1), round(np.degrees(self.est[2]), 1)],
      yaw_gyro=round(float(np.degrees(self.yaw_gyro)), 1),
      sim=None,
    )  # fmt: skip
    if (
      self.sim is not None
    ):  # the twin: where the simulated palm really is, from the map's start
      now = np.array(self.sim.pose())
      if self.sim0 is None:
        self.sim0 = now
      c, s, d = np.cos(self.sim0[2]), np.sin(self.sim0[2]), now - self.sim0
      x, y = c * d[0] + s * d[1], -s * d[0] + c * d[1]
      snap["sim"] = [
        round(x * 1e3, 1),
        round(y * 1e3, 1),
        round(float(np.degrees(d[2])), 1),
      ]
    with self.lock:
      self.snap = snap

  def play(self, name):
    ctl = self.ctl
    clip = wlc.Clip(name)
    if clip.start != ctl.where:
      self.event(f"{name}: it starts {clip.start}, the hand is {ctl.where}: skipped")
      return
    self.base = self.est.copy()
    self.clip = clip
    on = covers(self.policy, name)
    self.event(
      f"{name}: playing{' (policy ' + ('LIVE' if self.live else 'shadow') + ')' if on else ''}"
    )
    try:
      ctl.play(
        clip
      )  # the map follows the plan frame by frame (tick), so a cut clip stays cut
    finally:
      self.clip = None
    self.where = ctl.where
    if name == "stand_up":
      self.reset_map()

  def walk_loop(self, ctl):
    """Control.steer with the page's keys, from the stance and back to it."""
    self.walk_pending = False
    if ctl.where != "stand":
      self.event(f"walking starts standing, the hand is {ctl.where}: skipped")
      return
    if not isinstance(self.policy, VelocityPolicy):
      self.event("walking needs a velocity policy: skipped")
      return
    self.walking, self.walk_cmd = True, "stand"
    try:
      ctl.steer(PageKeys(self), ctl.q.copy())
    finally:
      self.walking, self.walk_quit, self.walk_key, self.walk_cmd = (
        False,
        False,
        None,
        "stand",
      )
    self.where = ctl.where

  def reset_map(self):
    self.est[:], self.base[:], self.yaw_gyro, self.sim0 = 0.0, 0.0, 0.0, None
    self.map_epoch += 1

  def run(self):
    rec = None
    try:
      imu, hand = self.devices()
      with imu:
        imu.wait()
        if self.mode == "real":
          time.sleep(1.2)
          if (imu.quality or 0) < 4:
            raise wlc.Abort(f"IMU quality {imu.quality}: leave it still half a minute")
          if imu.latest.tilt() > 10:
            raise wlc.Abort(f"not lying level ({imu.latest.tilt():.0f} deg)")
        with hand:
          meta = dict(
            when=time.strftime("%Y-%m-%d %H:%M:%S"), panel=True, mode=self.mode,
            gains=self.gains, joint_names=wlc.JOINT_NAMES, current_limit_ma=350,
            columns=dict(motor=wlc.Recorder.MOTOR, imu=wlc.Recorder.IMU, cmd=wlc.Recorder.CMD),
          )  # fmt: skip
          rec = wlc.Recorder(hand, imu, f"panel_{self.mode}", meta)
          rec.event = self._recorder_event(rec.event)
          ctl = wlc.Control(
            hand, imu, rec, max_off=180.0 if self.mode == "fake" else 15.0
          )
          ctl.where, ctl.keep_policy = "lying", True
          ctl.policy, ctl.live = self.policy, self.live
          ctl.on_tick = self.tick
          self.ctl = ctl
          self.loop(ctl, hand)
    except Exception as e:  # noqa: BLE001 - whatever it was, the page must hear it
      self.event(f"session ended: {type(e).__name__}: {e}")
    finally:
      if rec is not None:
        try:
          rec.close()
        except Exception as e:  # noqa: BLE001
          self.event(f"log not written: {e}")
      self.ctl, self.clip, self.queue, self.state = None, None, [], "off"
      self.walking, self.walk_pending, self.walk_quit, self.walk_key = (
        False,
        False,
        False,
        None,
      )
      self.event("torque off")

  def _recorder_event(self, original):
    def event(text):
      original(text)
      self.event(text)

    return event

  def loop(self, ctl, hand):
    try:
      ctl.start()
      t = hand.temps()
      self.temps = [float(np.nanmin(t)), float(np.nanmax(t))]
      self.state = "on"
      self.event(f"torque on, lying, {self.temps[0]:.0f}-{self.temps[1]:.0f} C")
      last_temps = time.monotonic()
      while True:
        try:
          kind, name = self.cmds.get_nowait()
        except queue.Empty:
          ctl.step(ctl.q)  # hold, still checking
          if time.monotonic() - last_temps > 10:
            t = hand.temps()
            self.temps, last_temps = (
              [float(np.nanmin(t)), float(np.nanmax(t))],
              time.monotonic(),
            )
            if np.nanmax(t) > 60:
              raise wlc.SoftStop(f"a motor is at {np.nanmax(t):.0f} C") from None
          continue
        if kind == "walk":
          self.walk_loop(ctl)
          continue
        if name in self.queue:
          self.queue.remove(name)
        if kind == "end":
          if ctl.where == "stand":
            self.play("lie_down")
          return
        self.play(name)
        t = hand.temps()
        self.temps, last_temps = (
          [float(np.nanmin(t)), float(np.nanmax(t))],
          time.monotonic(),
        )
    except wlc.SoftStop as e:
      self.event(f"stopping: {e}")
      self.queue.clear()
      if ctl.where in ("stand", "moving"):
        try:
          wlc.soft_down(ctl)
        except (wlc.Abort, wlc.SoftStop) as e2:
          self.event(f"ABORT on the way down: {e2} -> hold, then torque off")
          hand.hold()
          time.sleep(0.3)
    except wlc.Abort as e:
      self.event(f"ABORT: {e} -> hold, then torque off")
      hand.hold()
      time.sleep(0.3)

  # --------------------------------------------------------------- what the page reads
  def status(self):
    with self.lock:
      s = dict(self.snap)
    clip = self.clip
    s.update(
      mode=self.mode,
      state=self.state,
      where=self.ctl.where if self.ctl is not None else self.where,
      clip=None if clip is None else clip.name,
      nframes=None if clip is None else len(clip.q),
      queue=list(self.queue),
      policy=self.policy_name,
      live=self.live,
      policy_on=bool(self.walking or (clip is not None and covers(self.policy, clip.name))),
      policy_kind=None if self.policy is None else ("velocity" if isinstance(self.policy, VelocityPolicy) else "motion"),
      walking=self.walking,
      walk_cmd=self.walk_cmd if self.walking else None,
      temps=[None if np.isnan(x) else round(x) for x in self.temps],
      event_id=self.event_id,
      map_epoch=self.map_epoch,
    )  # fmt: skip
    return s

  def events_after(self, k):
    with self.lock:
      return [e for e in self.events if e[0] > k]


def catalog(policies):
  """What the page offers: the moves, the emotes, the policies and what each one covers."""
  names = wlc.clip_names()
  clips = {}
  for name in set(MOVES.values()) | {e for e, _ in EMOTES} | {"stand_up", "lie_down"}:
    if name in names:
      c = wlc.Clip(name)
      title = (
        c.meta.get("title") or TITLES.get(name) or name.replace("_", " ").capitalize()
      )
      travel = (
        None
        if c.palm is None
        else [
          round(c.palm[-1, 0] * 1e3),
          round(c.palm[-1, 1] * 1e3),
          round(float(np.degrees(c.palm[-1, 2]))),
        ]
      )
      clips[name] = dict(title=title, seconds=round(len(c.q) / wlc.HZ, 1), note=c.meta.get("note", ""), travel=travel)  # fmt: skip
  pols = {}
  for name, path in policies.items():
    kind = policy_kind(path)
    if kind == "velocity":  # walks on the keys: each command's velocity, mm/s and deg/s
      p = VelocityPolicy(path)
      pols[name] = dict(
        kind=kind, inputs=int(p.mean.size), imu=True, covers=[],
        commands={
          c: [round(v[0] * 1e3, 1), round(v[1] * 1e3, 1), round(float(np.degrees(v[2])), 1)]
          for c, v in p.commands.items()
        },
      )  # fmt: skip
      continue
    if kind != "motion":
      print(f"{name}: a {kind} policy, not offered on the panel")
      continue
    p = MotionPolicy(path)
    pols[name] = dict(
      kind=kind, inputs=int(p.mean.size), imu=p.imu,
      covers=sorted(n for n in names if p.covers(n)),
    )  # fmt: skip
  return dict(
    moves={k: v for k, v in MOVES.items() if v in clips},
    emotes=[dict(name=n, icon=i) for n, i in EMOTES if n in clips],
    clips=clips,
    policies=pols,
  )


def find_policies():
  out = {}
  if os.path.isdir(wlc.POLICIES):
    for f in sorted(os.listdir(wlc.POLICIES)):
      if f.endswith(".npz"):
        out[f[:-4]] = os.path.join(wlc.POLICIES, f)
  return out


def handler(session, info):
  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
      pass

    def send(self, code, body, kind="application/json"):
      data = body if isinstance(body, bytes) else json.dumps(body).encode()
      self.send_response(code)
      self.send_header("Content-Type", kind)
      self.send_header("Content-Length", str(len(data)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(data)

    def do_GET(self):
      path = self.path.split("?")[0]
      if path == "/api/config":
        return self.send(200, dict(info, policies=info["policies"]))
      if path == "/api/state":
        return self.send(200, dict(session.status(), events=session.events_after(0)))
      if path == "/api/stream":
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen = 0
        try:
          while True:
            s = session.status()
            new = session.events_after(seen)
            if new:
              seen = new[-1][0]
            s["events"] = new
            self.wfile.write(f"data: {json.dumps(s)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
          return
      name = "index.html" if path == "/" else path.lstrip("/")
      full = os.path.realpath(os.path.join(STATIC, name))
      if not full.startswith(os.path.realpath(STATIC)) or not os.path.isfile(full):
        return self.send(404, {"error": "not found"})
      kind = mimetypes.guess_type(full)[0] or "application/octet-stream"
      with open(full, "rb") as f:
        return self.send(200, f.read(), kind)

    def do_POST(self):
      n = int(self.headers.get("Content-Length") or 0)
      try:
        body = json.loads(self.rfile.read(n) or b"{}")
      except json.JSONDecodeError:
        body = {}
      path = self.path.split("?")[0]
      if path == "/api/connect":
        ok, msg = session.start()
      elif path == "/api/stop":
        ok, msg = session.stop()
      elif path == "/api/end":
        ok, msg = session.ask("end")
      elif path == "/api/stand_up":
        ok, msg = session.ask("clip", "stand_up")
      elif path == "/api/lie_down":
        ok, msg = session.ask("clip", "lie_down")
      elif path == "/api/move":
        key = str(body.get("key", "")).lower()
        ok, msg = (
          session.ask("clip", MOVES[key])
          if key in info["moves"]
          else (False, "no such move")
        )
      elif path == "/api/emote":
        name = body.get("name")
        known = {e["name"] for e in info["emotes"]}
        ok, msg = (
          session.ask("clip", name) if name in known else (False, "no such emote")
        )
      elif path == "/api/walk":  # repeated while a key is held: refusals not logged
        if body.get("end"):
          ok, msg = session.end_walk()
        else:
          key = body.get("key")
          command = WALK.get(str(key).lower()) if key else None
          ok, msg = session.walk(command, bool(body.get("start")))
        return self.send(200, {"ok": ok, "msg": msg})
      elif path == "/api/policy":
        ok, msg = session.set_policy(body.get("name"), body.get("live", False))
      elif path == "/api/clear":
        session.queue.clear()
        while not session.cmds.empty():
          try:
            session.cmds.get_nowait()
          except queue.Empty:
            break
        ok, msg = True, "queue cleared"
      elif path == "/api/reset_map":
        session.reset_map()
        ok, msg = True, "map reset"
      else:
        return self.send(404, {"ok": False, "msg": "no such command"})
      if not ok:
        session.event(f"refused: {msg}")
      self.send(200, {"ok": ok, "msg": msg})

  return Handler


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  mode = ap.add_mutually_exclusive_group()
  mode.add_argument("--fake", action="store_true", help="the driver's fake motors")
  mode.add_argument(
    "--twin", action="store_true", help="a simulated hand (needs mujoco)"
  )
  ap.add_argument("--port", type=int, default=8765)
  ap.add_argument("--host", default="127.0.0.1")
  ap.add_argument("--gains", default="1200,0,600")
  a = ap.parse_args()
  kind = "twin" if a.twin else "fake" if a.fake else "real"
  gains = tuple(int(x) for x in a.gains.split(","))
  policies = find_policies()
  info = dict(
    catalog(policies),
    mode=kind,
    gains=gains,
    rest_q=r(wlc.Clip("stand_up").q[0]),
    walk=WALK,
  )
  session = Session(kind, gains, policies)
  if kind == "real":
    from leap_hand.hand import raise_on_signals

    raise_on_signals()
  server = ThreadingHTTPServer((a.host, a.port), handler(session, info))
  server.daemon_threads = True
  print(
    f"Walking Leap panel ({kind}): http://localhost:{a.port}  (Ctrl-C: lies down, torque off)"
  )
  try:
    server.serve_forever()
  except (KeyboardInterrupt, SystemExit):
    print("closing: lying down if standing, then torque off")
    session.stop("the panel is closing")
    if session.thread is not None:
      session.thread.join(40)
  return 0


if __name__ == "__main__":
  sys.exit(main())
