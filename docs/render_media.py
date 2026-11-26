"""Render the README images and GIFs of the LEAP Hand base.

Plain MuJoCo on the CPU, using the model mjlab builds for the robot, the
environment's physics options and its action pipeline
(target = clip(nominal + ACTION_SCALE * action, joint limits)). No policy is
involved. Run from the repository root:

  uv run python docs/render_media.py
"""

from pathlib import Path

import mediapy
import mujoco
import numpy as np
from mjlab.entity import Entity

from walking_leap import robot
from walking_leap.clips import CLIPS_DIR
from walking_leap.env_cfg import ACTION_SCALE, walking_leap_flat_env_cfg

OUT = Path(__file__).parent / "media"
FPS = 15
GIF_SIZE = (240, 320)  # (height, width)


def build_model(**hand) -> mujoco.MjModel:
  spec = Entity(robot.get_leap_hand_cfg(**hand)).spec
  spec.add_texture(
    name="sky",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=(0.85, 0.9, 0.95),
    rgb2=(0.55, 0.65, 0.75),
    width=256,
    height=256,
  )
  spec.add_texture(
    name="grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(0.82, 0.82, 0.8),
    rgb2=(0.72, 0.72, 0.7),
    mark=mujoco.mjtMark.mjMARK_EDGE,
    markrgb=(0.6, 0.6, 0.6),
    width=300,
    height=300,
  )
  floor_mat = spec.add_material(name="floor", texrepeat=(20, 20), texuniform=True)
  floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01), material="floor"
  )
  spec.worldbody.add_light(pos=(0.3, -0.3, 1.0), dir=(-0.3, 0.3, -1.0), castshadow=True)
  # Visual-only site to draw the palm (robot base) frame.
  spec.body(robot.ROOT_BODY).add_site(name="palm_frame", size=(0.001, 0, 0))
  m = spec.compile()
  walking_leap_flat_env_cfg().sim.mujoco.apply(m)
  # Palm frame axes ~6 cm long (lengths scale with stat.meansize).
  m.vis.scale.framelength = 0.06 / m.stat.meansize
  m.vis.scale.framewidth = 0.004 / m.stat.meansize
  m.vis.scale.contactwidth = 0.15
  m.vis.scale.contactheight = 0.05
  return m


def nominal_data(m: mujoco.MjModel, lift: float = 0.0) -> mujoco.MjData:
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, m.key("init_state").id)
  d.qpos[2] += lift
  mujoco.mj_forward(m, d)
  return d


def camera(d: mujoco.MjData, azimuth: float, elevation: float, distance: float = 0.45):
  cam = mujoco.MjvCamera()
  cam.lookat[:] = d.qpos[:3] - np.array([0.03, 0.02, 0.07])
  cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
  return cam


def options(frames: bool = False, contacts: bool = False) -> mujoco.MjvOption:
  opt = mujoco.MjvOption()
  if frames:
    opt.frame = mujoco.mjtFrame.mjFRAME_SITE
  opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contacts
  return opt


def render_pose(m: mujoco.MjModel, name: str, views: tuple, **opt_kwargs) -> None:
  """Views of the initial pose after settling 1 s on the feet."""
  d = nominal_data(m)
  for _ in range(500):
    mujoco.mj_step(m, d)
  with mujoco.Renderer(m, height=420, width=560) as r:
    images = []
    for az, el in views:
      r.update_scene(d, camera(d, az, el), options(**opt_kwargs))
      images.append(r.render())
  mediapy.write_image(OUT / name, np.concatenate(images, axis=1))


def render_nominal_pose(m: mujoco.MjModel) -> None:
  """Three views of the nominal pose; palm frame and contacts."""
  views = ((135.0, -20.0), (90.0, -5.0), (180.0, -5.0))
  render_pose(m, "nominal_pose.png", views, frames=True, contacts=True)


def render_as_built(m: mujoco.MjModel) -> None:
  """The hand with its feet and IMU, folded-thumb stance."""
  render_pose(m, "as_built.png", ((135.0, -20.0), (90.0, -5.0), (150.0, -60.0)))


def render_hello(m: mujoco.MjModel) -> None:
  """The `hello` clip's joints as position targets, in physics: no policy."""
  fps = 10
  clip = np.load(CLIPS_DIR / "hello.npz")
  d = nominal_data(m)
  d.qpos[:] = np.r_[
    clip["body_pos_w"][0, 0], clip["body_quat_w"][0, 0], clip["joint_pos"][0]
  ]
  mujoco.mj_forward(m, d)
  view = camera(d, 135.0, -14.0, distance=0.5)  # Fixed: the palm's moves are the point.
  frames = []
  with mujoco.Renderer(m, *GIF_SIZE) as r:
    for i, target in enumerate(clip["joint_pos"]):
      d.ctrl[:] = target
      for _ in range(10):
        mujoco.mj_step(m, d)
      if i % (int(clip["fps"][0]) // fps) == 0:
        r.update_scene(d, view)
        frames.append(r.render())
  mediapy.write_video(OUT / "hello.gif", frames, fps=fps, codec="gif")


def record(m, d, seconds, controller=None, cam_args=(135.0, -20.0), **opt_kwargs):
  frames = []
  steps_per_frame = round(1.0 / (FPS * m.opt.timestep))
  with mujoco.Renderer(m, *GIF_SIZE) as r:
    for k in range(round(seconds / m.opt.timestep)):
      if controller is not None and k % 10 == 0:  # 50 Hz policy rate
        controller(d, k * m.opt.timestep)
      mujoco.mj_step(m, d)
      if k % steps_per_frame == 0:
        r.update_scene(d, camera(d, *cam_args), options(**opt_kwargs))
        frames.append(r.render())
  return frames


def render_standing(m: mujoco.MjModel) -> None:
  """Zero action: reset 5 mm above the ground (max reset drop), then hold."""
  d = nominal_data(m, lift=0.005)
  frames = record(m, d, 3.0, contacts=True)
  mediapy.write_video(OUT / "zero_action.gif", frames, fps=FPS, codec="gif")


def render_joint_sweep(m: mujoco.MjModel) -> None:
  """Gravity off, hand floating: each digit sweeps its four joints in turn."""
  gravity = m.opt.gravity.copy()
  m.opt.gravity[:] = 0.0
  d = nominal_data(m, lift=0.1)
  nominal = d.ctrl.copy()
  lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
  period = 1.25

  def controller(d, t):
    digit = min(int(t // period), 3)
    phase = np.sin(2 * np.pi * (t % period) / period)
    target = nominal.copy()
    target[4 * digit : 4 * digit + 4] += 0.5 * phase
    d.ctrl[:] = np.clip(target, lo, hi)

  frames = record(m, d, 4 * period, controller, cam_args=(150.0, -25.0))
  m.opt.gravity[:] = gravity
  mediapy.write_video(OUT / "joint_sweep.gif", frames, fps=FPS, codec="gif")


def render_random_actions(m: mujoco.MjModel) -> None:
  """Uniform random actions in [-1, 1] through the env's action mapping."""
  d = nominal_data(m)
  nominal = d.ctrl.copy()
  lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
  rng = np.random.default_rng(0)

  def controller(d, t):
    action = rng.uniform(-1.0, 1.0, size=m.nu)
    d.ctrl[:] = np.clip(nominal + ACTION_SCALE * action, lo, hi)

  frames = record(m, d, 4.0, controller, contacts=True)
  mediapy.write_video(OUT / "random_actions.gif", frames, fps=FPS, codec="gif")


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  m = build_model()
  render_nominal_pose(m)
  render_standing(m)
  render_joint_sweep(m)
  render_random_actions(m)
  as_built = build_model(
    as_built=True, folded_thumb=True, effort_limit=robot.API_EFFORT_LIMIT
  )
  render_as_built(as_built)
  render_hello(as_built)
  for f in sorted(OUT.iterdir()):
    print(f"{f.relative_to(OUT.parent.parent)}: {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
  main()
