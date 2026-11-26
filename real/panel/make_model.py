"""The hand as built, for the panel's 3D view: its kinematic tree and its visual meshes.

  uv run python real/panel/make_model.py      # writes real/panel/static/hand.json and hand.bin

The meshes are thinned by vertex clustering (``--cell`` mm) so the browser gets a few
hundred kB. hand.json also holds a few random poses with MuJoCo's body positions and
orientations, which the page checks its own kinematics against (``?selftest``).
"""

import argparse
import json
import os

import mujoco
import numpy as np
from mjlab.entity import Entity

from walking_leap import robot

HERE = os.path.dirname(os.path.abspath(__file__))


def thin(
  vert: np.ndarray, face: np.ndarray, cell: float
) -> tuple[np.ndarray, np.ndarray]:
  """Vertex clustering: one vertex per occupied cell (their mean), faces remapped."""
  key = np.floor(vert / cell).astype(np.int64)
  _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
  inverse = inverse.reshape(-1)
  n = len(first)
  out = np.zeros((n, 3))
  np.add.at(out, inverse, vert)
  out /= np.bincount(inverse, minlength=n)[:, None]
  f = inverse[face]
  keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
  f = f[keep]
  _, once = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
  return out.astype(np.float32), f[np.sort(once)].astype(np.uint32)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--cell", type=float, default=1.0, help="mm")
  a = ap.parse_args()
  m = Entity(robot.get_leap_hand_cfg(as_built=True, feet=False)).spec.compile()
  d = mujoco.MjData(m)
  J = robot.JOINT_NAMES
  joint_index = {m.joint(n).id: i for i, n in enumerate(J)}

  bodies = []
  for b in range(1, m.nbody):
    joint = None
    for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
      if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
        joint = dict(
          index=joint_index[j], axis=m.jnt_axis[j].tolist(), pos=m.jnt_pos[j].tolist()
        )
    bodies.append(
      dict(
        name=m.body(b).name,
        parent=int(m.body_parentid[b]) - 1,
        pos=m.body_pos[b].tolist(),
        quat=m.body_quat[b].tolist(),
        joint=joint,
      )
    )

  blobs, meshes, offset = [], {}, 0
  geoms = []
  for g in range(m.ngeom):
    if m.geom_group[g] not in (1, 2) or m.geom_bodyid[g] == 0:
      continue
    rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    geom = dict(
      body=int(m.geom_bodyid[g]) - 1,
      type=int(m.geom_type[g]),
      size=m.geom_size[g].tolist(),
      pos=m.geom_pos[g].tolist(),
      quat=m.geom_quat[g].tolist(),
      rgba=[round(float(x), 3) for x in rgba],
    )
    k = int(m.geom_dataid[g])
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      if k not in meshes:
        v0, nv = m.mesh_vertadr[k], m.mesh_vertnum[k]
        f0, nf = m.mesh_faceadr[k], m.mesh_facenum[k]
        v, f = thin(m.mesh_vert[v0 : v0 + nv], m.mesh_face[f0 : f0 + nf], a.cell * 1e-3)
        meshes[k] = dict(
          vert=offset, nvert=len(v), face=offset + v.nbytes, nface=len(f)
        )
        blobs += [v.tobytes(), f.tobytes()]
        offset += v.nbytes + f.nbytes
      geom["mesh"] = meshes[k]
    geoms.append(geom)

  # Poses to check the page's kinematics against.
  rng = np.random.default_rng(0)
  lo = np.array([m.jnt_range[m.joint(n).id][0] for n in J])
  hi = np.array([m.jnt_range[m.joint(n).id][1] for n in J])
  checks = []
  for _ in range(3):
    q = rng.uniform(lo, hi)
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    d.qpos[:3], d.qpos[3:7] = rng.normal(size=3) * 0.05, quat
    for i, n in enumerate(J):
      d.qpos[m.joint(n).qposadr[0]] = q[i]
    mujoco.mj_kinematics(m, d)
    checks.append(
      dict(q=q.tolist(), root_pos=d.qpos[:3].tolist(), root_quat=quat.tolist(),
           xpos=d.xpos[1:].tolist(), xquat=d.xquat[1:].tolist())
    )  # fmt: skip

  out = os.path.join(HERE, "static")
  os.makedirs(out, exist_ok=True)
  with open(os.path.join(out, "hand.bin"), "wb") as f:
    f.write(b"".join(blobs))
  with open(os.path.join(out, "hand.json"), "w") as f:
    json.dump(dict(joints=list(J), bodies=bodies, geoms=geoms, checks=checks), f)
  faces = sum(x["nface"] for x in meshes.values())
  print(
    f"{len(bodies)} bodies, {len(geoms)} geoms, {len(meshes)} meshes, {faces} faces, {offset / 1e3:.0f} kB"
  )


if __name__ == "__main__":
  main()
