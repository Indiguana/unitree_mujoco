"""Reproducible environment smoke test — MuJoCo + G1 on macOS/Apple Silicon.

Run: python scripts/entrance_test/check_env.py
Verifies the model compiles, physics steps, and offscreen rendering works
(the last one matters: it is the camera path every VLA observation depends on).
"""
import sys
import mujoco
import numpy as np
from PIL import Image

SCENE = "unitree_robots/g1/scene_29dof.xml"


def main() -> int:
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    print(f"scene      : {SCENE}")
    print(f"bodies     : {m.nbody}")
    print(f"dof (nv)   : {m.nv}   # 29 joints + 6 free-base DOF")
    print(f"actuators  : {m.nu}")
    print(f"cameras    : {m.ncam}   # none yet -- must be added for VLA observations")
    print(f"timestep   : {m.opt.timestep}")

    h0 = float(d.qpos[2])
    for _ in range(500):
        mujoco.mj_step(m, d)
    h1 = float(d.qpos[2])
    print(f"pelvis z   : {h0:.3f} -> {h1:.3f} after 500 steps with zero control")

    r = mujoco.Renderer(m, height=480, width=640)
    r.update_scene(d)
    Image.fromarray(r.render()).save("/tmp/g1_check.png")
    print("render     : OK -> /tmp/g1_check.png")
    print("\nenvironment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
