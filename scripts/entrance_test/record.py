"""Record the grasp sequence to an MP4, and/or export the scene as standalone MJCF.

    python scripts/entrance_test/record.py                      # -> grasp.mp4
    python scripts/entrance_test/record.py --out demo.mp4 --fps 60
    python scripts/entrance_test/record.py \
        --export unitree_robots/g1/_exported_scene.xml      # standalone MJCF

The exported MJCF keeps relative mesh paths (meshdir="meshes/"), so write it
into unitree_robots/g1/ or it will not find the robot meshes. Once exported it
opens in any MuJoCo viewer:  mjpython -m mujoco.viewer --mjcf=<path>

Unlike view.py this needs no GUI and no mjpython, so it works over SSH and is
what the demo video for the report will be built from.
"""
from __future__ import annotations

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

from scene import build
from control import ArmPD, ramp

STROKE = 0.024


def record(out, fps=30, width=960, height=720, camera="scene_cam"):
    import imageio

    spec, m = build()
    d = mujoco.MjData(m)
    nid = lambda t, n: mujoco.mj_name2id(m, t, n)
    palm = nid(mujoco.mjtObj.mjOBJ_BODY, "right_gripper_palm")
    jadr = m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "block_free")]
    fingers = [nid(mujoco.mjtObj.mjOBJ_ACTUATOR, n)
               for n in ("right_finger_l_act", "right_finger_r_act")]
    arm = ArmPD(m)
    hold = np.zeros(arm.n())
    goal = hold.copy(); goal[0] = -0.5

    every = max(1, int(round(1.0 / (fps * m.opt.timestep))))
    renderer = mujoco.Renderer(m, height, width)
    frames = []

    def run(steps, ctrl_fn):
        for i in range(steps):
            ctrl_fn(i)
            mujoco.mj_step(m, d)
            if i % every == 0:
                renderer.update_scene(d, camera=camera)
                frames.append(renderer.render())

    gravity = m.opt.gravity.copy()
    m.opt.gravity[:] = 0
    mujoco.mj_forward(m, d)
    d.qpos[jadr:jadr + 3] = d.xpos[palm] + np.array([0.035, 0.0, 0.0])
    d.qpos[jadr + 3:jadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)

    for a in fingers:
        d.ctrl[a] = 0.0
    run(200, lambda i: arm.apply(d, hold))                    # open
    for a in fingers:
        d.ctrl[a] = STROKE
    run(400, lambda i: arm.apply(d, hold))                    # close
    m.opt.gravity[:] = gravity
    run(400, lambda i: arm.apply(d, hold))                    # gravity on
    run(2500, lambda i: arm.apply(d, ramp(hold, goal, i, 1500)))   # lift
    run(600, lambda i: arm.apply(d, goal))                    # settle

    imageio.mimsave(out, frames, fps=fps, quality=8)
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="grasp.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--export", metavar="PATH",
                    help="write the built scene as a standalone MJCF file and exit")
    args = ap.parse_args()

    if args.export:
        spec, _ = build()
        with open(args.export, "w") as f:
            f.write(spec.to_xml())
        print(f"wrote {args.export}")
        return 0

    n = record(args.out, args.fps, args.width, args.height)
    print(f"wrote {args.out}  ({n} frames @ {args.fps}fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
