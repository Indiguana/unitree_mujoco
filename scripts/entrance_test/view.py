"""Watch the scene in MuJoCo's interactive viewer.

macOS note: the interactive viewer must run under `mjpython`, not `python` --
Cocoa requires the UI on the process main thread. mjpython ships with the
mujoco pip package.

    mjpython scripts/entrance_test/view.py           # run the grasp on a loop
    mjpython scripts/entrance_test/view.py --static  # just look around

Mouse: drag to orbit, right-drag to pan, scroll to zoom.
Double-click a body then Ctrl-drag to shove it around.
"""
from __future__ import annotations

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco
import mujoco.viewer

from scene import build
from control import ArmPD, ramp

STROKE = 0.024


def grasp_sequence(m, d, arm, fingers, jadr, palm):
    """Yields once per physics step, looping open -> close -> lift -> reset."""
    hold = np.zeros(arm.n())
    goal = hold.copy()
    goal[0] = -0.5
    while True:
        # reset: block back between the fingers, gravity briefly off
        g = m.opt.gravity.copy()
        m.opt.gravity[:] = 0
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        d.qpos[jadr:jadr + 3] = d.xpos[palm] + np.array([0.035, 0.0, 0.0])
        d.qpos[jadr + 3:jadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)

        for a in fingers:
            d.ctrl[a] = 0.0
        for _ in range(150):
            arm.apply(d, hold); yield
        for a in fingers:
            d.ctrl[a] = STROKE
        for _ in range(400):
            arm.apply(d, hold); yield

        m.opt.gravity[:] = g
        for _ in range(400):
            arm.apply(d, hold); yield
        for i in range(2500):
            arm.apply(d, ramp(hold, goal, i, 1500))
            for a in fingers:
                d.ctrl[a] = STROKE
            yield
        for _ in range(600):
            arm.apply(d, goal); yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true", help="no motion, just look")
    args = ap.parse_args()

    spec, m = build()
    d = mujoco.MjData(m)
    nid = lambda t, n: mujoco.mj_name2id(m, t, n)
    palm = nid(mujoco.mjtObj.mjOBJ_BODY, "right_gripper_palm")
    jadr = m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "block_free")]
    fingers = [nid(mujoco.mjtObj.mjOBJ_ACTUATOR, n)
               for n in ("right_finger_l_act", "right_finger_r_act")]
    arm = ArmPD(m)

    seq = None if args.static else grasp_sequence(m, d, arm, fingers, jadr, palm)
    if args.static:
        mujoco.mj_forward(m, d)

    dt = m.opt.timestep
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.perf_counter()
            if seq is not None:
                next(seq)
                mujoco.mj_step(m, d)
            v.sync()
            lag = dt - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
