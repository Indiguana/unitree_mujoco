"""Regression test: the parallel-jaw gripper holds a block through a lift.

Run: python scripts/entrance_test/test_grasp.py

Verified 2026-08-24 on Apple M5 / MuJoCo 3.12: block rises ~0.09 m with
contacts maintained throughout.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

from scene import build
from control import ArmPD, ramp

STROKE = 0.024
LIFT_RAD = -0.5
RAMP_STEPS = 1500


def run(render_to=None):
    spec, m = build()
    d = mujoco.MjData(m)
    nid = lambda t, n: mujoco.mj_name2id(m, t, n)

    bid = nid(mujoco.mjtObj.mjOBJ_BODY, "block")
    palm = nid(mujoco.mjtObj.mjOBJ_BODY, "right_gripper_palm")
    jadr = m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "block_free")]
    pads = {nid(mujoco.mjtObj.mjOBJ_GEOM, "right_finger_l_pad"),
            nid(mujoco.mjtObj.mjOBJ_GEOM, "right_finger_r_pad")}
    bg = nid(mujoco.mjtObj.mjOBJ_GEOM, "block_geom")
    fingers = [nid(mujoco.mjtObj.mjOBJ_ACTUATOR, n)
               for n in ("right_finger_l_act", "right_finger_r_act")]

    arm = ArmPD(m)
    hold = np.zeros(arm.n())

    def contacts():
        return sum(1 for c in d.contact[:d.ncon]
                   if (c.geom1 in pads and c.geom2 == bg)
                   or (c.geom2 in pads and c.geom1 == bg))

    # Place the block between the fingers with gravity off, so this test
    # isolates the gripper from the (separate) reach-and-align problem.
    gravity = m.opt.gravity.copy()
    m.opt.gravity[:] = 0
    mujoco.mj_forward(m, d)
    d.qpos[jadr:jadr + 3] = d.xpos[palm] + np.array([0.035, 0.0, 0.0])
    d.qpos[jadr + 3:jadr + 7] = [1, 0, 0, 0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    for a in fingers:
        d.ctrl[a] = STROKE
    for _ in range(400):
        arm.apply(d, hold)
        mujoco.mj_step(m, d)

    m.opt.gravity[:] = gravity
    for _ in range(600):
        arm.apply(d, hold)
        mujoco.mj_step(m, d)

    closed_contacts = contacts()
    z0 = float(d.xpos[bid][2])

    goal = hold.copy()
    goal[0] = LIFT_RAD
    for i in range(2500):
        arm.apply(d, ramp(hold, goal, i, RAMP_STEPS))
        for a in fingers:
            d.ctrl[a] = STROKE
        mujoco.mj_step(m, d)

    rise = float(d.xpos[bid][2]) - z0
    held = contacts()

    if render_to:
        from PIL import Image
        r = mujoco.Renderer(m, 600, 800)
        r.update_scene(d, camera="scene_cam")
        Image.fromarray(r.render()).save(render_to)

    return closed_contacts, rise, held


def main():
    c0, rise, held = run(render_to=os.environ.get("GRASP_PNG"))
    print(f"contacts after close : {c0}")
    print(f"block rise on lift   : {rise:+.3f} m")
    print(f"contacts after lift  : {held}")
    ok = c0 > 0 and rise > 0.03 and held > 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
