"""Parallel-jaw gripper for the Unitree G1.

The stock G1 MJCF has no actuated end-effector: `left/right_rubber_hand` are
decorative meshes with contype=0/conaffinity=0, so they do not even collide.
This module adds a two-finger gripper to a wrist link via MuJoCo's spec API,
which keeps the upstream robot XML untouched.

Fingers extend along the wrist's +X (the arm's reach direction) and pinch
along +/-Y. Each finger is a slide joint driven by a position actuator.
"""
from __future__ import annotations

import mujoco

# Geometry (metres). Half-extents, per MuJoCo box convention.
PALM_OFFSET_X = 0.055
FINGER_HALF = (0.035, 0.005, 0.012)
FINGER_OPEN_Y = 0.030
STROKE = 0.024
KP = 60.0
KV = 3.0
FINGER_FRICTION = (1.6, 0.05, 0.001)


def add_parallel_gripper(spec, wrist_body: str, prefix: str):
    """Attach a parallel-jaw gripper to `wrist_body`. Returns joint names."""
    wrist = spec.body(wrist_body)

    palm = wrist.add_body()
    palm.name = f"{prefix}_gripper_palm"
    palm.pos = [PALM_OFFSET_X, 0, 0]

    joints = []
    for side, sign in (("l", 1.0), ("r", -1.0)):
        finger = palm.add_body()
        finger.name = f"{prefix}_finger_{side}"
        finger.pos = [0, sign * FINGER_OPEN_Y, 0]

        j = finger.add_joint()
        j.name = f"{prefix}_finger_{side}_slide"
        j.type = mujoco.mjtJoint.mjJNT_SLIDE
        j.axis = [0, -sign, 0]          # +ctrl closes, both sides
        j.range = [0.0, STROKE]
        j.damping = [1.0, 0.0, 0.0]   # per-DOF vector, not a scalar
        joints.append(j.name)

        g = finger.add_geom()
        g.name = f"{prefix}_finger_{side}_pad"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(FINGER_HALF)
        g.pos = [FINGER_HALF[0], 0, 0]
        g.friction = list(FINGER_FRICTION)
        g.rgba = [0.15, 0.15, 0.18, 1.0]

        a = spec.add_actuator()
        a.name = f"{prefix}_finger_{side}_act"
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.target = j.name
        a.ctrlrange = [0.0, STROKE]
        a.set_to_position(kp=KP, kv=KV)

    return joints
