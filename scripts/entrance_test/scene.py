"""Scene construction for the entrance-test manipulation tasks.

Two deliberate departures from the stock G1 scene, both recorded in REPORT.md:

1. The pelvis free joint is removed. The stock model has nv=35 (29 joints +
   6 free-base DOF), so the robot must actively balance. These are tabletop
   manipulation tasks; balancing is a separate research problem and would
   dominate the two weeks. Fixing the base matches how LIBERO-style
   manipulation benchmarks are set up.
2. A parallel-jaw gripper is attached (see gripper.py).
"""
from __future__ import annotations

import mujoco

from gripper import add_parallel_gripper

G1_XML = "unitree_robots/g1/g1_29dof.xml"

TABLE_POS = (0.55, 0.0, 0.35)
TABLE_HALF = (0.35, 0.5, 0.02)
BLOCK_HALF = (0.022, 0.022, 0.022)


def build(block_pos=(0.5, -0.2, 0.45), fix_base: bool = True):
    """Return (spec, model) for a G1 + gripper + table + block scene."""
    spec = mujoco.MjSpec.from_file(G1_XML)

    if fix_base:
        pelvis = spec.body("pelvis")
        for j in pelvis.joints:
            spec.delete(j)

    add_parallel_gripper(spec, "right_wrist_yaw_link", "right")

    # Offscreen framebuffer must be >= any camera resolution we render for VLA data.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    world = spec.worldbody

    light = world.add_light()
    light.pos = [0, 0, 2.5]
    light.dir = [0, 0, -1]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    table = world.add_body()
    table.name = "table"
    table.pos = list(TABLE_POS)
    tg = table.add_geom()
    tg.name = "table_top"
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = list(TABLE_HALF)
    tg.rgba = [0.55, 0.42, 0.30, 1.0]

    block = world.add_body()
    block.name = "block"
    block.pos = list(block_pos)
    bj = block.add_joint()
    bj.name = "block_free"
    bj.type = mujoco.mjtJoint.mjJNT_FREE
    bg = block.add_geom()
    bg.name = "block_geom"
    bg.type = mujoco.mjtGeom.mjGEOM_BOX
    bg.size = list(BLOCK_HALF)
    bg.rgba = [0.85, 0.15, 0.15, 1.0]
    bg.friction = [1.6, 0.05, 0.001]
    bg.mass = 0.05

    cam = world.add_camera()
    cam.name = "scene_cam"
    cam.pos = [1.4, -1.0, 1.2]
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    cam.targetbody = "table"

    return spec, spec.compile()
