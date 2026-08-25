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

# Table geometry is derived from the arm's measured workspace, not guessed.
# Sampling 60k random arm poses shows the right grasp site cannot go below
# z=0.63 at all, and the usable front-right region only widens above z~0.80.
# A table top at 0.82 puts graspable objects inside that band.
TABLE_POS = (0.35, -0.20, 0.80)
TABLE_HALF = (0.16, 0.22, 0.02)
TABLE_TOP = TABLE_POS[2] + TABLE_HALF[2]          # 0.82
BLOCK_HALF = (0.022, 0.022, 0.022)
BLOCK_REST_Z = TABLE_TOP + BLOCK_HALF[2]          # resting centre height

# Region used for randomised placement. Narrower than the raw kinematic
# workspace: beyond y ~ -0.21 the gripper still reaches the point, but arrives
# at an orientation where a finger pad strikes the block during descent and
# knocks it away. Constraining orientation was tried and made things worse --
# the arm's natural approach at this table is forward-and-slightly-down
# (~[0.9, +/-0.3, -0.3]), about 70 deg off a top-down grasp, so imposing one
# fights the kinematics. Documented in REPORT.md as a known limitation.
REACH_X = (0.24, 0.38)
REACH_Y = (-0.20, -0.06)


def build(block_pos=None, fix_base: bool = True):
    """Return (spec, model) for a G1 + gripper + table + block scene."""
    if block_pos is None:
        block_pos = (0.30, -0.20, BLOCK_REST_Z)
    spec = mujoco.MjSpec.from_file(G1_XML)

    if fix_base:
        pelvis = spec.body("pelvis")
        for j in pelvis.joints:
            spec.delete(j)

    add_parallel_gripper(spec, "right_wrist_yaw_link", "right")

    # Offscreen framebuffer must be >= any camera resolution we render for VLA data.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    # Sky + ground. Purely visual, but the demo video and any human looking at
    # the scene both need it, and cameras used for VLA data see it too.
    sky = spec.add_texture()
    sky.name = "skybox"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.30, 0.50, 0.70]
    sky.rgb2 = [0.0, 0.0, 0.0]
    sky.width, sky.height = 512, 3072

    grid = spec.add_texture()
    grid.name = "groundtex"
    grid.type = mujoco.mjtTexture.mjTEXTURE_2D
    grid.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    grid.rgb1 = [0.20, 0.30, 0.40]
    grid.rgb2 = [0.10, 0.20, 0.30]
    grid.mark = mujoco.mjtMark.mjMARK_EDGE
    grid.markrgb = [0.8, 0.8, 0.8]
    grid.width, grid.height = 300, 300

    ground = spec.add_material()
    ground.name = "groundplane"
    ground.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundtex"
    ground.texuniform = True
    ground.texrepeat = [5, 5]
    ground.reflectance = 0.15

    world = spec.worldbody

    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.material = "groundplane"

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
    cam.pos = [1.3, -1.3, 1.15]
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    cam.targetbody = "pelvis"

    return spec, spec.compile()


def sample_block_pos(rng):
    """A random reachable spot on the table, for setup variants."""
    return (float(rng.uniform(*REACH_X)),
            float(rng.uniform(*REACH_Y)),
            BLOCK_REST_Z)
