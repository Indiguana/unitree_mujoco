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


# --- Task 2: conditional pick and place ------------------------------------
# Bins sit at the far edge of the table. Release tolerates a much wider region
# than grasping does (measured: sub-mm IK error across x 0.24-0.40, y -0.40..-0.04
# at release height, versus the narrow band grasping needs), so the bins can go
# where the blocks cannot.
BIN_X = 0.385
BIN_Y = {"left": -0.075, "right": -0.335}
BIN_INNER = 0.038
BIN_WALL = 0.006
BIN_WALL_H = 0.025
BIN_RGBA = {"left": [0.20, 0.35, 0.75, 1.0], "right": [0.25, 0.55, 0.30, 1.0]}

COLORS = {"red": [0.85, 0.15, 0.15, 1.0], "yellow": [0.90, 0.80, 0.10, 1.0]}
# Two well-separated slots so the gripper cannot straddle both blocks at once.
SLOT_X = (0.25, 0.31)
SLOT_Y = {"a": (-0.20, -0.17), "b": (-0.10, -0.07)}


def _base_scene(fix_base: bool = True):
    """Robot + gripper + table + lighting + cameras, shared by all tasks."""
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

    # --- observation cameras for VLA data -----------------------------
    # A VLA needs the scene from a stable viewpoint plus a close-in view that
    # moves with the hand; the wrist view is what disambiguates fine alignment
    # once the object is out of the third-person camera's useful resolution.
    # Both are rigidly mounted, as they would be on real hardware.
    head = spec.body("torso_link").add_camera()
    head.name = "head_cam"
    head.pos = [0.08, 0.0, 0.42]                 # approx head height on torso
    head.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    head.targetbody = "table"
    head.fovy = 58

    wrist = spec.body(f"right_gripper_palm").add_camera()
    wrist.name = "wrist_cam"
    wrist.pos = [-0.045, 0.0, 0.055]
    wrist.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    # look along palm +x, tilted ~20 deg down toward the grasp point
    wrist.alt.xyaxes = [0, -1, 0, 0.34, 0, 0.94]
    wrist.fovy = 70

    cam = world.add_camera()
    cam.name = "scene_cam"
    cam.pos = [1.3, -1.3, 1.15]
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    cam.targetbody = "pelvis"

    return spec, world


def _add_block(world, name, pos, rgba):
    b = world.add_body()
    b.name = name
    b.pos = list(pos)
    j = b.add_joint()
    j.name = f"{name}_free"
    j.type = mujoco.mjtJoint.mjJNT_FREE
    g = b.add_geom()
    g.name = f"{name}_geom"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(BLOCK_HALF)
    g.rgba = list(rgba)
    g.friction = [1.6, 0.05, 0.001]
    g.mass = 0.05
    return b


def _add_bin(world, side):
    """Shallow open tray. Low walls so a block released from above drops in
    without bouncing back out."""
    y = BIN_Y[side]
    body = world.add_body()
    body.name = f"bin_{side}"
    body.pos = [BIN_X, y, TABLE_TOP]
    rgba = BIN_RGBA[side]

    base = body.add_geom()
    base.name = f"bin_{side}_base"
    base.type = mujoco.mjtGeom.mjGEOM_BOX
    base.size = [BIN_INNER + BIN_WALL, BIN_INNER + BIN_WALL, 0.003]
    base.pos = [0, 0, 0.003]
    base.rgba = rgba

    for k, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        w = body.add_geom()
        w.name = f"bin_{side}_wall{k}"
        w.type = mujoco.mjtGeom.mjGEOM_BOX
        w.size = ([BIN_WALL, BIN_INNER + BIN_WALL, BIN_WALL_H] if dx
                  else [BIN_INNER + BIN_WALL, BIN_WALL, BIN_WALL_H])
        w.pos = [dx * (BIN_INNER + BIN_WALL), dy * (BIN_INNER + BIN_WALL), BIN_WALL_H]
        w.rgba = rgba
    return body


def build(block_pos=None, fix_base: bool = True):
    """Task 1 scene: one red block on the table."""
    if block_pos is None:
        block_pos = (0.30, -0.20, BLOCK_REST_Z)
    spec, world = _base_scene(fix_base)
    _add_block(world, "block", block_pos, COLORS["red"])
    return spec, spec.compile()


def build_conditional(layout=None, fix_base: bool = True):
    """Task 2 scene: two coloured blocks and two bins.

    `layout` maps colour -> (x, y). Colour is assigned to slot randomly by
    `sample_conditional_layout`, so colour is never correlated with position --
    otherwise a policy could satisfy the instruction by memorising 'the block
    at the far slot' and ignore the word entirely.
    """
    if layout is None:
        layout = {"red": (0.28, -0.185), "yellow": (0.28, -0.085)}
    spec, world = _base_scene(fix_base)
    for color, (x, y) in layout.items():
        _add_block(world, f"block_{color}", (x, y, BLOCK_REST_Z), COLORS[color])
    for side in ("left", "right"):
        _add_bin(world, side)
    return spec, spec.compile()


def sample_conditional_layout(rng):
    """Randomise both positions and which colour occupies which slot."""
    colors = list(COLORS)
    rng.shuffle(colors)
    layout = {}
    for color, slot in zip(colors, ("a", "b")):
        layout[color] = (float(rng.uniform(*SLOT_X)),
                         float(rng.uniform(*SLOT_Y[slot])))
    return layout


def sample_block_pos(rng):
    """A random reachable spot on the table, for setup variants."""
    return (float(rng.uniform(*REACH_X)),
            float(rng.uniform(*REACH_Y)),
            BLOCK_REST_Z)
