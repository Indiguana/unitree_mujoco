# VLA Project Entrance Test — Report

**Author:** Indi Kolluru
**Period:** 2026-08-21 to 2026-09-02
**Fork:** https://github.com/Indiguana/unitree_mujoco (branch `entrance-test`)
**Diff vs upstream:** https://github.com/Indiguana/unitree_mujoco/compare/main...entrance-test
**Demo video:** _TBD_
**Dataset:** _TBD_

> Per Qiaoan Shen's guidance, this work uses **MuJoCo with `unitree_mujoco`** rather than
> Isaac Lab, as Isaac Lab does not support macOS. Everything here runs on an Apple M5 laptop.

---

## 0. Summary

_What I built, what worked, what didn't. Write last, put first._

## 1. Starting point and constraints

`unitree_mujoco` is a low-level sim-to-real bridge (DDS `LowCmd`/`LowState`), not a task
suite. Verified against `unitree_robots/g1/scene_29dof.xml` on day one:

| Property | Value | Consequence for this project |
|---|---|---|
| Bodies / nv / nu | 31 / 35 / 29 | nv−nu = 6 → free-floating base |
| Cameras | 0 | Must author camera sites for VLA observations |
| End-effector | `left/right_rubber_hand`, fixed mesh | No actuated gripper; grasping needs a solution |
| Scene contents | floor, light, skybox | Table, objects, bins all authored from scratch |

### Design decisions taken from these

**Fixed base.** The pelvis free joint is removed (`scene.py`). These are tabletop
manipulation tasks; whole-body balancing is a separate research problem that would
dominate the two weeks. This matches how LIBERO-style manipulation benchmarks are posed.

**Parallel-jaw gripper, real contact physics** (`gripper.py`). Two slide-jointed
finger pads per wrist, driven by position actuators, added through MuJoCo's spec API
so the upstream robot XML stays untouched. Rejected alternatives: a weld-constraint
"magnetic" grasp (fast, but the grasp would not be physical, a weak answer to the
brief's "interact with the environment correctly"), and Menagerie's three-finger G1
(highest fidelity, but multi-finger grasp control is a research problem in itself).

**Software PD control** (`control.py`). Every G1 actuator here is a pure torque source
(gaintype=FIXED, biastype=NONE) with no gravity compensation, so an uncommanded arm
hangs limp. Joint-space PD is the minimum needed for any scripted motion.

### What tuning actually revealed

The first lift attempts dropped the block, and the instinct was to grip harder. That
was wrong. A parameter sweep over grip stiffness and friction showed:

| finger kp | friction | outcome |
|---|---|---|
| 60 | 1.6 | **held**, +0.093 m |
| 250 | 1.6 | **held**, +0.091 m |
| 800 | 1.6 | dropped |
| 60 / 250 / 800 | 3.0 | dropped |

Raising either friction or stiffness made grasping strictly *worse* — excessive
contact stiffness destabilises MuJoCo's solver and ejects the object. The real fix
was in the controller, not the gripper: stepping the PD target discontinuously
saturates the torque limits and shakes the block out. Ramping the target smoothly
(`control.ramp`) made the stock parameters work.

**Grasp assist deferred.** A contact-gated weld (weld only while both finger pads
report real contact) remains available as a stabiliser if longer trajectories or
randomised object variants prove brittle. It is not needed for the current motion
profile, and unearned complexity is worth avoiding. If added, the recorded *action*
will remain the gripper command, never the weld state — a policy trained on this data
must learn to close the gripper, because that is what real hardware executes.

**Verified:** `scripts/entrance_test/test_grasp.py` -- 8 contacts on close,
+0.093 m lift, contacts maintained throughout.

## 2. Task design

### Task 1 — pick and place a block  *(implemented)*

Instruction: *"pick up the red block and put it down at the target spot."*
The block starts at a random reachable spot on the table; the robot reaches,
grasps, lifts, transports and releases it. Success = block within 6 cm of the
target in xy and resting at table height.

### Task 2 — conditional pick and place  *(next)*
### Task 3 — multi-stage / tool use  *(next)*

### Setup variants / anti-overfitting

Block position is sampled uniformly over the reachable table region
(`scene.sample_block_pos`), so no fixed start pose can be memorised. This is
why reaching is IK-solved rather than scripted from hand-tuned joint angles:
hand-tuned waypoints only work for one block position and would defeat the
purpose of randomising. Colour, size and distractor objects are the next axes.

## 3. Model-free control

Scripted control, no learned policy. The pipeline per episode is:
reach to pre-grasp -> descend -> close gripper -> lift -> transport -> release.

**Measured over 20 randomised block placements: 18/20 success (90%).**
Typical run: reach error 3-5 mm, 10 finger-block contacts on close, 117 mm lift,
final placement within 2-3 mm of target.

### Getting there took three corrections

**1. Table placement was guessed, and wrong.** The first table (top at z=0.37)
was completely unreachable -- IK errors of 240-490 mm everywhere. Sampling 60k
random arm poses showed the right grasp site cannot descend below z=0.63 at all,
and the usable front-right region only opens up above z~0.80. The table is now
positioned from that measured workspace rather than by eye.

**2. Open-loop IK does not survive PD droop.** Solving IK once and ramping to
the solution left the fingertip 79 mm from the block, even though the IK answer
itself was correct to 0.33 mm (verified by teleporting the joints to it). The
cause is ~0.02 rad of steady-state error per joint compounding down a 7-link
chain. Raising the gains made it worse -- kp=400+ oscillates and saturates the
+/-25 Nm limit, pushing the error to 100-350 mm. The fix was closed-loop
task-space control: re-solve a damped-least-squares IK increment against the
*measured* site position every control step, so residual is driven out instead
of accumulating. Error dropped from 79 mm to 3-5 mm.

**3. Gravity compensation.** Arm actuators are pure torque sources with no
gravity compensation, so `data.qfrc_bias` is fed forward and the PD term handles
only tracking error.

### Known limitation: grasp orientation is uncontrolled

Position-only IK leaves the wrist free to arrive at any rotation. Beyond
y ~ -0.21 the gripper still reaches the target point, but arrives rotated such
that a finger pad strikes the block during descent and knocks it away -- the
error trace *diverges* (18 -> 53 mm) while the contact list shows
`right_finger_l_pad` against `block_geom`.

Adding a 6-DOF orientation constraint was tried and made results worse (2/12).
The reason: the arm's natural approach at this table is forward-and-slightly-down
(~[0.9, +/-0.3, -0.3]), roughly 70 degrees away from a top-down grasp, so
imposing one fights the kinematics. The randomised region is therefore
restricted to the band where the natural approach works. Choosing an orientation
target derived from the arm's natural pose, rather than an idealised top-down
frame, is the obvious next step.

## 4. Data collection pipeline

_What is recorded per frame and **why a VLA needs each field**._

| Field | Source | Why a VLA needs it |
|---|---|---|
| | | |

_Episode counts, storage footprint, and how I validated the data by replaying it._

## 5. Policy integration (stretch)

_Which policy, how it was wired in, results, honest failure analysis._

## 6. Problems encountered

_Including the ones I did not solve._

## 7. Hours spent

| Phase | Hours |
|---|---|
| Environment setup + baseline verification | |
| Scene authoring (MJCF) | |
| Scripted control | |
| Data pipeline | |
| Policy integration | |
| Reporting + video | |
| **Total** | |
