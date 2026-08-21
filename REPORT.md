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

_Gripper approach and why. Base fixed vs. balancing, and why. Record the reasoning —
these are the load-bearing choices in the whole project._

## 2. Task design

### Task 1 —
### Task 2 —
### Task 3 —

### Setup variants / anti-overfitting

_Randomization axes and why each matters for generalization._

## 3. Model-free control

_Scripted controller. How I verified grasps hold, objects don't tunnel, and success
detection fires only on true successes._

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
