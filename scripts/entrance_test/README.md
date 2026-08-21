# Entrance test scripts

Original work for the VLA entrance test. Upstream `unitree_mujoco` is a low-level
sim-to-real bridge, so the task layer below is built from scratch.

| Script | Purpose | PDF task |
|---|---|---|
| `check_env.py` | Environment smoke test: model compiles, physics steps, offscreen render works | — |
| `scenes/` | MJCF scenes: table, objects, bins, cameras, gripper | 1 |
| `scripted_control.py` | Model-free scripted controller | 2 |
| `collect_demos.py` | Episode recorder (RGB + state + action + instruction) | 3 |
| `to_lerobot.py` | Converts recorded episodes -> LeRobot dataset | 3 |
| `eval_policy.py` | Pretrained VLA inference + success rate | 4 |

## Environment

```
conda activate vla-tools     # python 3.11, mujoco 3.12
python scripts/entrance_test/check_env.py
```

## Verified baseline (2026-08-21, Apple M5, macOS)

- `unitree_robots/g1/scene_29dof.xml` compiles: 31 bodies, nv=35, nu=29, ngeom=74
- Physics steps and offscreen rendering both work natively on Apple Silicon
- **No cameras** in the stock scene (`ncam=0`)
- **No actuated end-effector** — `left/right_rubber_hand` are fixed meshes with
  sculpted fingers, not articulated grippers; all 29 actuators are body/arm joints
- **Free-floating base** — nv=35 is 29 joints + 6 base DOF, so the robot balances
  under gravity and will topple without a controller
