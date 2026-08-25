"""Collect demonstration episodes for task 1 (pick and place).

    python scripts/entrance_test/collect_demos.py --episodes 50 --out data/task1

Demonstrations come from the scripted controller, not teleoperation: upstream's
data path routes through `xr_teleoperate`, which needs an XR headset. Scripted
control is also what the brief asks for in its own right ("control the simulated
robot without a model"), so the same controller serves both purposes.

Failed episodes are recorded too, flagged `success: false`, and are kept out of
the training split by default. They are worth storing: failure data is useful
for evaluation and for training value/critic heads later.
"""
from __future__ import annotations

import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from pick_place import PickPlace
from recorder import EpisodeRecorder

INSTRUCTION = "pick up the red block and place it at the target spot"


def collect(out_dir, episodes, fps=30, start_seed=0, image_wh=(320, 240)):
    os.makedirs(out_dir, exist_ok=True)
    index, kept, t0 = [], 0, time.time()

    for n in range(episodes):
        seed = start_seed + n
        task = PickPlace(seed=seed)
        rec = EpisodeRecorder(out_dir, INSTRUCTION, fps=fps,
                              timestep=task.m.opt.timestep, image_wh=image_wh)
        task.attach_recorder(rec)

        block0 = task.block_xyz().tolist()
        log = task.run()
        path, nframes = rec.save(n, log["success"], extra={
            "seed": seed,
            "block_start": block0,
            "reach_error_m": log["site_to_block"],
            "contacts_after_close": log["contacts_after_close"],
            "lift_m": log["lift"],
            "place_error_m": log["place_err"],
        })
        kept += bool(log["success"])
        index.append({"episode": n, "seed": seed, "frames": nframes,
                      "success": bool(log["success"]),
                      "place_error_m": round(log["place_err"], 4)})
        print(f"  ep {n:>3}  seed={seed:<4} frames={nframes:<4} "
              f"place_err={log['place_err']*1000:6.1f}mm  "
              f"{'ok' if log['success'] else 'FAIL'}")

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"task": INSTRUCTION, "fps": fps,
                   "episodes": len(index), "successes": kept,
                   "image_size": list(image_wh), "records": index}, f, indent=2)

    dt = time.time() - t0
    print(f"\n{kept}/{episodes} succeeded  ({kept/episodes*100:.0f}%)  "
          f"in {dt:.0f}s ({dt/episodes:.1f}s/episode)")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="data/task1")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    a = ap.parse_args()
    collect(a.out, a.episodes, a.fps, a.start_seed, (a.width, a.height))
    return 0


if __name__ == "__main__":
    sys.exit(main())
