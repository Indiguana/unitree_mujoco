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
from pick_conditional import ConditionalPickPlace
from recorder import EpisodeRecorder

INSTRUCTION = "pick up the red block and place it at the target spot"


# The four task-2 instructions, cycled rather than sampled. At 50 episodes,
# random draws leave visible imbalance (measured: one combination 9 times and
# another 14), and target colour correlated with table slot 16:7 -- enough for a
# policy to shortcut on position instead of reading the word. Cycling makes the
# instruction distribution exactly uniform; the block layout stays random.
TASK2_COMBOS = [(c, s) for c in ("red", "yellow") for s in ("left", "right")]


def _make_task(task, seed, episode=None):
    """Return (task_object, instruction) for the requested task."""
    if task == "task1":
        t = PickPlace(seed=seed)
        return t, INSTRUCTION
    if task == "task2":
        color, side = TASK2_COMBOS[(episode if episode is not None else seed)
                                   % len(TASK2_COMBOS)]
        t = ConditionalPickPlace(seed=seed, color=color, side=side)
        return t, t.instruction
    raise ValueError(f"unknown task: {task}")


def collect(out_dir, episodes, fps=30, start_seed=0, image_wh=(320, 240),
            task="task1"):
    os.makedirs(out_dir, exist_ok=True)
    index, kept, t0 = [], 0, time.time()

    instructions = {}
    for n in range(episodes):
        seed = start_seed + n
        obj, instr = _make_task(task, seed, episode=n)
        rec = EpisodeRecorder(out_dir, instr, fps=fps,
                              timestep=obj.m.opt.timestep, image_wh=image_wh)
        obj.attach_recorder(rec)

        log = obj.run()
        instructions[instr] = instructions.get(instr, 0) + 1
        extra = {"seed": seed,
                 "contacts_after_close": log.get("contacts_after_close"),
                 "lift_m": log.get("lift")}
        if task == "task1":
            extra["block_start"] = obj.block_xyz().tolist()
            extra["reach_error_m"] = log["site_to_block"]
            extra["place_error_m"] = log["place_err"]
        else:
            extra.update({"target_color": log["color"], "target_side": log["side"],
                          "layout": {k: list(v) for k, v in obj.layout.items()},
                          "distractor_moved_m": log["distractor_moved_m"],
                          "placed_in_wrong_bin": log["placed_in_wrong_bin"]})
        path, nframes = rec.save(n, log["success"], extra=extra)
        kept += bool(log["success"])
        index.append({"episode": n, "seed": seed, "frames": nframes,
                      "instruction": instr, "success": bool(log["success"])})
        detail = (f"place_err={log['place_err']*1000:6.1f}mm" if task == "task1"
                  else f"{log['color']:>6}->{log['side']:<5}")
        print(f"  ep {n:>3}  seed={seed:<4} frames={nframes:<4} {detail}  "
              f"{'ok' if log['success'] else 'FAIL'}")

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"task_name": task, "fps": fps,
                   "instructions": instructions,
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
    ap.add_argument("--task", choices=("task1", "task2"), default="task1")
    a = ap.parse_args()
    collect(a.out, a.episodes, a.fps, a.start_seed, (a.width, a.height), a.task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
