"""Round-trip check: can LeRobot read back what to_lerobot.py wrote?

    python scripts/entrance_test/verify_lerobot.py --root lerobot/task1

Converting is not proof of anything on its own -- the dataset has to load and
yield correctly shaped tensors. This asserts that, and is the check to re-run
after any change to the recorder or the converter.
"""
from __future__ import annotations

import sys, argparse

VIDEO_BACKEND = "pyav"   # see module docstring in to_lerobot.py


def verify(root, repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, video_backend=VIDEO_BACKEND)
    print(f"episodes : {ds.num_episodes}")
    print(f"frames   : {ds.num_frames}")
    print(f"fps      : {ds.fps}")
    print(f"robot    : {ds.meta.robot_type}")

    problems = []
    for idx in (0, ds.num_frames // 2, ds.num_frames - 1):
        s = ds[idx]
        for key in ("observation.state", "action", "task"):
            if key not in s:
                problems.append(f"frame {idx}: missing {key}")
        for key in [k for k in s if k.startswith("observation.images")]:
            v = s[key]
            if v.ndim != 3 or v.shape[0] != 3:
                problems.append(f"frame {idx}: {key} has shape {tuple(v.shape)}, expected (3,H,W)")
            if not (0.0 <= float(v.min()) and float(v.max()) <= 1.0):
                problems.append(f"frame {idx}: {key} outside [0,1]")
        if not isinstance(s.get("task"), str) or not s["task"]:
            problems.append(f"frame {idx}: task is not a non-empty string")

    s = ds[0]
    print(f"state    : {tuple(s['observation.state'].shape)}")
    print(f"action   : {tuple(s['action'].shape)}")
    for k in [k for k in s if k.startswith("observation.images")]:
        print(f"{k:25s}: {tuple(s[k].shape)} {s[k].dtype}")
    print(f"task     : {s['task']!r}")

    if problems:
        print("\nFAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("\nPASS - dataset loads and every checked frame is well formed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="lerobot/task1")
    ap.add_argument("--repo-id", default="indiguana/g1-pickplace-task1")
    a = ap.parse_args()
    return verify(a.root, a.repo_id)


if __name__ == "__main__":
    sys.exit(main())
