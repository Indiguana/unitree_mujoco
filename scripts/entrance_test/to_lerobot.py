"""Convert recorded episodes to a LeRobot dataset.

    python scripts/entrance_test/to_lerobot.py \
        --in data/task1 data/task2 \
        --repo-id indiguana/g1-manipulation --out lerobot/all

LeRobot is the format GR00T N1.5 and openpi consume, so converting to it is what
makes this data usable by an off-the-shelf VLA rather than only by my own code.

Feature naming follows the LeRobot convention:
  observation.images.<cam>   visual streams, encoded to MP4 per episode
  observation.state          proprioception vector
  action                     commanded control vector
  task                       instruction string, attached per frame

LeRobot adds timestamp / frame_index / episode_index / index / task_index itself.

By default only successful episodes are converted: failures are useful for
evaluation but would teach a behaviour-cloning policy to drop things.

Reading back on Apple Silicon: LeRobot defaults to the `torchcodec` decoder,
which needs arm64 FFmpeg shared libraries under /opt/homebrew. If the machine
only has an x86 Homebrew FFmpeg (/usr/local), torchcodec fails to load with a
libavutil dlopen error. Pass `video_backend="pyav"` when opening the dataset --
see verify_lerobot.py. Writing is unaffected; only decoding is.
"""
from __future__ import annotations

import sys, os, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

STATE_NAMES = ["arm_q0", "arm_q1", "arm_q2", "arm_q3", "arm_q4", "arm_q5",
               "arm_q6", "finger_l", "finger_r"]
ACTION_NAMES = ["arm_target0", "arm_target1", "arm_target2", "arm_target3",
                "arm_target4", "arm_target5", "arm_target6", "gripper_cmd"]


def build_features(h, w, cameras, state_dim, action_dim):
    feats = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,),
                              "names": STATE_NAMES[:state_dim]},
        "action": {"dtype": "float32", "shape": (action_dim,),
                   "names": ACTION_NAMES[:action_dim]},
    }
    for cam in cameras:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return feats


def convert(in_dirs, repo_id, out_dir, include_failures=False, limit=None):
    """Convert one or more recorded task directories into a single dataset.

    Several tasks belong in one dataset, not several: a VLA is trained to
    condition on the instruction, so mixing tasks under distinct instructions is
    the point. LeRobot tracks each distinct `task` string separately.
    """
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if isinstance(in_dirs, str):
        in_dirs = [in_dirs]

    with open(os.path.join(in_dirs[0], "index.json")) as f:
        index = json.load(f)

    chosen = []
    for in_dir in in_dirs:
        for d in sorted(glob.glob(os.path.join(in_dir, "episode_*"))):
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            if meta["success"] or include_failures:
                chosen.append((d, meta))
    if limit:
        chosen = chosen[:limit]
    if not chosen:
        raise SystemExit("no episodes to convert")

    _, m0 = chosen[0]
    w, h = m0["image_size"]
    cams = m0["cameras"]
    features = build_features(h, w, cams, m0["state_dim"], m0["action_dim"])

    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=index["fps"], features=features,
        root=out_dir, robot_type="unitree_g1_29dof_parallel_gripper",
        use_videos=True,
    )

    total = 0
    tasks = {}
    for d, meta in chosen:
        tasks[meta["task"]] = tasks.get(meta["task"], 0) + 1
        arr = np.load(os.path.join(d, "frames.npz"))
        state, action = arr["state"], arr["action"]
        n = len(state)
        for t in range(n):
            frame = {
                "observation.state": state[t].astype(np.float32),
                "action": action[t].astype(np.float32),
                "task": meta["task"],
            }
            for cam in cams:
                p = os.path.join(d, "images", cam, f"{t:06d}.jpg")
                frame[f"observation.images.{cam}"] = np.asarray(Image.open(p).convert("RGB"))
            ds.add_frame(frame)
        ds.save_episode()
        total += n
        print(f"  {os.path.basename(d)}: {n} frames")

    if hasattr(ds, "finalize"):
        ds.finalize()
    print(f"\nconverted {len(chosen)} episodes / {total} frames -> {out_dir}")
    print("instructions in dataset:")
    for t, n in sorted(tasks.items()):
        print(f"  {n:>3}x  {t}")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", nargs="+", default=["data/task1"])
    ap.add_argument("--repo-id", default="indiguana/g1-pickplace-task1")
    ap.add_argument("--out", default="lerobot/task1")
    ap.add_argument("--include-failures", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    convert(a.in_dir, a.repo_id, a.out, a.include_failures, a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
