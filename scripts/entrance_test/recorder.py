"""Episode recorder for VLA training data.

What gets stored, and why a VLA needs it:

  observation.images.head    third-person RGB. Gives scene context -- where the
                             object is, where the target is, what else is around.
  observation.images.wrist   hand-mounted RGB. Once the gripper is close, the
                             object leaves the useful resolution of the head
                             view; the wrist view is what resolves fine
                             alignment, and it is the view that transfers best
                             across changes in robot base pose.
  observation.state          proprioception: 7 arm joint angles + 2 finger
                             positions. The policy needs to know the configuration
                             it is issuing actions from -- images alone are
                             ambiguous about joint angles behind occlusions.
  action                     7 commanded arm joint targets + 1 gripper command.
                             Deliberately the *commanded* target, not the achieved
                             state: a policy must output what the controller
                             consumes, and on real hardware the achieved state
                             lags the command.
  task                       the natural-language instruction. This is what makes
                             the dataset vision-*language*-action rather than
                             plain behaviour cloning, and it is what lets one
                             policy serve several tasks.
  timestamp / frame_index    needed to reconstruct temporal order and to chunk
                             action sequences for models that predict horizons.

Layout on disk (one directory per episode):

    <root>/episode_000000/
        meta.json                  instruction, success, fps, schema
        frames.npz                 state, action, timestamp arrays
        images/head/000000.jpg
        images/wrist/000000.jpg
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import mujoco

OBS_CAMERAS = {"head": "head_cam", "wrist": "wrist_cam"}
IMAGE_WH = (320, 240)


class EpisodeRecorder:
    def __init__(self, root, instruction, fps=30, timestep=0.002,
                 image_wh=IMAGE_WH, cameras=None):
        self.root = root
        self.instruction = instruction
        self.fps = fps
        self.cameras = cameras or OBS_CAMERAS
        self.w, self.h = image_wh
        self.every = max(1, int(round(1.0 / (fps * timestep))))
        self.renderer = None
        self._reset()

    def _reset(self):
        self.states, self.actions, self.times = [], [], []
        self.images = {k: [] for k in self.cameras}
        self.i = 0

    def attach(self, model):
        self.renderer = mujoco.Renderer(model, self.h, self.w)

    def step(self, model, data, state, action):
        """Call once per physics step; samples at the configured fps."""
        if self.i % self.every == 0:
            self.states.append(np.asarray(state, dtype=np.float32))
            self.actions.append(np.asarray(action, dtype=np.float32))
            self.times.append(self.i * model.opt.timestep)
            for key, cam in self.cameras.items():
                self.renderer.update_scene(data, camera=cam)
                self.images[key].append(self.renderer.render())
        self.i += 1

    def save(self, index, success, extra=None):
        from PIL import Image

        d = os.path.join(self.root, f"episode_{index:06d}")
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

        for key, frames in self.images.items():
            sub = os.path.join(d, "images", key)
            os.makedirs(sub)
            for t, fr in enumerate(frames):
                Image.fromarray(fr).save(os.path.join(sub, f"{t:06d}.jpg"), quality=92)

        states = np.stack(self.states) if self.states else np.zeros((0, 0), np.float32)
        actions = np.stack(self.actions) if self.actions else np.zeros((0, 0), np.float32)
        np.savez_compressed(os.path.join(d, "frames.npz"),
                            state=states, action=actions,
                            timestamp=np.asarray(self.times, dtype=np.float32))

        meta = {
            "episode_index": index,
            "task": self.instruction,
            "success": bool(success),
            "fps": self.fps,
            "num_frames": len(self.times),
            "image_size": [self.w, self.h],
            "cameras": list(self.cameras),
            "state_dim": int(states.shape[1]) if states.size else 0,
            "action_dim": int(actions.shape[1]) if actions.size else 0,
            "state_names": ["arm_q0", "arm_q1", "arm_q2", "arm_q3", "arm_q4",
                            "arm_q5", "arm_q6", "finger_l", "finger_r"],
            "action_names": ["arm_target0", "arm_target1", "arm_target2",
                             "arm_target3", "arm_target4", "arm_target5",
                             "arm_target6", "gripper_cmd"],
        }
        if extra:
            meta.update(extra)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        n = len(self.times)
        self._reset()
        return d, n
