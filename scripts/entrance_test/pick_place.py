"""Scripted pick-and-place: reach, grasp, lift, place. (Brief tasks 1 and 2.)

Motion is IK-solved rather than hand-tuned so the same script works for
randomised block placements. Joint targets are always ramped -- stepping a PD
target discontinuously saturates the torque limits and ejects held objects.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

from scene import build, BLOCK_REST_Z, sample_block_pos
from control import ArmPD, ramp
from ik import ArmIK, TOP_DOWN

ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint",
              "right_wrist_roll_joint", "right_wrist_pitch_joint",
              "right_wrist_yaw_joint"]
STROKE = 0.024
APPROACH_H = 0.12          # pre-grasp height above the block


class PickPlace:
    def __init__(self, block_pos=None, seed=None):
        rng = np.random.default_rng(seed)
        spec, self.m = build(block_pos if block_pos is not None
                             else (sample_block_pos(rng) if seed is not None else None))
        self.d = mujoco.MjData(self.m)
        nid = lambda t, n: mujoco.mj_name2id(self.m, t, n)
        self.bid = nid(mujoco.mjtObj.mjOBJ_BODY, "block")
        self.jadr = self.m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "block_free")]
        self.sid = nid(mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site")
        self.pads = {nid(mujoco.mjtObj.mjOBJ_GEOM, "right_finger_l_pad"),
                     nid(mujoco.mjtObj.mjOBJ_GEOM, "right_finger_r_pad")}
        self.bg = nid(mujoco.mjtObj.mjOBJ_GEOM, "block_geom")
        self.fingers = [nid(mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                        for n in ("right_finger_l_act", "right_finger_r_act")]
        self.arm = ArmPD(self.m)
        self.ik = ArmIK(self.m, "right_grasp_site", ARM_JOINTS)
        self.qadr = self.ik.qadr
        self.fadr = [self.m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in ("right_finger_l_slide", "right_finger_r_slide")]
        mujoco.mj_forward(self.m, self.d)
        self.frames = []
        self.renderer = None
        self.recorder = None
        self._grip_cmd = 0.0

    # ---- helpers -------------------------------------------------------
    def contacts(self):
        return sum(1 for c in self.d.contact[:self.d.ncon]
                   if (c.geom1 in self.pads and c.geom2 == self.bg)
                   or (c.geom2 in self.pads and c.geom1 == self.bg))

    def block_xyz(self):
        return self.d.xpos[self.bid].copy()

    def site_xyz(self):
        return self.d.site_xpos[self.sid].copy()

    def enable_recording(self, w=960, h=720):
        self.renderer = mujoco.Renderer(self.m, h, w)

    def attach_recorder(self, recorder):
        recorder.attach(self.m)
        self.recorder = recorder

    def observation(self):
        """Proprioception: 7 arm joint angles + 2 finger positions."""
        return np.concatenate([self.d.qpos[self.qadr],
                               [self.d.qpos[a] for a in self.fadr]])

    def _record(self, q_target):
        if self.recorder is not None:
            action = np.concatenate([q_target, [self._grip_cmd]])
            self.recorder.step(self.m, self.d, self.observation(), action)

    def _maybe_frame(self, i, every=17):
        if self.renderer is not None and i % every == 0:
            self.renderer.update_scene(self.d, camera="scene_cam")
            self.frames.append(self.renderer.render())

    def move_to(self, xyz, steps=1800, grip=None, tol=0.003, orient=False):
        """Servo the grasp site to `xyz` with closed-loop task-space IK.

        Each step re-solves against the measured site position, so PD droop is
        corrected rather than accumulated. Returns the final error in metres.
        """
        err = np.inf
        for i in range(steps):
            q_target, err = self.ik.step_toward(
                self.d, xyz, TOP_DOWN if orient else None)
            self.arm.apply(self.d, q_target)
            if grip is not None:
                self._grip_cmd = grip
                for a in self.fingers:
                    self.d.ctrl[a] = grip
            self._record(q_target)
            mujoco.mj_step(self.m, self.d)
            self._maybe_frame(i)
            if err < tol and i > steps // 4:
                break
        return err

    def set_grip(self, value, steps=400):
        q = self.d.qpos[self.qadr].copy()
        self._grip_cmd = value
        for i in range(steps):
            self.arm.apply(self.d, q)
            for a in self.fingers:
                self.d.ctrl[a] = value
            self._record(q)
            mujoco.mj_step(self.m, self.d)
            self._maybe_frame(i)

    # ---- the task ------------------------------------------------------
    def run(self, place_xy=(0.34, -0.10)):
        b = self.block_xyz()
        log = {}
        log["ik_pregrasp"] = self.move_to(b + np.array([0, 0, APPROACH_H]), grip=0.0)
        log["ik_grasp"] = self.move_to(b, grip=0.0)
        log["site_to_block"] = float(np.linalg.norm(self.site_xyz() - self.block_xyz()))
        self.set_grip(STROKE)
        log["contacts_after_close"] = self.contacts()
        z_before = float(self.block_xyz()[2])
        self.move_to(self.block_xyz() + np.array([0, 0, APPROACH_H]), grip=STROKE)
        log["lift"] = float(self.block_xyz()[2]) - z_before
        log["contacts_after_lift"] = self.contacts()
        self.move_to(np.array([place_xy[0], place_xy[1], BLOCK_REST_Z + APPROACH_H]),
                     grip=STROKE)
        self.move_to(np.array([place_xy[0], place_xy[1], BLOCK_REST_Z]), grip=STROKE)
        self.set_grip(0.0)
        self.set_grip(0.0, steps=600)
        final = self.block_xyz()
        log["place_err"] = float(np.linalg.norm(final[:2] - np.array(place_xy)))
        log["final_z"] = float(final[2])
        log["success"] = bool(log["place_err"] < 0.06
                              and abs(final[2] - BLOCK_REST_Z) < 0.03)
        return log


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--video", metavar="PATH", help="record the run to MP4")
    args = ap.parse_args()

    t = PickPlace(seed=args.seed)
    if args.video:
        t.enable_recording()
    log = t.run()
    if args.video:
        import imageio
        imageio.mimsave(args.video, t.frames, fps=30, quality=8)
        print(f"  wrote {args.video} ({len(t.frames)} frames)")
    for k, v in log.items():
        print(f"  {k:24s} {v}")
    return 0 if log["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
