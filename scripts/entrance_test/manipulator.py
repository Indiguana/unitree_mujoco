"""Shared manipulation machinery: motion primitives, sensing, recording.

Task 1 (pick_place.py) and task 2 (pick_conditional.py) differ only in scene
contents, instruction, and success criterion -- the arm control is identical, so
it lives here once.
"""
from __future__ import annotations

import numpy as np
import mujoco

from control import ArmPD
from ik import ArmIK

ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint",
              "right_wrist_roll_joint", "right_wrist_pitch_joint",
              "right_wrist_yaw_joint"]

# Only the right arm performs the task, but the left arm is a torque-driven
# chain like any other: left uncommanded it sags into a collapsed pose with the
# elbow at ~1.5 rad, folded across the chest. That is visible in every rendered
# frame and in the head camera, so it is held at a neutral rest pose instead.
LEFT_ARM_ACTUATORS = ["left_shoulder_pitch", "left_shoulder_roll",
                      "left_shoulder_yaw", "left_elbow",
                      "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"]
LEFT_REST = [0.10, 0.18, 0.0, 0.25, 0.0, 0.0, 0.0]

# The waist is unactuated too, and the arm's reaction forces drag it: measured
# drift of yaw -0.28, roll -0.52, pitch +0.52 rad (about 30 deg) over a single
# episode. This is not cosmetic -- head_cam is mounted on torso_link, so an
# uncommanded waist means the "fixed" third-person camera swings through every
# episode. Holding it upright keeps the viewpoint actually fixed.
WAIST_ACTUATORS = ["waist_yaw", "waist_roll", "waist_pitch"]
WAIST_REST = [0.0, 0.0, 0.0]
STROKE = 0.024
APPROACH_H = 0.12


class Manipulator:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.nid = lambda t, n: mujoco.mj_name2id(self.m, t, n)
        self.sid = self.nid(mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site")
        self.pads = {self.nid(mujoco.mjtObj.mjOBJ_GEOM, "right_finger_l_pad"),
                     self.nid(mujoco.mjtObj.mjOBJ_GEOM, "right_finger_r_pad")}
        self.fingers = [self.nid(mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                        for n in ("right_finger_l_act", "right_finger_r_act")]
        self.arm = ArmPD(self.m)
        self.left_arm = ArmPD(self.m, LEFT_ARM_ACTUATORS)
        self.left_rest = np.array(LEFT_REST)
        self.waist = ArmPD(self.m, WAIST_ACTUATORS)
        self.waist_rest = np.array(WAIST_REST)
        self.ik = ArmIK(self.m, "right_grasp_site", ARM_JOINTS)
        self.qadr = self.ik.qadr
        self.fadr = [self.m.jnt_qposadr[self.nid(mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in ("right_finger_l_slide", "right_finger_r_slide")]
        mujoco.mj_forward(self.m, self.d)
        self.frames = []
        self.renderer = None
        self.recorder = None
        self._grip_cmd = 0.0

    # ---- sensing -------------------------------------------------------
    def body_xyz(self, name):
        return self.d.xpos[self.nid(mujoco.mjtObj.mjOBJ_BODY, name)].copy()

    def site_xyz(self):
        return self.d.site_xpos[self.sid].copy()

    def contacts_with(self, geom_name):
        g = self.nid(mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        return sum(1 for c in self.d.contact[:self.d.ncon]
                   if (c.geom1 in self.pads and c.geom2 == g)
                   or (c.geom2 in self.pads and c.geom1 == g))

    def observation(self):
        """7 arm joint angles + 2 finger positions."""
        return np.concatenate([self.d.qpos[self.qadr],
                               [self.d.qpos[a] for a in self.fadr]])

    # ---- recording -----------------------------------------------------
    def enable_recording(self, w=960, h=720):
        self.renderer = mujoco.Renderer(self.m, h, w)

    def attach_recorder(self, recorder):
        recorder.attach(self.m)
        self.recorder = recorder

    def _hold_left(self):
        """Hold the non-task joints steady: left arm and waist."""
        self.left_arm.apply(self.d, self.left_rest)
        self.waist.apply(self.d, self.waist_rest)

    def _record(self, q_target):
        if self.recorder is not None:
            action = np.concatenate([q_target, [self._grip_cmd]])
            self.recorder.step(self.m, self.d, self.observation(), action)

    def _maybe_frame(self, i, every=17):
        if self.renderer is not None and i % every == 0:
            self.renderer.update_scene(self.d, camera="scene_cam")
            self.frames.append(self.renderer.render())

    # ---- motion --------------------------------------------------------
    def move_to(self, xyz, steps=1800, grip=None, tol=0.003):
        """Closed-loop task-space servo. See ik.step_toward for why open-loop
        IK is not enough here."""
        err = np.inf
        for i in range(steps):
            q_target, err = self.ik.step_toward(self.d, xyz)
            self.arm.apply(self.d, q_target)
            self._hold_left()
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
            self._hold_left()
            for a in self.fingers:
                self.d.ctrl[a] = value
            self._record(q)
            mujoco.mj_step(self.m, self.d)
            self._maybe_frame(i)

    # ---- composite -----------------------------------------------------
    def pick(self, target_xyz):
        """Approach from above, descend, close. Returns a small log dict."""
        log = {}
        log["ik_pregrasp"] = self.move_to(target_xyz + np.array([0, 0, APPROACH_H]),
                                          grip=0.0)
        log["ik_grasp"] = self.move_to(target_xyz, grip=0.0)
        self.set_grip(STROKE)
        return log

    def place(self, target_xyz, release_h=APPROACH_H):
        """Lift, traverse above the target, lower, open."""
        self.move_to(self.site_xyz() + np.array([0, 0, release_h]), grip=STROKE)
        self.move_to(np.array([target_xyz[0], target_xyz[1],
                               target_xyz[2] + release_h]), grip=STROKE)
        self.set_grip(0.0)
        self.set_grip(0.0, steps=600)
