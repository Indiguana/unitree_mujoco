"""Joint-space PD control for the G1 arm.

Every G1 actuator in `unitree_mujoco` is a pure torque source
(gaintype=FIXED, biastype=NONE, ctrlrange +/-25 Nm on the arm, +/-5 Nm on the
wrist). There is no position servo and no gravity compensation, so an
unactuated arm simply hangs limp. Scripted control therefore needs a software
PD loop, which is what this module provides.
"""
from __future__ import annotations

import numpy as np
import mujoco

RIGHT_ARM = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
KP, KD = 120.0, 8.0


class ArmPD:
    """Holds/moves a named set of torque actuators toward joint targets."""

    def __init__(self, model, actuator_names=RIGHT_ARM, kp=KP, kd=KD):
        self.m, self.kp, self.kd = model, kp, kd
        self.aid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                    for n in actuator_names]
        if any(i < 0 for i in self.aid):
            missing = [n for n, i in zip(actuator_names, self.aid) if i < 0]
            raise ValueError(f"unknown actuators: {missing}")
        jid = [model.actuator_trnid[i, 0] for i in self.aid]
        self.qadr = [model.jnt_qposadr[j] for j in jid]
        self.vadr = [model.jnt_dofadr[j] for j in jid]

    def n(self):
        return len(self.aid)

    def apply(self, data, target):
        """Write PD torques for one step. `target` is joint angles, radians."""
        for k, a in enumerate(self.aid):
            tau = (self.kp * (target[k] - data.qpos[self.qadr[k]])
                   - self.kd * data.qvel[self.vadr[k]])
            data.ctrl[a] = np.clip(tau, *self.m.actuator_ctrlrange[a])


def ramp(start, goal, i, steps):
    """Smooth linear interpolation. Stepping a PD target discontinuously
    saturates the actuators and ejects held objects -- always ramp."""
    f = min(1.0, i / max(steps, 1))
    return np.asarray(start) + (np.asarray(goal) - np.asarray(start)) * f
