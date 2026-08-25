"""Differential inverse kinematics for the G1 arm.

Hand-tuned joint waypoints would not survive the randomised object placements
the brief asks for ("create several setup variants to reduce overfitting"), so
reaching is solved rather than scripted.

Damped least squares on the site Jacobian:
    dq = J^T (J J^T + lambda^2 I)^-1 * dx
The damping term keeps the solution finite near singularities, where a plain
pseudo-inverse would demand enormous joint velocities.
"""
from __future__ import annotations

import numpy as np
import mujoco


class ArmIK:
    def __init__(self, model, site_name, joint_names, damping=0.08, max_step=0.02):
        self.m = model
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.sid < 0:
            raise ValueError(f"unknown site: {site_name}")
        self.jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    for n in joint_names]
        if any(i < 0 for i in self.jid):
            missing = [n for n, i in zip(joint_names, self.jid) if i < 0]
            raise ValueError(f"unknown joints: {missing}")
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jid])
        self.vadr = np.array([model.jnt_dofadr[j] for j in self.jid])
        self.lim = model.jnt_range[self.jid].copy()
        self.damping = damping
        self.max_step = max_step

    def solve(self, data, target_xyz, iters=200, tol=1e-3):
        """Return joint angles putting the site at `target_xyz`.

        Iterates on a scratch copy of the state, so `data` is left untouched.
        Returns (q, error). Caller should check the error rather than assume
        success -- an unreachable target converges to the closest pose it can.
        """
        d = mujoco.MjData(self.m)
        d.qpos[:] = data.qpos
        mujoco.mj_forward(self.m, d)

        jacp = np.zeros((3, self.m.nv))
        target = np.asarray(target_xyz, dtype=float)
        err = np.inf

        for _ in range(iters):
            mujoco.mj_forward(self.m, d)
            dx = target - d.site_xpos[self.sid]
            err = float(np.linalg.norm(dx))
            if err < tol:
                break
            n = np.linalg.norm(dx)
            if n > self.max_step:
                dx = dx / n * self.max_step

            mujoco.mj_jacSite(self.m, d, jacp, None, self.sid)
            J = jacp[:, self.vadr]
            lam2 = self.damping ** 2
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), dx)

            q = d.qpos[self.qadr] + dq
            d.qpos[self.qadr] = np.clip(q, self.lim[:, 0], self.lim[:, 1])

        return d.qpos[self.qadr].copy(), err


    def step_toward(self, data, target_xyz, target_mat=None, gain=1.0,
                    rot_weight=0.5):
        """One resolved-rate IK increment from the CURRENT measured state.

        Solving IK once and ramping open-loop does not work here: the PD
        controller settles ~0.02 rad short on each joint, which compounds down
        the 7-link chain into ~0.08 m of end-effector error. Raising the gains
        just oscillates and saturates the +/-25 Nm limit. Re-solving against the
        measured site position every control step closes the loop in task
        space, so the residual is driven out instead of accumulating.

        If `target_mat` (3x3, row-major) is given, orientation is controlled
        too. Position-only IK leaves the wrist free to arrive at any rotation,
        and at some block positions it arrives with a finger pad facing the
        object -- which knocks the block away instead of straddling it.

        Returns (q_target, position_error).
        """
        jacp = np.zeros((3, self.m.nv))
        dx = np.asarray(target_xyz, dtype=float) - data.site_xpos[self.sid]
        err = float(np.linalg.norm(dx))
        n = np.linalg.norm(dx)
        if n > self.max_step:
            dx = dx / n * self.max_step

        if target_mat is None:
            mujoco.mj_jacSite(self.m, data, jacp, None, self.sid)
            J = jacp[:, self.vadr]
            e = dx * gain
        else:
            jacr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, data, jacp, jacr, self.sid)
            q_cur = np.zeros(4); q_des = np.zeros(4)
            q_neg = np.zeros(4); q_err = np.zeros(4); dw = np.zeros(3)
            mujoco.mju_mat2Quat(q_cur, np.asarray(data.site_xmat[self.sid],
                                                  dtype=float).ravel())
            mujoco.mju_mat2Quat(q_des, np.asarray(target_mat, dtype=float).ravel())
            mujoco.mju_negQuat(q_neg, q_cur)
            mujoco.mju_mulQuat(q_err, q_des, q_neg)
            mujoco.mju_quat2Vel(dw, q_err, 1.0)
            nw = np.linalg.norm(dw)
            if nw > self.max_step * 4:
                dw = dw / nw * self.max_step * 4
            J = np.vstack([jacp[:, self.vadr], jacr[:, self.vadr]])
            e = np.concatenate([dx * gain, dw * rot_weight])

        lam2 = self.damping ** 2
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(J.shape[0]), e)
        q = data.qpos[self.qadr] + dq
        return np.clip(q, self.lim[:, 0], self.lim[:, 1]), err


# Gripper pointing straight down, finger pads straddling along world Y.
# Site frame: +x is the approach direction, +y is the pinch axis.
TOP_DOWN = np.array([[0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0],
                     [-1.0, 0.0, 0.0]])
