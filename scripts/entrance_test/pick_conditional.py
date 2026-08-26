"""Task 2: language-conditioned pick and place.

Two coloured blocks, two bins. The instruction names which block goes where:

    "put the red block in the left bin"
    "put the yellow block in the right bin"

This is the task that makes the dataset vision-*language*-action. In task 1 the
instruction was constant, so a policy could ignore it entirely and lose nothing.
Here the same visual scene maps to four different correct behaviours depending
on the sentence, so the language has to be read.

Two properties are enforced to stop the language being shortcut-able:
  - colour is assigned to a table slot at random, so "red" cannot be inferred
    from position (see scene.sample_conditional_layout);
  - target bin is drawn independently of the target colour, so neither word
    predicts the other.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

from scene import (build_conditional, sample_conditional_layout, COLORS,
                   BIN_X, BIN_Y, BIN_INNER, TABLE_TOP, BLOCK_HALF, BLOCK_REST_Z)
from manipulator import Manipulator, STROKE, APPROACH_H

SIDES = ("left", "right")
RELEASE_H = 0.10          # above the bin rim before opening
DISTRACTOR_TOL = 0.03     # how far the other block may move and still pass


def instruction(color, side):
    return f"put the {color} block in the {side} bin"


class ConditionalPickPlace(Manipulator):
    def __init__(self, seed=None, layout=None, color=None, side=None):
        rng = np.random.default_rng(seed)
        if layout is None:
            layout = (sample_conditional_layout(rng) if seed is not None
                      else {"red": (0.28, -0.185), "yellow": (0.28, -0.085)})
        self.layout = layout
        self.color = color or str(rng.choice(list(COLORS)))
        self.side = side or str(rng.choice(SIDES))
        self.instruction = instruction(self.color, self.side)
        spec, model = build_conditional(layout)
        super().__init__(model)
        self.distractor = next(c for c in COLORS if c != self.color)
        self._distractor_start = self.body_xyz(f"block_{self.distractor}")

    def bin_center(self, side):
        return np.array([BIN_X, BIN_Y[side], TABLE_TOP])

    def in_bin(self, block_xyz, side):
        c = self.bin_center(side)
        inside_xy = np.all(np.abs(block_xyz[:2] - c[:2]) < BIN_INNER)
        # resting on the bin floor, not balanced on a wall or still in the air
        on_floor = abs(block_xyz[2] - (TABLE_TOP + 0.006 + BLOCK_HALF[2])) < 0.02
        return bool(inside_xy and on_floor)

    def run(self):
        target = f"block_{self.color}"
        b = self.body_xyz(target)
        log = {"instruction": self.instruction, "color": self.color,
               "side": self.side}
        log.update(self.pick(b))
        log["contacts_after_close"] = self.contacts_with(f"{target}_geom")

        z0 = float(self.body_xyz(target)[2])
        self.move_to(self.site_xyz() + np.array([0, 0, APPROACH_H]), grip=STROKE)
        log["lift"] = float(self.body_xyz(target)[2]) - z0

        drop = self.bin_center(self.side) + np.array([0, 0, RELEASE_H])
        self.move_to(drop, grip=STROKE)
        self.set_grip(0.0)
        self.set_grip(0.0, steps=800)

        final = self.body_xyz(target)
        dist_moved = float(np.linalg.norm(
            self.body_xyz(f"block_{self.distractor}") - self._distractor_start))
        log["placed_in_target_bin"] = self.in_bin(final, self.side)
        log["placed_in_wrong_bin"] = self.in_bin(
            final, "right" if self.side == "left" else "left")
        log["distractor_moved_m"] = dist_moved
        log["final_xyz"] = final.round(4).tolist()
        log["success"] = bool(log["placed_in_target_bin"]
                              and dist_moved < DISTRACTOR_TOL)
        return log


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--color", choices=list(COLORS))
    ap.add_argument("--side", choices=SIDES)
    ap.add_argument("--video", metavar="PATH")
    a = ap.parse_args()

    t = ConditionalPickPlace(seed=a.seed, color=a.color, side=a.side)
    if a.video:
        t.enable_recording()
    log = t.run()
    for k, v in log.items():
        print(f"  {k:24s} {v}")
    if a.video:
        import imageio
        imageio.mimsave(a.video, t.frames, fps=30, quality=8)
        print(f"  wrote {a.video}")
    return 0 if log["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
