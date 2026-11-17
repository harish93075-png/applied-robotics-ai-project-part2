"""
Supervisor: place the four coloured blocks at random, non-overlapping
positions at the start of every run (keeps them away from the robot start
point and the arena walls), then exits.
"""

import random

from controller import Supervisor

ARENA_HALF = 1.5          # 3 m x 3 m arena
BLOCK_HALF = 0.15         # 0.3 m blocks
WALL_MARGIN = 0.1
MIN_GAP = 0.5             # min centre distance between blocks
ROBOT_CLEARANCE = 0.5     # keep blocks away from the e-puck start (0, 0)
BLOCK_DEFS = ["BLOCK_RED", "BLOCK_GREEN", "BLOCK_BLUE", "BLOCK_YELLOW"]


def random_positions(n, seed=None):
    rng = random.Random(seed)
    limit = ARENA_HALF - BLOCK_HALF - WALL_MARGIN
    placed = []
    while len(placed) < n:
        x, y = rng.uniform(-limit, limit), rng.uniform(-limit, limit)
        if (x * x + y * y) ** 0.5 < ROBOT_CLEARANCE:
            continue
        if all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 >= MIN_GAP for px, py in placed):
            placed.append((x, y))
    return placed


def main():
    sup = Supervisor()
    for name, (x, y) in zip(BLOCK_DEFS, random_positions(len(BLOCK_DEFS))):
        node = sup.getFromDef(name)
        if node is None:
            print(f"[RANDOMISER] {name} not found in world")
            continue
        node.getField("translation").setSFVec3f([x, y, 0.05])
        node.getField("rotation").setSFRotation([0, 0, 1, random.uniform(0, 3.14)])
        node.resetPhysics()
        print(f"[RANDOMISER] {name:<13} -> ({x:+.2f}, {y:+.2f})")
    sup.step(int(sup.getBasicTimeStep()))


if __name__ == "__main__":
    main()
