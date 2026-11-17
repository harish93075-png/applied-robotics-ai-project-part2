"""
Applied Robotics and AI Project - Part 2
Single e-puck controller for both Webots worlds (replaces epuck_colour_detector,
camera_tour and block_randomiser).

The mode is chosen automatically from the world (or from controllerArgs:
"search" / "tour"):

MODE "search"  (worlds/colour_block_search.wbt)
    Shuffles the 4 blocks and 4 toys to random positions, then runs a state
    machine:
        SEARCH   -> wander + spin, scan camera for any target colour
        APPROACH -> lock on to one colour, keep it centred, drive to it
        RECORD   -> log it, save a camera frame, recolour it with a pale
                    "found" colour (a different one each time)
        AVOID    -> reverse and turn when the proximity sensors fire
        DONE     -> every object found (or time limit) -> summary
    Log: captures/run_log.txt.  Keys (click the 3D view): S = save frame,
    Q = summary + stop.

MODE "tour"    (worlds/cnn_object_test.wbt)
    Shuffles the miniature CIFAR-10 objects (dog, cat, horse, deer, airplane)
    and 4 small coloured blocks, then drives to the nearest unvisited object
    each time: saves a frame of each CIFAR object for the CNN, and identifies
    each block's colour with HSV thresholding.  Log: captures/tour_log.txt.

Shared: vectorised HSV vision, colour ranges, random layout, logging.
Optimisations: camera processed on 1 in N steps, frames down-sampled x2,
optional TFLite CNN in a background thread.
"""

import math
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
from controller import Keyboard, Supervisor

# ------------------------------------------------------------------ tuning
MAX_SPEED = 6.28                 # e-puck wheel limit (rad/s)
CAMERA_EVERY_N = 4               # run vision on 1 in N control steps
DOWNSAMPLE = 2                   # pixel stride for vision
MIN_BLOB_FRACTION = 0.01         # blob must cover >=1% of frame to start an approach
KEEP_BLOB_FRACTION = 0.002       # ...but only 0.2% to keep tracking it (hysteresis)
CLOSE_BLOB_FRACTION = 0.35       # blob this large => we are at the block
OBSTACLE_THRESHOLD = 80.0        # ps sensor value that means "very close"
TIME_LIMIT_S = 600               # give up and summarise after 10 sim-minutes
CAPTURE_DIR = Path(__file__).parent / "captures"
LOG_FILE = CAPTURE_DIR / "run_log.txt"        # colour-search mode
TOUR_LOG_FILE = CAPTURE_DIR / "tour_log.txt"  # tour mode


def log(message, log_file=None):
    """Print to the Webots console and append to the current mode's log file."""
    print(message)
    try:
        with open(log_file or LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except OSError:
        pass
TFLITE_MODEL = Path(__file__).parent / "cifar10_cnn_int8.tflite"

# HSV ranges, OpenCV-style scale: H 0-179, S 0-255, V 0-255.
# Red wraps around hue 0, so it has two ranges.
COLOUR_RANGES = {
    # red needs S >= 150: the wooden floor is a low-saturation red (S up to ~122)
    "red":    [((0, 150, 60), (8, 255, 255)), ((170, 150, 60), (180, 255, 255))],
    "green":  [((40, 80, 40), (85, 255, 255))],
    "blue":   [((100, 120, 40), (130, 255, 255))],
    "yellow": [((20, 120, 40), (35, 255, 255))],   # V from 40: shaded faces look olive
}

# objects in the world: DEF name -> (label, starting colour)
# (original RGB is in ORIGINAL_RGB, used to restore colours at start-up)
TARGET_OBJECTS = {
    "BLOCK_RED": ("red block", "red"),
    "BLOCK_GREEN": ("green block", "green"),
    "BLOCK_BLUE": ("blue block", "blue"),
    "BLOCK_YELLOW": ("yellow block", "yellow"),
    "TOY_CAR": ("toy car", "blue"),
    "DUCK": ("rubber duck", "yellow"),
    "BALL": ("ball", "red"),
    "TREE": ("tree", "green"),
}
ORIGINAL_RGB = {
    "BLOCK_RED": [1, 0, 0], "BLOCK_GREEN": [0, 0.8, 0],
    "BLOCK_BLUE": [0, 0.2, 1], "BLOCK_YELLOW": [1, 0.9, 0],
    "TOY_CAR": [0, 0.3, 1], "DUCK": [1, 0.85, 0],
    "BALL": [0.9, 0.05, 0.05], "TREE": [0, 0.6, 0.1],
}
MATCH_RADIUS = 0.6               # recorded object must be this close (m)
# colours given to objects once found: all pale (saturation < 0.45), so they
# are outside every HSV range whatever the lighting, and a found object is
# never detected again
FOUND_COLOURS = [("white", [0.95, 0.95, 0.95]), ("pink", [1.0, 0.65, 0.8]),
                 ("lavender", [0.78, 0.68, 1.0]), ("mint", [0.7, 1.0, 0.82]),
                 ("peach", [1.0, 0.82, 0.68]), ("sky", [0.72, 0.86, 1.0]),
                 ("lemon", [1.0, 1.0, 0.72]), ("grey", [0.6, 0.6, 0.6])]

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


# ------------------------------------------------------------------ vision
def bgra_to_hsv(frame):
    """Vectorised RGB->HSV on a (h, w, 4) BGRA uint8 frame, OpenCV scale."""
    rgb = frame[:, :, 2::-1].astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = rgb.max(axis=2)
    cmin = rgb.min(axis=2)
    delta = cmax - cmin + 1e-6
    hue = np.where(cmax == r, ((g - b) / delta) % 6,
          np.where(cmax == g, (b - r) / delta + 2, (r - g) / delta + 4))
    hue = hue * 30.0                                    # 0-180 scale
    sat = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-6), 0) * 255.0
    val = cmax * 255.0
    return np.stack([hue, sat, val], axis=2)


def colour_mask(hsv, ranges):
    """Boolean mask of pixels inside any of the HSV ranges."""
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for low, high in ranges:
        mask |= np.all((hsv >= low) & (hsv <= high), axis=2)
    return mask


def classify_rgb(rgb):
    """Name of the detection colour an RGB triple (0-1) falls in, or None."""
    pixel = np.zeros((1, 1, 4), np.uint8)
    pixel[0, 0, 2::-1] = np.clip(np.array(rgb) * 255, 0, 255).astype(np.uint8)
    hsv = bgra_to_hsv(pixel)
    for colour, ranges in COLOUR_RANGES.items():
        if colour_mask(hsv, ranges)[0, 0]:
            return colour
    return None


def detect_colour_blob(frame, colour, min_fraction=KEEP_BLOB_FRACTION):
    """(colour, area_fraction, centre_x_norm) for one specific colour, or None."""
    small = frame[::DOWNSAMPLE, ::DOWNSAMPLE]
    mask = colour_mask(bgra_to_hsv(small), COLOUR_RANGES[colour])
    fraction = mask.mean()
    if fraction < min_fraction:
        return None
    xs = np.nonzero(mask)[1]
    centre = (xs.mean() / (mask.shape[1] - 1)) * 2 - 1
    return (colour, float(fraction), float(centre))


def detect_largest_blob(frame, ignore=()):
    """Return (colour, area_fraction, centre_x_norm) of the biggest blob, or None.

    Colours in `ignore` (blocks already recorded) are skipped so the robot
    does not keep re-targeting them. centre_x_norm is -1 (left) ... +1 (right).
    """
    small = frame[::DOWNSAMPLE, ::DOWNSAMPLE]
    hsv = bgra_to_hsv(small)
    best = None
    for colour, ranges in COLOUR_RANGES.items():
        if colour in ignore:
            continue
        mask = colour_mask(hsv, ranges)
        fraction = mask.mean()
        if fraction < MIN_BLOB_FRACTION:
            continue
        if best is None or fraction > best[1]:
            xs = np.nonzero(mask)[1]
            centre = (xs.mean() / (mask.shape[1] - 1)) * 2 - 1
            best = (colour, float(fraction), float(centre))
    return best


# ------------------------------------------------------------------ CNN worker
class CnnWorker:
    """Runs the optimised (TFLite int8) CIFAR-10 CNN off the control loop."""

    def __init__(self, model_path):
        self.enabled = False
        self.last_result = None
        self._busy = False
        try:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite import Interpreter
            self.interp = Interpreter(model_path=str(model_path), num_threads=1)
            self.interp.allocate_tensors()
            self.inp = self.interp.get_input_details()[0]
            self.out = self.interp.get_output_details()[0]
            self.enabled = True
            print("[CNN] loaded", model_path.name)
        except Exception as err:  # model or runtime missing -> colour-only mode
            print("[CNN] disabled:", err)

    def submit(self, frame):
        """Classify a frame asynchronously; ignored if the worker is busy."""
        if not self.enabled or self._busy:
            return
        self._busy = True
        threading.Thread(target=self._run, args=(frame.copy(),), daemon=True).start()

    def _run(self, frame):
        try:
            rgb = frame[:, :, 2::-1]
            # nearest-neighbour resize to 32x32 (no extra dependencies)
            ys = np.linspace(0, rgb.shape[0] - 1, 32).astype(int)
            xs = np.linspace(0, rgb.shape[1] - 1, 32).astype(int)
            img = rgb[ys][:, xs].astype(np.float32) / 255.0
            self.interp.set_tensor(self.inp["index"], img[None].astype(self.inp["dtype"]))
            t0 = time.perf_counter()
            self.interp.invoke()
            ms = (time.perf_counter() - t0) * 1000
            logits = self.interp.get_tensor(self.out["index"])[0].astype(np.float64)
            probs = np.exp(logits - logits.max())        # model outputs logits
            probs /= probs.sum()
            k = int(np.argmax(probs))
            self.last_result = (CIFAR10_CLASSES[k], float(probs[k]), ms)
        finally:
            self._busy = False


# ------------------------------------------------------------------ robot
class ColourSearch:
    """Mode "search": wander, detect, approach, record and recolour objects."""

    def __init__(self, robot):
        self.robot = robot
        self.dt = int(self.robot.getBasicTimeStep())

        self.left = self.robot.getDevice("left wheel motor")
        self.right = self.robot.getDevice("right wheel motor")
        for motor in (self.left, self.right):
            motor.setPosition(float("inf"))   # velocity control mode
            motor.setVelocity(0.0)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.dt * CAMERA_EVERY_N)
        self.cam_w = self.camera.getWidth()
        self.cam_h = self.camera.getHeight()

        self.ps = [self.robot.getDevice(f"ps{i}") for i in range(8)]
        for sensor in self.ps:
            sensor.enable(self.dt)

        self.keyboard = Keyboard()
        self.keyboard.enable(self.dt)

        CAPTURE_DIR.mkdir(exist_ok=True)
        LOG_FILE.write_text("", encoding="utf-8")       # fresh log each run
        self.cnn = CnnWorker(TFLITE_MODEL)

        self.state = "SEARCH"
        self.state_timer = 0
        self.step_count = 0
        self.target = None                 # colour we are approaching
        self.last_centre = 0.0             # where the target was last seen
        self.found = {}                    # object DEF -> record dict
        self.ignore = set()                # colours to skip (fallback only)
        self.vision_ms = []                # vision cost per processed frame
        self.search_turn = 1
        self.avoid_turn_steps = 40
        try:
            randomise_objects(self.robot, list(TARGET_OBJECTS), half_size=0.15,
                              min_gap=0.45, clearance=0.4)
        except Exception as err:
            print("[LAYOUT] could not randomise:", err)
        try:
            self.restore_colours()
        except Exception as err:                       # not a supervisor
            print("[RESET] could not restore colours:", err)

    # -------------------------------------------------- helpers
    def set_speed(self, left, right):
        self.left.setVelocity(float(np.clip(left, -MAX_SPEED, MAX_SPEED)))
        self.right.setVelocity(float(np.clip(right, -MAX_SPEED, MAX_SPEED)))

    def obstacle_ahead(self):
        """True if a front proximity sensor (ps0, ps1, ps6, ps7) is triggered."""
        front = [self.ps[i].getValue() for i in (0, 1, 6, 7)]
        return max(front) > OBSTACLE_THRESHOLD

    def get_frame(self):
        raw = self.camera.getImage()
        if raw is None:
            return None
        return np.frombuffer(raw, np.uint8).reshape(self.cam_h, self.cam_w, 4)

    def save_frame(self, tag):
        path = CAPTURE_DIR / f"{tag}_{self.robot.getTime():07.2f}s.png"
        self.camera.saveImage(str(path), 100)
        print(f"[CAM] saved {path.name}")
        return path.name

    def change_state(self, new_state):
        if new_state != self.state:
            print(f"[FSM] {self.state} -> {new_state}")
        self.state = new_state
        self.state_timer = 0

    # -------------------------------------------------- states
    def search(self, blob):
        if blob:
            self.target = blob[0]
            self.change_state("APPROACH")
            return
        # wander: drive forward ~5 s, then spin a full turn (~4 s) to scan
        cycle = self.state_timer % 280
        if cycle < 155:
            if self.obstacle_ahead():
                self.change_state("AVOID")
                return
            self.set_speed(0.6 * MAX_SPEED, 0.6 * MAX_SPEED)
        else:
            self.set_speed(-2.0 * self.search_turn, 2.0 * self.search_turn)
        if cycle == 279:
            self.search_turn *= -1

    def approach(self, blob):
        if blob is None or blob[0] != self.target:
            if self.state_timer > 90:                  # lost it for ~3 s
                self.change_state("SEARCH")
            else:                                      # turn back towards it
                turn = 1.2 if self.last_centre >= 0 else -1.2
                self.set_speed(turn, -turn)
            return
        colour, fraction, centre = blob
        self.last_centre = centre
        if fraction >= CLOSE_BLOB_FRACTION or self.obstacle_ahead():
            self.change_state("RECORD")
            return
        # proportional steering on horizontal blob centre
        base = 0.5 * MAX_SPEED
        turn = 2.5 * centre
        self.set_speed(base + turn, base - turn)
        self.state_timer = min(self.state_timer, 1)    # reset "lost" timer

    def nearest_unfound(self, colour):
        """DEF name of the closest not-yet-found object of `colour`, or None."""
        me = self.robot.getSelf()
        if me is None:
            return None
        rx, ry = me.getPosition()[:2]
        best, best_d = None, MATCH_RADIUS
        for def_name, (_, obj_colour) in TARGET_OBJECTS.items():
            if obj_colour != colour or def_name in self.found:
                continue
            node = self.robot.getFromDef(def_name)
            if node is None:
                continue
            ox, oy = node.getPosition()[:2]
            d = ((ox - rx) ** 2 + (oy - ry) ** 2) ** 0.5
            if d < best_d:
                best, best_d = def_name, d
        return best

    def paint(self, node, colour, rgb, match=None):
        """Recursively recolour every Shape under `node` that shows `colour`.

        `match` optionally replaces the test: a function(rgb) -> bool.
        """
        match = match or (lambda c: classify_rgb(c) == colour)
        changed = 0
        type_name = node.getTypeName()
        if type_name == "Shape":
            appearance = node.getField("appearance").getSFNode()
            if appearance is not None:
                field = appearance.getField("baseColor")
                if field is not None and match(field.getSFColor()):
                    field.setSFColor(rgb)
                    changed += 1
            return changed
        children = node.getField("children") if type_name in (
            "Solid", "Pose", "Transform", "Group") else None
        if children is not None:
            for i in range(children.getCount()):
                changed += self.paint(children.getMFNode(i), colour, rgb, match)
        return changed

    def restore_colours(self):
        """Undo recolouring left over from an earlier run (reset keeps it)."""
        # current palette plus the earlier one (purple, magenta, cyan, black)
        found_rgbs = [rgb for _, rgb in FOUND_COLOURS] + [
            [0.5, 0.0, 0.8], [1.0, 0.0, 0.6], [0.0, 0.8, 0.8], [1.0, 0.6, 0.8],
            [0.08, 0.08, 0.08]]

        def was_found(c):
            return any(max(abs(a - b) for a, b in zip(c, f)) < 0.02 for f in found_rgbs)

        restored = 0
        for def_name, rgb in ORIGINAL_RGB.items():
            node = self.robot.getFromDef(def_name)
            if node is not None:
                restored += self.paint(node, None, rgb, match=was_found)
        if restored:
            log(f"[RESET] restored original colours on {restored} shapes")

    def recolour(self, def_name, colour):
        """Paint object `def_name` with the next 'found' colour; return its name."""
        node = self.robot.getFromDef(def_name)
        if node is None:
            return None
        name, rgb = FOUND_COLOURS[(len(self.found) - 1) % len(FOUND_COLOURS)]
        try:
            if self.paint(node, colour, rgb) == 0:
                return None
        except Exception as err:                       # keep driving regardless
            print("[RECOLOUR] failed:", err)
            return None
        return name

    def record(self, frame):
        self.set_speed(0, 0)
        colour = self.target
        def_name = self.nearest_unfound(colour)
        if def_name is None and self.robot.getSelf() is not None:
            # stopped by an obstacle before reaching the object: not a find
            log(f"[SKIP] blocked before reaching the {colour} object")
            self.change_state("AVOID")
            return
        key = def_name or f"{colour} object"
        label = TARGET_OBJECTS[def_name][0] if def_name else key
        image = self.save_frame(label.replace(" ", "_"))
        self.found[key] = {"label": label, "colour": colour,
                           "time_s": round(self.robot.getTime(), 1),
                           "image": image, "cnn": None, "now": None}
        new_colour = self.recolour(def_name, colour) if def_name else None
        if new_colour:
            self.found[key]["now"] = new_colour
            log(f"[RECOLOUR] {label} -> {new_colour}")
        else:                       # could not recolour: stop chasing this colour
            self.ignore.add(colour)
        log(f"[FOUND] {label} ({colour}) at t={self.robot.getTime():.1f}s "
            f"({len(self.found)}/{len(TARGET_OBJECTS)})")
        if frame is not None:
            self.cnn.submit(frame)
        self.change_state("AVOID")

    def avoid(self):
        if self.state_timer < 40:
            self.set_speed(-0.5 * MAX_SPEED, -0.5 * MAX_SPEED)   # reverse
        elif self.state_timer < 40 + self.avoid_turn_steps:
            self.set_speed(0.5 * MAX_SPEED, -0.5 * MAX_SPEED)    # turn away
        else:
            self.avoid_turn_steps = int(np.random.randint(30, 70))
            done = len(self.found) >= len(TARGET_OBJECTS)
            self.change_state("DONE" if done else "SEARCH")

    def print_summary(self):
        log("============ SEARCH SUMMARY ============")
        log(f"Objects found: {len(self.found)}/{len(TARGET_OBJECTS)}")
        for rec in self.found.values():
            cnn = f", CNN: {rec['cnn']}" if rec["cnn"] else ""
            now = f" -> {rec['now']}" if rec["now"] else ""
            log(f"  {rec['label']:<13} {rec['colour']:<6}{now:<11} "
                  f"at {rec['time_s']:>6}s{cnn}")
        missing = [lbl for d, (lbl, _) in TARGET_OBJECTS.items() if d not in self.found]
        if missing:
            log("  Not found: " + ", ".join(missing))
        if self.vision_ms:
            log(f"Vision cost: mean {np.mean(self.vision_ms):.2f} ms/frame, "
                  f"max {np.max(self.vision_ms):.2f} ms "
                  f"(control step = {self.dt} ms)")
        log("========================================")

    # -------------------------------------------------- main loop
    def run(self):
        blob, frame = None, None
        while self.robot.step(self.dt) != -1:
            self.step_count += 1
            self.state_timer += 1

            key = self.keyboard.getKey()
            if key in (ord("S"), ord("s")):
                self.save_frame("manual")
            elif key in (ord("Q"), ord("q")):
                self.change_state("DONE")

            # pre-filter: only look at the camera every N steps
            if self.step_count % CAMERA_EVERY_N == 0:
                frame = self.get_frame()
                if frame is not None:
                    t0 = time.perf_counter()
                    if self.state == "APPROACH" and self.target:
                        # lock on to the chosen colour while approaching
                        blob = detect_colour_blob(frame, self.target)
                    else:
                        blob = detect_largest_blob(frame, ignore=self.ignore)
                    self.vision_ms.append((time.perf_counter() - t0) * 1000)

            # attach any finished CNN result to the latest record
            if self.cnn.last_result and self.found:
                last = list(self.found)[-1]
                if self.found[last]["cnn"] is None:
                    label, conf, ms = self.cnn.last_result
                    self.found[last]["cnn"] = f"{label} {conf:.0%} ({ms:.1f} ms)"
                    print(f"[CNN] {self.found[last]['label']} classified as "
                          f"{label} ({conf:.0%})")
                    self.cnn.last_result = None

            if self.robot.getTime() > TIME_LIMIT_S and self.state != "DONE":
                log("[FSM] time limit reached")
                self.change_state("DONE")

            if self.state == "SEARCH":
                self.search(blob)
            elif self.state == "APPROACH":
                self.approach(blob)
            elif self.state == "RECORD":
                self.record(frame)
            elif self.state == "AVOID":
                self.avoid()
            elif self.state == "DONE":
                self.set_speed(0, 0)
                self.print_summary()
                break



# ------------------------------------------------------------------ shared layout
def randomise_objects(sup, def_names, half_size, min_gap, clearance, arena_half=None):
    """Move every named object to a random, non-overlapping spot (was block_randomiser).

    The arena size is read from the RectangleArena's floorSize, so the same code
    works in the 3 m colour-search arena and the 3 m CNN arena.
    """
    if arena_half is None:
        arena_half = 1.5
        root = sup.getRoot().getField("children")
        for i in range(root.getCount()):
            node = root.getMFNode(i)
            if node.getTypeName() == "RectangleArena":
                arena_half = node.getField("floorSize").getSFVec2f()[0] / 2
    limit = arena_half - half_size - 0.1
    rng = random.Random()
    nodes = [(n, sup.getFromDef(n)) for n in def_names]
    nodes = [(n, node) for n, node in nodes if node is not None]
    for _ in range(200):
        placed = []
        for _ in range(5000):
            if len(placed) == len(nodes):
                break
            x, y = rng.uniform(-limit, limit), rng.uniform(-limit, limit)
            if math.hypot(x, y) < clearance:
                continue
            if all(math.hypot(x - px, y - py) >= min_gap for px, py in placed):
                placed.append((x, y))
        if len(placed) == len(nodes):
            break
    for (name, node), (x, y) in zip(nodes, placed):
        z = node.getField("translation").getSFVec3f()[2]
        node.getField("translation").setSFVec3f([x, y, z])
        node.getField("rotation").setSFRotation([0, 0, 1, rng.uniform(-math.pi, math.pi)])
        node.resetPhysics()
        print(f"[LAYOUT] {name:<13} -> ({x:+.2f}, {y:+.2f})")


# ------------------------------------------------------------------ tour mode
# DEF name -> (label, kind, stop distance from the object's centre in metres)
TOUR_OBJECTS = {
    "DOG": ("dog", "cnn", 0.70),
    "CAT": ("cat", "cnn", 0.65),
    "HORSE": ("horse", "cnn", 0.60),
    "DEER": ("deer", "cnn", 0.60),
    "AIRPLANE": ("airplane", "cnn", 0.55),
    "BLOCK_RED": ("block 1", "colour", 0.25),
    "BLOCK_GREEN": ("block 2", "colour", 0.25),
    "BLOCK_BLUE": ("block 3", "colour", 0.25),
    "BLOCK_YELLOW": ("block 4", "colour", 0.25),
}
TOUR_TRUE_COLOUR = {"BLOCK_RED": "red", "BLOCK_GREEN": "green",
               "BLOCK_BLUE": "blue", "BLOCK_YELLOW": "yellow"}

# layout
TOUR_RANDOMISE = True
TOUR_ARENA_LIMIT = 1.25        # keep objects this far inside the walls (m)
TOUR_START_CLEARANCE = 0.45    # nothing closer than this to the robot's start
TOUR_MIN_GAP = 0.5             # minimum distance between two objects (m)

# motion
TOUR_DRIVE_SPEED = 4.0         # wheel speed when driving straight (rad/s)
TOUR_TURN_GAIN = 4.0
TOUR_ARRIVE_TOLERANCE = 0.04   # m
TOUR_HEADING_TOLERANCE = math.radians(2)
TOUR_TARGET_TIMEOUT_S = 45     # give up on one object after this long
TOUR_SETTLE_STEPS = 10

def tlog(message):
    log(message, TOUR_LOG_FILE)


def identify_colour(frame):
    """Return (colour, fraction) of the dominant colour in the centre of the frame.

    Only the lower-centre of the image is used: the robot has turned to face the
    block, so other objects at the sides are ignored, and the sky (which reads as
    blue) is above the horizon line in the middle of the frame.
    """
    h, w = frame.shape[:2]
    hsv = bgra_to_hsv(frame[h // 2:, w // 4: 3 * w // 4][::2, ::2])
    best, best_fraction = None, 0.0
    for colour, ranges in COLOUR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=bool)
        for low, high in ranges:
            mask |= np.all((hsv >= low) & (hsv <= high), axis=2)
        if mask.mean() > best_fraction:
            best, best_fraction = colour, float(mask.mean())
    return (best, best_fraction) if best_fraction >= 0.01 else (None, 0.0)


def angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class ObjectTour:
    """Mode "tour": drive to every object; CNN frames + block colour identification."""

    def __init__(self, robot):
        self.sup = robot
        self.dt = int(self.sup.getBasicTimeStep())
        self.left = self.sup.getDevice("left wheel motor")
        self.right = self.sup.getDevice("right wheel motor")
        for m in (self.left, self.right):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.camera = self.sup.getDevice("camera")
        self.camera.enable(self.dt)
        self.imu = self.sup.getDevice("inertial unit")
        self.imu.enable(self.dt)
        self.ps = [self.sup.getDevice(f"ps{i}") for i in range(8)]
        for s in self.ps:
            s.enable(self.dt)
        self.me = self.sup.getSelf()
        CAPTURE_DIR.mkdir(exist_ok=True)
        TOUR_LOG_FILE.write_text("", encoding="utf-8")

    # ---------------------------------------------- helpers
    def step(self):
        return self.sup.step(self.dt) != -1

    def pose(self):
        x, y = self.me.getPosition()[:2]
        return x, y, self.imu.getRollPitchYaw()[2]

    def set_speed(self, l, r):
        self.left.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, l)))
        self.right.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, r)))

    def stop(self):
        self.set_speed(0, 0)

    def obstacle_ahead(self):
        return max(self.ps[i].getValue() for i in (0, 1, 6, 7)) > OBSTACLE_THRESHOLD

    def frame(self):
        raw = self.camera.getImage()
        h, w = self.camera.getHeight(), self.camera.getWidth()
        return np.frombuffer(raw, np.uint8).reshape(h, w, 4)

    def turn_to(self, heading, deadline):
        while self.step() and self.sup.getTime() < deadline:
            err = angle_diff(heading, self.pose()[2])
            if abs(err) < TOUR_HEADING_TOLERANCE:
                break
            speed = max(0.3, min(2.5, TOUR_TURN_GAIN * abs(err)))
            d = 1 if err > 0 else -1
            self.set_speed(-d * speed, d * speed)
        self.stop()

    def sidestep(self):
        """Back off, turn ~50 degrees, drive a little: get round an obstacle."""
        t_end = self.sup.getTime() + 0.6
        while self.step() and self.sup.getTime() < t_end:
            self.set_speed(-3, -3)
        side = random.choice((-1, 1))
        t_end = self.sup.getTime() + 0.55
        while self.step() and self.sup.getTime() < t_end:
            self.set_speed(-3 * side, 3 * side)
        t_end = self.sup.getTime() + 1.2
        while self.step() and self.sup.getTime() < t_end and not self.obstacle_ahead():
            self.set_speed(3.5, 3.5)
        self.stop()

    def drive_to(self, gx, gy, deadline):
        """Go-to-point with proportional heading control; True if arrived."""
        while self.step():
            if self.sup.getTime() > deadline:
                self.stop()
                return False
            x, y, yaw = self.pose()
            dist = math.hypot(gx - x, gy - y)
            if dist < TOUR_ARRIVE_TOLERANCE:
                self.stop()
                return True
            err = angle_diff(math.atan2(gy - y, gx - x), yaw)
            if abs(err) > math.radians(35):           # turn on the spot first
                d = 1 if err > 0 else -1
                self.set_speed(-2.5 * d, 2.5 * d)
                continue
            if self.obstacle_ahead():
                self.sidestep()
                continue
            base = TOUR_DRIVE_SPEED * min(1.0, dist / 0.15 + 0.3)
            self.set_speed(base - TOUR_TURN_GAIN * err, base + TOUR_TURN_GAIN * err)

    # ---------------------------------------------- main
    def run(self):
        if TOUR_RANDOMISE:
            randomise_objects(self.sup, list(TOUR_OBJECTS), half_size=0.25,
                              min_gap=TOUR_MIN_GAP, clearance=TOUR_START_CLEARANCE)
        self.step()
        remaining = [n for n in TOUR_OBJECTS if self.sup.getFromDef(n) is not None]
        colour_results, cnn_frames = [], []
        while remaining:
            x, y, _ = self.pose()
            # nearest unvisited object
            name = min(remaining, key=lambda n: math.hypot(
                self.sup.getFromDef(n).getPosition()[0] - x,
                self.sup.getFromDef(n).getPosition()[1] - y))
            remaining.remove(name)
            label, kind, stop_dist = TOUR_OBJECTS[name]
            ox, oy = self.sup.getFromDef(name).getPosition()[:2]
            dist = math.hypot(ox - x, oy - y)
            tlog(f"[GO] {label} ({kind}) at ({ox:+.2f}, {oy:+.2f}), {dist:.2f} m away")
            deadline = self.sup.getTime() + TOUR_TARGET_TIMEOUT_S
            # approach point: on the line from the robot to the object
            if dist > stop_dist:
                ux, uy = (ox - x) / dist, (oy - y) / dist
                arrived = self.drive_to(ox - ux * stop_dist, oy - uy * stop_dist, deadline)
                if not arrived:
                    tlog(f"[SKIP] could not reach {label} in {TOUR_TARGET_TIMEOUT_S} s")
                    continue
            x, y, _ = self.pose()
            self.turn_to(math.atan2(oy - y, ox - x), deadline + 5)
            for _ in range(TOUR_SETTLE_STEPS):
                self.step()
            img = self.frame()
            if kind == "cnn":
                path = CAPTURE_DIR / f"{label}.png"
                self.camera.saveImage(str(path), 100)
                cnn_frames.append(label)
                tlog(f"[DETECT] {label}: reached at t={self.sup.getTime():.1f}s, "
                    f"frame saved as {path.name} for the CNN")
            else:
                colour, frac = identify_colour(img)
                truth = TOUR_TRUE_COLOUR[name]
                ok = colour == truth
                colour_results.append((label, colour, truth, ok))
                self.camera.saveImage(str(CAPTURE_DIR / f"{label.replace(' ', '_')}.png"), 100)
                tlog(f"[COLOUR] {label}: identified as {colour or 'unknown'} "
                    f"({frac:.0%} of frame) - actual {truth} - "
                    f"{'correct' if ok else 'WRONG'}")
            # reverse a little so the next turn does not clip the object
            t_end = self.sup.getTime() + 0.6
            while self.step() and self.sup.getTime() < t_end:
                self.set_speed(-3, -3)
            self.stop()

        correct = sum(r[3] for r in colour_results)
        tlog("============ TOUR SUMMARY ============")
        tlog(f"CNN frames saved: {len(cnn_frames)}/5 ({', '.join(cnn_frames)})")
        tlog(f"Block colours correct: {correct}/{len(colour_results)}")
        for label, colour, truth, ok in colour_results:
            tlog(f"  {label}: {colour or 'unknown'} (actual {truth}) {'OK' if ok else 'X'}")
        tlog(f"Finished at t={self.sup.getTime():.1f}s")
        tlog("======================================")
        self.stop()
        while self.step():
            pass



# ------------------------------------------------------------------ entry point
def choose_mode(robot):
    """'search' or 'tour': from controllerArgs if given, else from the world."""
    for arg in sys.argv[1:]:
        if arg in ("search", "tour"):
            return arg
    return "tour" if robot.getFromDef("AIRPLANE") or robot.getFromDef("DOG") else "search"


def main():
    robot = Supervisor()                   # one Supervisor for both modes
    mode = choose_mode(robot)
    print(f"[MODE] {mode}")
    if mode == "tour":
        ObjectTour(robot).run()
    else:
        ColourSearch(robot).run()


if __name__ == "__main__":
    main()
