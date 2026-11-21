# Applied Robotics and AI Project – Part 2

Harish Ashokkumar (8757HIQT) · University of Hertfordshire / PSB Academy

A CNN for CIFAR-10 object classification, trained off the robot in Google Colab and optimised for the robot,
plus an e-puck in Webots that searches for coloured blocks and runs the CNN live on its camera.

| Result | Value |
|---|---|
| CNN test accuracy (ResNet-9) | **93.1%** (93.5% with flip test-time augmentation) |
| My own photos | 4 of 4 correct (86–92% confidence) |
| e-puck camera in Webots | 3 of 5 correct (Colab and live on the robot) |
| Block search (red, blue, green) | 3 of 3 found in every test run, "I see <colour>" + summary |
| Optimisation | ResNet-9 int8: 25.1 MB → 6.3 MB, accuracy unchanged; small CNN 1.3 ms per image |

## Repository layout

```
cnn/
  CNN_Project_Part2.ipynb     full Colab notebook: 3 CNNs, evaluation, predictions, TFLite optimisation
  CNN_Simple.ipynb            short 5-step notebook: load, build (all layers), train (graphs), accuracy, predict 2 images
  stage2_resnet9.py           ResNet-9 training, evaluation, predictions and TFLite benchmark
  my_images/                  my own photos (car, cat, dog, ship)
  webots_images/              e-puck camera frames (dog, cat, horse, deer, airplane)
  results/                    results.json and the result figures
webots/
  worlds/colour_block_search.wbt   Part 1 brief: red, blue, green blocks 0.3 × 0.3 × 0.1 m, random positions
  worlds/cnn_object_test.wbt       CNN test: miniature CIFAR-10 objects + 4 coloured blocks
  controllers/epuck_controller/    the single e-puck controller + resnet9_int8.tflite
docs/                         diagrams, screenshots, Trello and GitHub evidence
report/                       final report (Assignment 4) and the script that builds it
```

## 1. CNN (Google Colab)

Open `cnn/CNN_Simple.ipynb` (or the full `CNN_Project_Part2.ipynb`) in Google Colab, choose
*Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. Step 5 shows a *Choose files* button:
pick the images to classify. An e-puck frame (320 × 240) is cropped to the region of interest automatically.

| Model | Parameters | Test accuracy |
|---|---|---|
| Baseline (Session 8 CNN_v3), SGD / Adam | 122,570 | 61.6% / 69.7% |
| Improved CNN (BatchNorm, Dropout, augmentation, GAP) | 323,946 | 84.5% |
| ResNet-9 (one-cycle, label smoothing, Cutout) | 6,577,610 | 93.1% |

ResNet-9 layers: 8 convolutions (3 × 3, each with BatchNorm and ReLU; 64 → 128 → 256 → 512 filters), two residual
blocks, three max-pooling stages, global max pooling and a Dense 10 output.

## 2. Webots (R2023b, Python 3 with numpy)

### World 1 – colour block search (`colour_block_search.wbt`, Part 1 brief)

Open the world and press Run. The controller runs in **search mode** (`controllerArgs "search"`).

1. **Layout** – `randomise_objects()` moves the red, blue and green blocks (0.3 m × 0.3 m × 0.1 m, physics
   bounding boxes) to random, non-overlapping positions every run (≥ 0.45 m apart, ≥ 0.4 m from the robot).
2. **State machine** – SEARCH (drive ~5 s, then spin to scan) → APPROACH (lock on to the colour, proportional
   steering) → RECORD → AVOID (reverse and turn) → back to SEARCH, until DONE.
3. **Response** – on the **first encounter** of each block the robot displays `I see red` / `I see blue` /
   `I see green`, then a summary of the blocks encountered so far and those still to find. The found block is
   repainted a pale colour so it is never reported twice.
4. **Output** – console, on-screen labels in the 3D view (`Supervisor.setLabel`), `captures/run_log.txt`,
   one camera frame per block and a screenshot `captures/search_final.jpg`.
5. **Error handling** – if a block cannot be reached within 10 simulated minutes, the final summary lists it as
   not found; a blocked approach is skipped rather than reported.

Example log (`captures/run_log.txt`):

```
I see blue
Encountered so far (1/3): blue  |  still to find: red, green
I see green
Encountered so far (2/3): blue, green  |  still to find: red
I see red
Encountered so far (3/3): blue, green, red  |  all blocks found
Objects found: 3/3 ... Vision cost: mean 7.72 ms/frame (control step = 32 ms)
```

### World 2 – CNN on the robot camera (`cnn_object_test.wbt`)

The controller runs in **tour mode** (`controllerArgs "tour"`): it shuffles the miniature dog, cat, horse, deer,
airplane and four 8 cm blocks, drives to each in nearest-first order, classifies every CIFAR object **live with
the ResNet-9** (TFLite int8, square region of interest above the floor resized to 32 × 32) and names each block
colour with HSV. Example run: CNN 3/5 (deer 66%, horse 83%, airplane 92%), colours 4/4, 101.7 s.
To see the camera: select the e-puck, then *Overlays → Camera Devices → Show 'camera' overlay*.

### Colour ranges (HSV, OpenCV scale H 0–179, S/V 0–255)

| Colour | H | S | V | Note |
|---|---|---|---|---|
| Red | 0–8 and 170–180 | 150–255 | 60–255 | two ranges (hue wraps); S ≥ 150 because the wooden floor is a faint red |
| Green | 40–85 | 80–255 | 40–255 | |
| Blue | 100–130 | 120–255 | 40–255 | |
| Yellow | 20–35 | 120–255 | 40–255 | V from 40: shaded faces look olive (tour mode) |

### Optimisation in the controller

* Camera pre-filter: vision runs on 1 control step in 4, on a frame down-sampled 2× (5–8 ms per frame vs 32 ms step).
* The CNN runs in a background thread (`CnnWorker`), so motor commands never wait for it; only while stopped.
* The model is quantised to int8 TFLite (6.3 MB). In search mode the CNN is off: plain blocks are not CIFAR classes.

### Set-up notes

* **CNN on the robot:** needs TensorFlow (`pip install numpy pillow tensorflow`, or `tflite-runtime`) in the Python
  that Webots uses; without it the controller runs colour-only and prints `[CNN] disabled`.
* **Python for Webots:** if your default `python` lacks numpy, add `runtime.ini` in the controller folder:
  `[python]` / `COMMAND = C:\path\to\python.exe`.

### Hardware notes (e-puck)

* Camera: front-facing, 320 × 240 (set in the world files).
* Motors: two differential-drive wheels, max 6.28 rad/s, velocity control (`setPosition(inf)`).
* Proximity: 8 IR sensors `ps0`–`ps7`; front sensors ps0, ps1, ps6, ps7 trigger obstacle avoidance.
* InertialUnit in `turretSlot` for heading (tour mode).

## Project management

Tasks tracked on Trello (https://trello.com/b/xMSJpIeZ) and code on GitHub; see the final report for details.
