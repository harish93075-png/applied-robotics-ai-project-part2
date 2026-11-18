# Applied Robotics and AI Project – Part 2

CNN object classification (CIFAR-10, Session 8 `cnn_v3`) and colour detection with an e-puck in Webots.

## Repository layout

```
cnn/
  CNN_Project_Part2.ipynb     Colab notebook: CNN build, training, evaluation, predictions, TFLite
  my_images/                  put 3 of your own photos here (jpg/png)
  webots_images/              e-puck camera frames (dog, cat, horse, deer, airplane) for section 9
webots/
  worlds/cnn_object_test.wbt  walled 3 × 3 m arena: miniature CIFAR objects + 4 coloured blocks
  controllers/epuck_controller/epuck_controller.py   the single e-puck controller
docs/screenshots/             Webots screenshots and camera frames used in the slides
```

## 1. CNN (Google Colab notebook)

`cnn/CNN_Project_Part2.ipynb` builds on the Session 8 `CNN_v3` notebook. Open it in Google Colab,
choose *Runtime → Change runtime type → GPU*, then *Runtime → Run all*. Upload images when asked.

| Section | Brief requirement | Output (in `outputs/`) |
|---|---|---|
| 2, 4 | Show all layers | `model.summary()`, `baseline_layers.png`, `improved_layers.png` |
| 3, 5 | Accuracy score | `baseline_sgd_vs_adam.png`, `improved_curves.png`, `accuracy_comparison.png` |
| 6 | Error analysis | `confusion_matrix.png`, per-class accuracy |
| 8 | 3 own images | `own_image_predictions.png` |
| 9 | Webots camera image | `webots_predictions.png` |
| 10 | Optimisation | `cifar10_cnn_fp32/fp16/int8.tflite`, `optimisation.png` |
| 11 | All numbers | `results.json`, `cnn_outputs.zip` |

Models:
* **Baseline (Session 8 CNN_v3):** Conv32 → Pool → Conv64 → Pool → Conv64 → Flatten → Dense64 → Dense10,
  trained 10 epochs with SGD and with Adam (each from a fresh model).
* **Improved:** three blocks of 2 × Conv3×3 + BatchNorm + ReLU, MaxPool, Dropout (32 → 64 → 128 filters),
  global average pooling, Dense256 + Dropout, Dense10; Adam, augmentation, early stopping.

## 2. Webots simulation (Webots R2023b, Python 3, numpy)

Open `webots/worlds/cnn_object_test.wbt` and press run.

* **Arena:** walled 3 m × 3 m. Miniature dog, cat, horse and deer (Webots animal models with their
  `scale` field set to 0.14–0.6, about 0.25–0.3 m long), a miniature airplane built from primitives,
  and four 8 cm blocks (red, green, blue, yellow). Only CIFAR-10 classes from Session 8 are used.
* **Controller:** one file, `epuck_controller.py`, running as a supervisor (`controllerArgs "tour"`).
  1. SHUFFLE – moves all 9 objects to random, non-overlapping positions.
  2. CHOOSE – picks the nearest unvisited object.
  3. DRIVE – go-to-point with proportional heading control (position from the supervisor, heading from
     an InertialUnit in the turret slot) and bump-and-sidestep obstacle avoidance (front IR sensors).
  4. DETECT – CIFAR objects: stops 0.55–0.7 m away, faces it, saves `captures/<object>.png` and
     classifies the camera frame **live with the ResNet-9 CNN** (TFLite int8, about 10–40 ms): a square
     region of interest above the floor, resized to 32×32, exactly as in the notebook.
     Blocks: stops 0.25 m away and identifies the colour by HSV thresholding on the lower-centre of
     the image (ignores sky and side objects).
  5. SUMMARY – printed and written to `captures/tour_log.txt`.
  On-screen labels in the 3D view show the current target, each CNN or colour verdict and a running score.
  Example run (`docs/webots_live_cnn_final.jpg`): CNN 3/5 on the robot camera (deer 66%, horse 83%,
  airplane 92%; cat and dog missed: the simulation domain gap), colours 4/4, 101.7 s.
* **Colour ranges (OpenCV scale):** red H 0–8 and 170–180, S ≥ 150 (the wooden floor is a faint red);
  yellow H 20–35; green H 40–85; blue H 100–130; S/V minimums in the code comments.
* The same controller also contains a "search" mode (wander, detect, approach, recolour found objects)
  from the earlier colour-block world; it is selected with `controllerArgs "search"`.
* **CNN on the robot:** `resnet9_int8.tflite` (6.3 MB) sits next to the controller. It needs TensorFlow
  (`pip install numpy pillow tensorflow==2.21.0`, or `tflite-runtime` where available) in the Python
  that Webots uses; without it the controller runs colour-only and prints `[CNN] disabled`.
* **Python for Webots:** if your default `python` lacks numpy or TensorFlow, add a `runtime.ini` in the
  controller folder (not committed, because the path is machine-specific):
  `[python]` / `COMMAND = C:\path\to\python.exe`.
* **Camera window:** select the e-puck, then Overlays → Camera Devices → Show 'camera' overlay.

### Hardware notes (e-puck)

* Camera: front-facing, set to 320 × 240 in the world file.
* Motors: two differential-drive wheels, max 6.28 rad/s, velocity control (`setPosition(inf)`).
* Proximity: 8 IR sensors `ps0`–`ps7`; front sensors ps0, ps1, ps6, ps7 used for collision avoidance.
* InertialUnit added in `turretSlot` for heading.

## Project management

Tasks tracked on Trello (board: *Applied Robotics & AI Project – Part 2*).
