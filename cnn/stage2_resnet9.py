# Stage 2 of CNN_Project_Part2.ipynb - run in the same Colab session after sections 1-9:
#   exec(open("stage2_resnet9.py").read())
# 1. ResNet-9 (Page / DAWNBench design) trained from scratch for maximum accuracy
# 2. accuracy comparison, curves, confusion matrix for the new best model
# 3. predictions on own images and e-puck frames: raw resize and object-centred square crop
# 4. TFLite fp32/fp16/int8 benchmark for ResNet-9 and the small improved CNN (robot trade-off)
import json, os, shutil, time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from PIL import Image

EPOCHS, BATCH, PEAK_LR, WD = 30, 512, 0.1, 5e-4
tf.keras.utils.set_random_seed(42)

# ---------------------------------------------------------------- data + augmentation
y_train_1h = tf.keras.utils.to_categorical(train_labels, 10)
y_test_1h = tf.keras.utils.to_categorical(test_labels, 10)
_yy, _xx = tf.meshgrid(tf.range(32), tf.range(32), indexing="ij")


def augment_r9(img, label):
    img = tf.pad(img, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")   # random crop with 4 px padding
    img = tf.image.random_crop(img, [32, 32, 3])
    img = tf.image.random_flip_left_right(img)
    cy = tf.random.uniform([], 0, 32, tf.int32)                     # Cutout: one 8x8 hole
    cx = tf.random.uniform([], 0, 32, tf.int32)
    keep = tf.logical_not((tf.abs(_yy - cy) < 4) & (tf.abs(_xx - cx) < 4))
    return img * tf.cast(keep, img.dtype)[..., None], label


train_r9 = (tf.data.Dataset.from_tensor_slices((train_images, y_train_1h))
            .shuffle(50000, seed=42).map(augment_r9, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH).prefetch(tf.data.AUTOTUNE))
test_r9 = tf.data.Dataset.from_tensor_slices((test_images, y_test_1h)).batch(1000)


# ---------------------------------------------------------------- model
def build_resnet9():
    reg = regularizers.l2(WD)

    def cbr(x, f):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        return layers.Activation("relu")(x)

    inputs = layers.Input(shape=(32, 32, 3), name="image")
    x = cbr(inputs, 64)                                   # prep        32x32x64
    x = layers.MaxPooling2D(2)(cbr(x, 128))               # layer 1     16x16x128
    x = layers.Add()([x, cbr(cbr(x, 128), 128)])          #   residual
    x = layers.MaxPooling2D(2)(cbr(x, 256))               # layer 2      8x8x256
    x = layers.MaxPooling2D(2)(cbr(x, 512))               # layer 3      4x4x512
    x = layers.Add()([x, cbr(cbr(x, 512), 512)])          #   residual
    x = layers.GlobalMaxPooling2D()(x)                    #            512
    x = layers.Dense(10, kernel_regularizer=reg)(x)
    outputs = layers.Activation("linear", dtype="float32", name="logits")(x)
    return models.Model(inputs, outputs, name="resnet9")


steps = int(np.ceil(50000 / BATCH))
total = EPOCHS * steps


class OneCycle(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up to PEAK_LR over the first 30% of steps, then linear decay to 0."""
    def __call__(self, step):
        s = tf.cast(step, tf.float32)
        up = 0.3 * total
        return tf.where(s < up, PEAK_LR * s / up, PEAK_LR * tf.maximum(0.0, (total - s) / (total - up)))

    def get_config(self):
        return {}


tf.keras.mixed_precision.set_global_policy("mixed_float16")   # T4 tensor cores: ~2-3x faster
r9_mp = build_resnet9()
r9_mp.compile(optimizer=tf.keras.optimizers.SGD(OneCycle(), momentum=0.9, nesterov=True),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
              metrics=["accuracy"])
t0 = time.time()
hist_r9 = r9_mp.fit(train_r9, epochs=EPOCHS, validation_data=test_r9, verbose=2)
r9_minutes = (time.time() - t0) / 60

# rebuild in float32 so TFLite conversion and saving are clean; the weights are float32 already
tf.keras.mixed_precision.set_global_policy("float32")
resnet9 = build_resnet9()
resnet9.set_weights(r9_mp.get_weights())


def acc_of(model, tta=False):
    logits = model.predict(test_images, batch_size=1000, verbose=0)
    if tta:   # test-time augmentation: average with the mirrored image
        logits = logits + model.predict(test_images[:, :, ::-1, :], batch_size=1000, verbose=0)
    return float((logits.argmax(1) == test_labels.flatten()).mean()), logits


r9_acc, _ = acc_of(resnet9)
r9_acc_tta, r9_logits = acc_of(resnet9, tta=True)
results["resnet9"] = {"test_accuracy": round(r9_acc, 4), "test_accuracy_tta": round(r9_acc_tta, 4),
                      "params": int(resnet9.count_params()), "epochs": EPOCHS,
                      "train_minutes": round(r9_minutes, 1),
                      "best_val_accuracy": round(float(max(hist_r9.history["val_accuracy"])), 4)}
print(json.dumps(results["resnet9"], indent=2))
resnet9.summary()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(hist_r9.history["accuracy"], label="train"); ax[0].plot(hist_r9.history["val_accuracy"], label="test")
ax[1].plot(hist_r9.history["loss"], label="train"); ax[1].plot(hist_r9.history["val_loss"], label="test")
ax[0].set_title("ResNet-9 - accuracy"); ax[1].set_title("ResNet-9 - loss (incl. label smoothing + L2)")
for a in ax: a.set_xlabel("Epoch"); a.legend()
plt.tight_layout(); plt.savefig("outputs/resnet9_curves.png", dpi=130); plt.show()

# ---------------------------------------------------------------- comparison + error analysis
names = {"baseline_sgd": "Baseline · SGD", "baseline_adam": "Baseline · Adam",
         "improved_adam": "Improved CNN · Adam", "resnet9": "ResNet-9 · SGD 1-cycle"}
keys = [k for k in names if k in results]
plt.figure(figsize=(8.5, 4.2))
bars = plt.bar([names[k] for k in keys], [results[k]["test_accuracy"] * 100 for k in keys],
               color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
for b in bars:
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}%", ha="center")
plt.ylabel("Test accuracy (%)"); plt.ylim(0, 100); plt.title("CIFAR-10 test accuracy (10,000 images)")
plt.tight_layout(); plt.savefig("outputs/accuracy_comparison.png", dpi=130); plt.show()

y_pred = r9_logits.argmax(1); y_true = test_labels.flatten()
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=10).numpy()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10), class_names, rotation=45, ha="right"); ax.set_yticks(range(10), class_names)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix - ResNet-9 (with flip TTA)")
fig.colorbar(im); plt.tight_layout(); plt.savefig("outputs/confusion_matrix_resnet9.png", dpi=130); plt.show()
per_class = cm.diagonal() / cm.sum(axis=1)
results["per_class_accuracy_resnet9"] = {n: round(float(a), 3) for n, a in zip(class_names, per_class)}
off = cm.copy(); np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
results["most_confused_resnet9"] = [class_names[i], class_names[j], int(off[i, j])]
print("Per-class:", results["per_class_accuracy_resnet9"], "| most confused:", results["most_confused_resnet9"])

resnet9.save("outputs/resnet9.keras")

# ---------------------------------------------------------------- predictions: raw and cropped
OWN_BOXES = {"my_car.jpg": (0.11, 0.41, 0.89, 0.85)}   # car sits low in a tall photo; others: centre square


def crop_square(path, box=None):
    im = Image.open(path).convert("RGB"); W, H = im.size
    if box:
        im = im.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))
    w, h = im.size; s = min(w, h); l, t = (w - s) // 2, (h - s) // 2
    return im.crop((l, t, l + s, t + s))


def robot_roi(path):
    """The e-puck parks with the object centred: fixed square region of interest above the floor."""
    W, H = Image.open(path).size
    side = 0.62 * H; half = side / 2 / W
    return crop_square(path, (0.5 - half, 0.02, 0.5 + half, 0.02 + side / H))


def predict_pil(model, im):
    x = np.asarray(im.convert("RGB").resize((32, 32), Image.BILINEAR), dtype="float32")[None] / 255.0
    p = tf.nn.softmax(model.predict(x, verbose=0)[0]).numpy()
    return [[class_names[i], round(float(p[i]), 3)] for i in p.argsort()[::-1][:3]]


def show(model, items, title, out_png):
    fig, ax = plt.subplots(1, len(items), figsize=(3.6 * len(items), 4.4), squeeze=False)
    found = []
    for k, (name, im) in enumerate(items):
        top3 = predict_pil(model, im)
        ax[0, k].imshow(im); ax[0, k].axis("off")
        ax[0, k].set_title("\n".join(f"{c}: {p:.1%}" for c, p in top3), fontsize=10)
        found.append({"image": name, "top3": top3})
        print(f"{name:<22} -> {top3[0][0]} ({top3[0][1]:.1%})")
    fig.suptitle(title); plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.show()
    return found


own_raw = [(os.path.basename(p), Image.open(p)) for p in own_paths]
own_crop = [(os.path.basename(p), crop_square(p, OWN_BOXES.get(os.path.basename(p)))) for p in own_paths]
web_raw = [(os.path.basename(p), Image.open(p)) for p in webots_paths]
web_crop = [(os.path.basename(p), robot_roi(p)) for p in webots_paths]
results["predictions_resnet9"] = {
    "own_raw": show(resnet9, own_raw, "ResNet-9 - my own images (whole photo)", "outputs/r9_own_raw.png"),
    "own_crop": show(resnet9, own_crop, "ResNet-9 - my own images (object-centred crop)", "outputs/r9_own_crop.png"),
    "webots_raw": show(resnet9, web_raw, "ResNet-9 - e-puck frames (whole frame)", "outputs/r9_webots_raw.png"),
    "webots_crop": show(resnet9, web_crop, "ResNet-9 - e-puck frames (square region of interest)", "outputs/r9_webots_crop.png")}
results["predictions_improved_crop"] = {
    "own_crop": show(improved, own_crop, "Improved CNN - my own images (object-centred crop)", "outputs/imp_own_crop.png"),
    "webots_crop": show(improved, web_crop, "Improved CNN - e-puck frames (square region of interest)", "outputs/imp_webots_crop.png")}

# ---------------------------------------------------------------- robot trade-off: TFLite
if "optimisation" in results and "optimisation_improved" not in results:
    results["optimisation_improved"] = results.pop("optimisation")
opt = {}
for mode in ["fp32", "fp16", "int8"]:
    path = to_tflite(resnet9, mode)
    os.rename(path, path.replace("cifar10_cnn_", "resnet9_"))
    opt[f"tflite_{mode}"] = bench_tflite(path.replace("cifar10_cnn_", "resnet9_"))
results["optimisation_resnet9"] = opt
print("ResNet-9 TFLite:", json.dumps(opt, indent=2))

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=2)
shutil.make_archive("cnn_outputs_stage2", "zip", "outputs")
from google.colab import files as _colab_files
_colab_files.download("cnn_outputs_stage2.zip")
print("DONE stage 2")
