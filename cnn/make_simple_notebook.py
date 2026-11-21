"""Write cnn/CNN_Simple.ipynb: the CNN in 5 short, clearly numbered steps for the presentation."""
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip().splitlines(True)}


cells = [
    md("""
# CNN on CIFAR-10 — 5 steps
**Harish A · Applied Robotics and AI Project, Part 2**

| Step | What it shows |
|---|---|
| 1 | Load the CIFAR-10 images |
| 2 | Build the CNN and **show all its layers** |
| 3 | Train it (graph: training vs validation) |
| 4 | **Accuracy score** on 10,000 test images |
| 5 | **Predictions**: one of my photos + one e-puck camera image from Webots |

Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**.
"""),
    md("## STEP 1 — Load the CIFAR-10 images"),
    code("""
import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train[:, 0], y_test[:, 0]                                # labels as a flat list
x_train, x_test = (x_train / 255).astype("float32"), (x_test / 255).astype("float32")   # pixels 0-255 -> 0-1
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print("training images:", x_train.shape, "  test images:", x_test.shape)
"""),
    md("## STEP 2 — Build the CNN (ResNet-9) and show all its layers"),
    code("""
tf.keras.mixed_precision.set_global_policy("mixed_float16")  # faster on the T4 GPU
L2 = tf.keras.regularizers.l2(5e-4)                          # weight decay

def conv(x, filters):                                       # convolution + batch norm + ReLU
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=L2)(x)
    return layers.ReLU()(layers.BatchNormalization(momentum=0.9)(x))

inputs = layers.Input((32, 32, 3))
x = conv(inputs, 64)
x = layers.MaxPooling2D()(conv(x, 128))
x = layers.Add()([x, conv(conv(x, 128), 128)])              # residual block 1 (skip connection)
x = layers.MaxPooling2D()(conv(x, 256))
x = layers.MaxPooling2D()(conv(x, 512))
x = layers.Add()([x, conv(conv(x, 512), 512)])              # residual block 2 (skip connection)
x = layers.GlobalMaxPooling2D()(x)
outputs = layers.Activation("linear", dtype="float32")(layers.Dense(10, kernel_regularizer=L2)(x))  # 10 scores
model = models.Model(inputs, outputs, name="ResNet9")
model.summary()
"""),
    md("## STEP 3 — Train the CNN\nTwo graphs, each with two lines: **training** (images it learns from) and **validation** (new images). "
       "The lines come together as it learns; a big gap would mean overfitting."),
    code("""
def augment(img, label):                                     # random crop + flip + Cutout (8x8 hole)
    img = tf.image.random_crop(tf.pad(img, [[4, 4], [4, 4], [0, 0]], "REFLECT"), [32, 32, 3])
    img = tf.image.random_flip_left_right(img)
    cy = tf.random.uniform([], 0, 32, tf.int32)
    cx = tf.random.uniform([], 0, 32, tf.int32)
    yy, xx = tf.meshgrid(tf.range(32), tf.range(32), indexing="ij")
    hole = (tf.abs(yy - cy) < 4) & (tf.abs(xx - cx) < 4)
    return img * tf.cast(~hole, img.dtype)[..., None], label

EPOCHS, STEPS = 30, 98                                       # 50,000 images / batch 512 = 98 steps
train_data = (tf.data.Dataset.from_tensor_slices((x_train, tf.one_hot(y_train, 10)))
              .shuffle(50000).map(augment).batch(512).prefetch(2))
class OneCycle(tf.keras.optimizers.schedules.LearningRateSchedule):  # learning rate: up to 0.1, then down to 0
    def __call__(self, step):
        s, total = tf.cast(step, tf.float32), EPOCHS * STEPS
        return tf.where(s < 0.3 * total, 0.1 * s / (0.3 * total), 0.1 * tf.maximum(0.0, (total - s) / (0.7 * total)))
    def get_config(self): return {}
model.compile(optimizer=tf.keras.optimizers.SGD(OneCycle(), momentum=0.9, nesterov=True),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
              metrics=["accuracy"])
history = model.fit(train_data, epochs=EPOCHS, validation_data=(x_test, tf.one_hot(y_test, 10)), verbose=2)

plt.figure(figsize=(12, 4))
for i, name in enumerate(["accuracy", "loss"]):                # left: accuracy, right: loss
    plt.subplot(1, 2, i + 1)
    plt.plot(history.history[name], label="training")
    plt.plot(history.history["val_" + name], label="validation")
    plt.xlabel("epoch"); plt.ylabel(name); plt.title("Training vs validation " + name); plt.legend()
plt.show()
"""),
    md("## STEP 4 — Accuracy score"),
    code("""
loss, accuracy = model.evaluate(x_test, tf.one_hot(y_test, 10), verbose=0)
print(f"Test accuracy on 10,000 new images: {accuracy:.1%}")
"""),
    md("## STEP 5 — Predict two images\nClick ▶ and pick **one of my photos** and **one e-puck camera image from Webots**."),
    code("""
from google.colab import files
from PIL import Image, ImageOps

def square(img):                                            # crop the object into a square, like CIFAR
    W, H = img.size
    if (W, H) == (320, 240):                                # e-puck camera: square above the floor
        s = int(0.62 * H); return img.crop(((W - s) // 2, int(0.02 * H), (W + s) // 2, int(0.02 * H) + s))
    s = min(W, H); return img.crop(((W - s) // 2, (H - s) // 2, (W + s) // 2, (H + s) // 2))

names = list(files.upload())
plt.figure(figsize=(9, 4.5))
for i, name in enumerate(names):
    img = square(ImageOps.exif_transpose(Image.open(name)).convert("RGB"))
    x = np.asarray(img.resize((32, 32)), dtype="float32")[None] / 255.0
    p = tf.nn.softmax(model.predict(x, verbose=0)[0]).numpy()
    plt.subplot(1, len(names), i + 1); plt.imshow(img); plt.axis("off")
    plt.title(f"{CLASSES[p.argmax()]} {p.max():.0%}", fontsize=14)
plt.show()
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).parent / "CNN_Simple.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
