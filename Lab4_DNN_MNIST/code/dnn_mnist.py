import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

tf.keras.utils.set_random_seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_mnist():
    """Try the standard Keras loader first, fall back to local pickle."""
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        return x_train, y_train, x_test, y_test
    except Exception as e:
        print(f"Keras MNIST download failed ({e}); falling back to local mnist.pkl.gz")
        pkl_path = os.path.join(DATASET_DIR, "mnist.pkl.gz")
        with gzip.open(pkl_path, "rb") as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        x_train = train_set[0].reshape(-1, 28, 28) * 255.0
        y_train = train_set[1]
        x_test = test_set[0].reshape(-1, 28, 28) * 255.0
        y_test = test_set[1]
        return x_train.astype(np.uint8), y_train, x_test.astype(np.uint8), y_test


# 1. Load and Preprocess Data

(x_train, y_train), (x_test, y_test) = (None, None), (None, None)
x_train, y_train, x_test, y_test = load_mnist()

# TODO (filled in): Normalize pixel values (0-255) to range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Train shape:", x_train.shape, "Test shape:", x_test.shape)

# 2. Build the DNN Architecture

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    # TODO (filled in): two dense hidden layers with ReLU activation
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

# 3. Compile the Model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=5, batch_size=32, verbose=2
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

# Part 2: Confusion Matrix

y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - MNIST DNN")
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(range(10))
plt.yticks(range(10))
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

# Identify the most confused digit pairs (off-diagonal)
cm_copy = cm.copy().astype(float)
np.fill_diagonal(cm_copy, 0)
top_confusions = []
flat_indices = np.dstack(np.unravel_index(np.argsort(-cm_copy.ravel()), cm_copy.shape))[0][:5]
for true_d, pred_d in flat_indices:
    if cm_copy[true_d, pred_d] > 0:
        top_confusions.append((int(true_d), int(pred_d), int(cm_copy[true_d, pred_d])))

print("\nTop confused digit pairs (true -> predicted : count):")
for t, p, c in top_confusions:
    print(f"  {t} -> {p} : {c}")

# Part 3: Error Analysis - find 3 misclassified examples

misclassified_idx = np.where(y_pred != y_test)[0]
sample_errors = misclassified_idx[:3]

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
error_log = []
for ax, idx in zip(axes, sample_errors):
    ax.imshow(x_test[idx], cmap="gray")
    ax.set_title(f"True:{y_test[idx]} Pred:{y_pred[idx]}")
    ax.axis("off")
    error_log.append({
        "Image ID (test index)": int(idx),
        "True Label": int(y_test[idx]),
        "Predicted Label": int(y_pred[idx]),
    })
plt.suptitle("Sample Misclassified Digits")
plt.savefig(os.path.join(OUTPUT_DIR, "misclassified_samples.png"), dpi=150, bbox_inches="tight")
plt.close()

# Save summary results
with open(os.path.join(OUTPUT_DIR, "results_summary.txt"), "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write("\nTop confused digit pairs (true -> predicted : count):\n")
    for t, p, c in top_confusions:
        f.write(f"  {t} -> {p} : {c}\n")
    f.write("\nMisclassified sample examples:\n")
    for e in error_log:
        f.write(f"  {e}\n")
    f.write("\nFull Confusion Matrix:\n")
    f.write(str(cm))

print(f"\nOutputs saved to {OUTPUT_DIR}")
