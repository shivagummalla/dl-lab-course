import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset: XOR gate (NOT linearly separable)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([0, 1, 1, 0], dtype=np.float32)


def build_model(activation, hidden_units=8):
    model = models.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(hidden_units, activation=activation),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model


def run_experiment(activation, lr, epochs=100):
    tf.keras.utils.set_random_seed(42)
    model = build_model(activation)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(X, y, epochs=epochs, verbose=0)
    final_loss = history.history["loss"][-1]
    final_acc = history.history["accuracy"][-1]
    return final_loss, final_acc, history.history


if __name__ == "__main__":
    experiments = [
        ("Sigmoid", "sigmoid", 0.01),
        ("Sigmoid", "sigmoid", 0.1),
        ("ReLU", "relu", 0.01),
        ("ReLU", "relu", 0.1),
    ]

    results = []
    histories = {}
    for name, act, lr in experiments:
        final_loss, final_acc, hist = run_experiment(act, lr, epochs=100)
        results.append({
            "Activation": name,
            "Learning Rate": lr,
            "Final Loss": round(float(final_loss), 4),
            "Final Accuracy": round(float(final_acc), 4),
        })
        histories[f"{name}_lr{lr}"] = hist["loss"]
        print(f"{name:7s} | lr={lr:<5} | Final Loss={final_loss:.4f} | Final Acc={final_acc:.4f}")

    # Save results table
    with open(os.path.join(OUTPUT_DIR, "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "experiment_results.txt"), "w") as f:
        f.write(f"{'Exp#':<5}{'Activation':<12}{'LR':<8}{'Final Loss':<14}{'Final Acc':<10}\n")
        for i, r in enumerate(results, 1):
            f.write(f"{i:<5}{r['Activation']:<12}{r['Learning Rate']:<8}{r['Final Loss']:<14}{r['Final Accuracy']:<10}\n")

    # Plot loss curves for comparison
    plt.figure(figsize=(7, 5))
    for label, loss_curve in histories.items():
        plt.plot(loss_curve, label=label)
    plt.title("Training Loss Curves: Sigmoid vs ReLU (XOR)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Binary Crossentropy)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nOutputs saved to {OUTPUT_DIR}")
