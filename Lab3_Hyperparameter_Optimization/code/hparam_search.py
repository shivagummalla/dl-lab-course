import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Dataset: synthetic "moons" dataset with noise (non-linearly separable)

X, y = make_moons(n_samples=1500, noise=0.25, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save dataset to CSV
df = pd.DataFrame(X, columns=["x1", "x2"])
df["label"] = y
df.to_csv(os.path.join(DATASET_DIR, "moons_dataset.csv"), index=False)


def build_model(hidden_units, dropout_rate):
    model = models.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model


def run_trial(lr, hidden_units, batch_size, dropout_rate, epochs=60):
    tf.keras.utils.set_random_seed(42)
    model = build_model(hidden_units, dropout_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, verbose=0
    )
    train_loss = history.history["loss"][-1]
    train_acc = history.history["accuracy"][-1]
    val_loss = history.history["val_loss"][-1]
    val_acc = history.history["val_accuracy"][-1]
    overfit_gap = train_acc - val_acc
    overfitting = "Yes" if overfit_gap > 0.07 else "No"
    return train_loss, train_acc, val_loss, val_acc, overfitting


if __name__ == "__main__":
    # 5 hyperparameter combinations (Random Search style)
    trials = [
        {"lr": 0.01, "hidden_units": 8,  "batch_size": 16,  "dropout_rate": 0.0},
        {"lr": 0.01, "hidden_units": 32, "batch_size": 16,  "dropout_rate": 0.5},
        {"lr": 0.1,  "hidden_units": 16, "batch_size": 64,  "dropout_rate": 0.2},
        {"lr": 0.001,"hidden_units": 64, "batch_size": 256, "dropout_rate": 0.0},
        {"lr": 0.01, "hidden_units": 32, "batch_size": 32,  "dropout_rate": 0.3},
    ]

    results = []
    for i, t in enumerate(trials, 1):
        train_loss, train_acc, val_loss, val_acc, overfit = run_trial(**t)
        row = {
            "Trial": i,
            "Learning Rate": t["lr"],
            "Hidden Units": t["hidden_units"],
            "Batch Size": t["batch_size"],
            "Dropout": t["dropout_rate"],
            "Training Loss": round(float(train_loss), 4),
            "Validation Accuracy": round(float(val_acc), 4),
            "Overfitting?": overfit,
        }
        results.append(row)
        print(row)

    with open(os.path.join(OUTPUT_DIR, "hparam_search_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "hparam_search_results.csv"), index=False)

    best = results_df.loc[results_df["Validation Accuracy"].idxmax()]
    with open(os.path.join(OUTPUT_DIR, "best_config.txt"), "w") as f:
        f.write("Best configuration:\n")
        f.write(str(best))

    # Plot Validation Accuracy per trial
    plt.figure(figsize=(7, 4))
    plt.bar(results_df["Trial"].astype(str), results_df["Validation Accuracy"])
    plt.xlabel("Trial #")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy across Hyperparameter Trials")
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(OUTPUT_DIR, "val_accuracy_per_trial.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\nBest config:\n", best)
    print(f"\nOutputs saved to {OUTPUT_DIR}")
