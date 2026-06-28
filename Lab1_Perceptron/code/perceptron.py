import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Dataset: AND gate (linearly separable)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 0, 0, 1])   # AND gate truth table


def train_perceptron(X, y, lr=0.1, epochs=10, verbose=True):
    """
    Trains a single-layer Perceptron using the classic update rule:
        w <- w + lr * (y_true - y_pred) * x
        b <- b + lr * (y_true - y_pred)
    """
    weights = np.zeros(X.shape[1])
    bias = 0.0
    errors_per_epoch = []

    for epoch in range(epochs):
        total_errors = 0
        for i in range(len(X)):
            # 1. Linear combination
            linear_output = np.dot(X[i], weights) + bias

            # 2. Heaviside step activation
            y_pred = 1 if linear_output >= 0 else 0

            # 3. Error * learning rate
            update = lr * (y[i] - y_pred)

            # 4. Weight & bias update
            weights += update * X[i]
            bias += update

            total_errors += int(update != 0)

        errors_per_epoch.append(total_errors)
        if verbose:
            print(f"Epoch {epoch+1:2d}: weights={weights}, bias={bias:.2f}, errors={total_errors}")

        # Early stop if fully converged
        if total_errors == 0:
            if verbose:
                print(f"Converged after {epoch+1} epochs.")
            break

    return weights, bias, errors_per_epoch


if __name__ == "__main__":
    weights, bias, errors = train_perceptron(X, y, lr=0.1, epochs=10)

    print("\nFinal weights:", weights)
    print("Final bias:", bias)

    # Predictions on training data
    print("\nPredictions:")
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        pred = 1 if linear_output >= 0 else 0
        print(f"Input: {X[i]} -> Predicted: {pred}, Actual: {y[i]}")

    
    # Plot 1: Error curve across epochs

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(errors) + 1), errors, marker="o")
    plt.title("Perceptron: Misclassifications per Epoch (AND gate)")
    plt.xlabel("Epoch")
    plt.ylabel("Number of misclassifications")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "error_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()


    # Plot 2: Decision boundary

    plt.figure(figsize=(5, 5))
    for i in range(len(X)):
        color = "blue" if y[i] == 1 else "red"
        plt.scatter(X[i][0], X[i][1], c=color, s=120, edgecolors="k",
                    label=f"Class {y[i]}" if i == 0 or y[i] != y[i - 1] else None)

    x_vals = np.linspace(-0.5, 1.5, 100)
    if weights[1] != 0:
        y_vals = -(weights[0] * x_vals + bias) / weights[1]
        plt.plot(x_vals, y_vals, "g--", label="Decision boundary")

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Perceptron Decision Boundary - AND Gate")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "decision_boundary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save a text log of the results
    with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
        f.write("Dataset: AND gate\n")
        f.write(f"Final weights: {weights}\n")
        f.write(f"Final bias: {bias}\n")
        f.write(f"Epochs to converge: {len(errors)}\n")
        f.write(f"Errors per epoch: {errors}\n")
        f.write(f"Decision boundary equation: {weights[0]:.2f}*x1 + {weights[1]:.2f}*x2 + {bias:.2f} = 0\n")

    print(f"\nOutputs saved to {OUTPUT_DIR}")
