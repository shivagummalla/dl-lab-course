# Deep Learning Lab Course - 22AIE304

This repository contains the code, datasets, and outputs for the four-lab
Deep Learning practical series (Perceptron → MLP → Hyperparameter
Optimization → Deep Neural Network), along with a final synthesis.

## Structure

```
.
├── Lab1_Perceptron/
│   ├── code/perceptron.py            # Perceptron from scratch (NumPy)
│   ├── dataset/and_gate.csv          # AND gate truth table
│   └── outputs/                      # error curve, decision boundary, results.txt
│
├── Lab2_MLP/
│   ├── code/mlp_xor.py               # MLP for XOR, Sigmoid vs ReLU, 2 learning rates
│   ├── dataset/xor_gate.csv          # XOR gate truth table
│   └── outputs/                      # loss curves, experiment_results.{json,txt}
│
├── Lab3_Hyperparameter_Optimization/
│   ├── code/hparam_search.py         # Random search: LR, hidden units, batch size, dropout
│   ├── dataset/moons_dataset.csv     # Synthetic non-linear "moons" dataset (1500 samples)
│   └── outputs/                      # hparam_search_results.{json,csv}, best_config.txt, plot
│
├── Lab4_DNN_MNIST/
│   ├── code/dnn_mnist.py             # DNN (2 hidden ReLU layers) for MNIST digit classification
│   ├── dataset/mnist.pkl.gz          # MNIST data (fallback if Google storage is firewalled)
│   └── outputs/                      # confusion_matrix.png, misclassified_samples.png, results_summary.txt
├── requirements.txt

```

## How to run

```bash
pip install -r requirements.txt
python Lab1_Perceptron/code/perceptron.py
python Lab2_MLP/code/mlp_xor.py
python Lab3_Hyperparameter_Optimization/code/hparam_search.py
python Lab4_DNN_MNIST/code/dnn_mnist.py
```

Each script regenerates its own `outputs/` folder contents.

## Notes on datasets

- **Lab 1 & 2** use the classic AND / XOR truth tables (4 samples each) — the
  standard way to demonstrate linear vs. non-linear separability.
- **Lab 3** uses scikit-learn's synthetic `make_moons` dataset (1500 samples,
  noise=0.25) rather than XOR, because batch size and dropout effects need a
  dataset large enough to have a meaningful train/validation split.
- **Lab 4** uses the real MNIST handwritten digit dataset (60,000 train /
  10,000 test images). `tf.keras.datasets.mnist.load_data()` is tried first;
  if that host is unreachable on your network, the script automatically
  falls back to the bundled `dataset/mnist.pkl.gz`.

## Results summary (from actual runs in this repo)

| Lab | Headline result |
|---|---|
| Lab 1 | Perceptron converges on AND gate in 4 epochs — weights ≈ [0.2, 0.1], bias ≈ -0.2 |
| Lab 2 | ReLU + lr=0.1 solves XOR (100% train acc); Sigmoid stalls at 50% in 100 epochs |
| Lab 3 | Best config: lr=0.01, hidden=8, batch=16, dropout=0.0 → 95.0% val accuracy |
| Lab 4 | DNN reaches 96.78% test accuracy on MNIST after 5 epochs |
