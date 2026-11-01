"""
Experiment 5: Backpropagation Neural Network (BPN)
File: backprop.py
Description: Implementation of Multi-Layer Perceptron (MLP) trained with the
             Error Backpropagation algorithm using pure Python standard library.
             Successfully learns non-linear decision boundaries (solving the XOR problem).
"""

import math
import random
from typing import List, Tuple


def sigmoid(x: float) -> float:
    """Standard logistic sigmoid activation function."""
    # Clip x to prevent math range overflow
    if x < -700:
        return 0.0
    if x > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(output: float) -> float:
    """Derivative of sigmoid given its activated output value: f'(x) = f(x) * (1 - f(x))."""
    return output * (1.0 - output)


class BackpropagationNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float = 0.5, momentum: float = 0.1, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.momentum = momentum

        random.seed(seed)

        # Initialize weights with small random values between -0.5 and 0.5
        self.w_ih = [[random.uniform(-0.5, 0.5) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b_h = [random.uniform(-0.5, 0.5) for _ in range(hidden_dim)]

        self.w_ho = [[random.uniform(-0.5, 0.5) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b_o = [random.uniform(-0.5, 0.5) for _ in range(output_dim)]

        # Momentum change tracking
        self.prev_delta_w_ih = [[0.0] * hidden_dim for _ in range(input_dim)]
        self.prev_delta_b_h = [0.0] * hidden_dim
        self.prev_delta_w_ho = [[0.0] * output_dim for _ in range(hidden_dim)]
        self.prev_delta_b_o = [0.0] * output_dim

    def forward(self, x: List[float]) -> Tuple[List[float], List[float]]:
        """
        Forward Propagation:
        Computes hidden layer activations and output layer activations.
        """
        # Hidden layer
        hidden_acts = []
        for j in range(self.hidden_dim):
            net = self.b_h[j] + sum(x[i] * self.w_ih[i][j] for i in range(self.input_dim))
            hidden_acts.append(sigmoid(net))

        # Output layer
        output_acts = []
        for k in range(self.output_dim):
            net = self.b_o[k] + sum(hidden_acts[j] * self.w_ho[j][k] for j in range(self.hidden_dim))
            output_acts.append(sigmoid(net))

        return hidden_acts, output_acts

    def backward(self, x: List[float], targets: List[float], hidden_acts: List[float], output_acts: List[float]):
        """
        Backward Error Propagation:
        Calculates error gradients and updates weights using gradient descent + momentum.
        """
        # 1. Output Layer Error Term: delta_o_k = (t_k - y_k) * y_k * (1 - y_k)
        delta_o = []
        for k in range(self.output_dim):
            error = targets[k] - output_acts[k]
            delta_o.append(error * sigmoid_derivative(output_acts[k]))

        # 2. Hidden Layer Error Term: delta_h_j = sum(delta_o_k * w_ho_jk) * h_j * (1 - h_j)
        delta_h = []
        for j in range(self.hidden_dim):
            sum_err = sum(delta_o[k] * self.w_ho[j][k] for k in range(self.output_dim))
            delta_h.append(sum_err * sigmoid_derivative(hidden_acts[j]))

        # 3. Update Hidden-to-Output Weights
        for j in range(self.hidden_dim):
            for k in range(self.output_dim):
                dw = self.lr * delta_o[k] * hidden_acts[j] + self.momentum * self.prev_delta_w_ho[j][k]
                self.w_ho[j][k] += dw
                self.prev_delta_w_ho[j][k] = dw

        for k in range(self.output_dim):
            db = self.lr * delta_o[k] + self.momentum * self.prev_delta_b_o[k]
            self.b_o[k] += db
            self.prev_delta_b_o[k] = db

        # 4. Update Input-to-Hidden Weights
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                dw = self.lr * delta_h[j] * x[i] + self.momentum * self.prev_delta_w_ih[i][j]
                self.w_ih[i][j] += dw
                self.prev_delta_w_ih[i][j] = dw

        for j in range(self.hidden_dim):
            db = self.lr * delta_h[j] + self.momentum * self.prev_delta_b_h[j]
            self.b_h[j] += db
            self.prev_delta_b_h[j] = db

    def train(self, X: List[List[float]], Y: List[List[float]], epochs: int = 5000, log_every: int = 500):
        print(f"\n--- Starting BPN Training ({epochs} Epochs, LR={self.lr}) ---")
        for epoch in range(1, epochs + 1):
            total_loss = 0.0

            for x_vec, target_vec in zip(X, Y):
                hidden_acts, output_acts = self.forward(x_vec)
                # Mean Squared Error: sum((t - y)^2) / 2
                loss = sum((t - o) ** 2 for t, o in zip(target_vec, output_acts)) * 0.5
                total_loss += loss
                self.backward(x_vec, target_vec, hidden_acts, output_acts)

            mse = total_loss / len(X)
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                print(f"Epoch {epoch:5d}/{epochs} | MSE Loss: {mse:.6f}")

    def predict(self, x: List[float]) -> List[float]:
        _, output_acts = self.forward(x)
        return output_acts


def solve_xor_problem():
    print("=" * 65)
    print("      BACKPROPAGATION NETWORK (BPN) - SOLVING XOR PROBLEM     ")
    print("=" * 65)
    print("Architecture: 2 Inputs -> 2 Hidden Neurons -> 1 Output Neuron")
    print("Activation  : Logistic Sigmoid: f(x) = 1 / (1 + e^-x)")

    # XOR Dataset
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    Y = [
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ]

    bpn = BackpropagationNN(input_dim=2, hidden_dim=3, output_dim=1, lr=0.8, momentum=0.2, seed=42)
    bpn.train(X, Y, epochs=6000, log_every=1000)

    print("\n" + "=" * 65)
    print("                 FINAL EVALUATION / PREDICTIONS               ")
    print("=" * 65)
    print(f"{'Input (X1, X2)':<16} | {'Target':<8} | {'Raw Output':<14} | {'Rounded':<8} | {'Status'}")
    print("-" * 65)

    all_correct = True
    for x_vec, target in zip(X, Y):
        raw_pred = bpn.predict(x_vec)[0]
        rounded_pred = 1 if raw_pred >= 0.5 else 0
        target_val = int(target[0])
        status = "PASSED" if rounded_pred == target_val else "FAILED"
        if status == "FAILED":
            all_correct = False
        print(f"({x_vec[0]:.0f}, {x_vec[1]:.0f}){' ' * 9} | {target_val:<8} | {raw_pred:<14.6f} | {rounded_pred:<8} | {status}")

    print("-" * 65)
    print(f"Overall Result: {'All 4 cases matched successfully!' if all_correct else 'Not fully converged.'}")
    print("=" * 65)


if __name__ == "__main__":
    solve_xor_problem()
