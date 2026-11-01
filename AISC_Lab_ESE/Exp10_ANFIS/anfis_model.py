"""
Experiment 10: Adaptive Neuro-Fuzzy Inference System (ANFIS)
File: anfis_model.py
Description: Implementation of Jang's 5-Layer Adaptive Neuro-Fuzzy Inference System (ANFIS)
             with a first-order Takagi-Sugeno-Kang (TSK) fuzzy model in pure Python.
             Architecture:
               Layer 1: Fuzzification (Gaussian Membership Functions - Premise Parameters)
               Layer 2: Rule Firing Strengths (Product T-norm)
               Layer 3: Normalized Firing Strengths
               Layer 4: Consequent Polynomials (Linear TSK functions - Consequent Parameters)
               Layer 5: Overall Output Summation
             Includes gradient descent learning for premise and consequent parameters.
"""

import math
import random
from typing import List, Tuple


def gaussian_mf(x: float, c: float, sigma: float) -> float:
    """Gaussian membership function: exp(-((x - c)^2) / (2 * sigma^2))"""
    sigma = max(1e-4, sigma)
    return math.exp(-((x - c) ** 2) / (2 * (sigma ** 2)))


class ANFIS:
    def __init__(self, num_rules: int = 4, lr: float = 0.05, seed: int = 42):
        self.num_rules = num_rules
        self.lr = lr
        random.seed(seed)

        # Layer 1 Premise parameters for 2 inputs: (mean c, spread sigma)
        # 2 membership functions per input -> 4 rules
        self.c_x = [random.uniform(-0.5, 0.5) for _ in range(num_rules)]
        self.sigma_x = [random.uniform(0.5, 1.5) for _ in range(num_rules)]

        self.c_y = [random.uniform(-0.5, 0.5) for _ in range(num_rules)]
        self.sigma_y = [random.uniform(0.5, 1.5) for _ in range(num_rules)]

        # Layer 4 Consequent parameters: f_i = p_i * x + q_i * y + r_i
        self.p = [random.uniform(-0.5, 0.5) for _ in range(num_rules)]
        self.q = [random.uniform(-0.5, 0.5) for _ in range(num_rules)]
        self.r = [random.uniform(-0.5, 0.5) for _ in range(num_rules)]

    def forward(self, x: float, y: float) -> Tuple[float, List[float], List[float], List[float]]:
        """
        Computes forward pass through all 5 ANFIS layers:
        Returns: overall_output, mu_x, mu_y, norm_w, f_vals
        """
        # Layer 1: Premise / Fuzzification
        mu_x = [gaussian_mf(x, self.c_x[i], self.sigma_x[i]) for i in range(self.num_rules)]
        mu_y = [gaussian_mf(y, self.c_y[i], self.sigma_y[i]) for i in range(self.num_rules)]

        # Layer 2: Rule Firing Strength (T-norm: product)
        w = [mu_x[i] * mu_y[i] for i in range(self.num_rules)]
        sum_w = sum(w)
        if sum_w == 0.0:
            sum_w = 1e-8

        # Layer 3: Normalized Firing Strengths
        norm_w = [w[i] / sum_w for i in range(self.num_rules)]

        # Layer 4: Consequent Functions
        f_vals = [self.p[i] * x + self.q[i] * y + self.r[i] for i in range(self.num_rules)]

        # Layer 5: Overall Output Summation
        overall_output = sum(norm_w[i] * f_vals[i] for i in range(self.num_rules))

        return overall_output, norm_w, f_vals

    def train_step(self, x: float, y: float, target: float):
        """Gradient descent backpropagation through ANFIS layers."""
        y_pred, norm_w, f_vals = self.forward(x, y)
        error = y_pred - target  # dE/dy_pred for MSE loss 0.5 * (y_pred - target)^2

        # 1. Update Consequent Parameters (Layer 4)
        for i in range(self.num_rules):
            dE_df = error * norm_w[i]
            self.p[i] -= self.lr * dE_df * x
            self.q[i] -= self.lr * dE_df * y
            self.r[i] -= self.lr * dE_df

        # 2. Update Premise Parameters (Layer 1)
        for i in range(self.num_rules):
            dE_dnorm_w = error * f_vals[i]

            mu_xi = gaussian_mf(x, self.c_x[i], self.sigma_x[i])
            mu_yi = gaussian_mf(y, self.c_y[i], self.sigma_y[i])

            # Gradient wrt mean c and spread sigma
            sigma_x = max(1e-4, self.sigma_x[i])
            sigma_y = max(1e-4, self.sigma_y[i])

            grad_c_x = (x - self.c_x[i]) / (sigma_x ** 2)
            grad_c_y = (y - self.c_y[i]) / (sigma_y ** 2)

            # Update premises
            self.c_x[i] -= self.lr * 0.1 * dE_dnorm_w * mu_xi * grad_c_x
            self.c_y[i] -= self.lr * 0.1 * dE_dnorm_w * mu_yi * grad_c_y

    def train(self, dataset: List[Tuple[float, float, float]], epochs: int = 500, log_every: int = 100):
        print(f"\n--- Training ANFIS Model ({epochs} Epochs, {self.num_rules} Rules) ---")
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for x, y, target in dataset:
                pred, _, _ = self.forward(x, y)
                total_loss += 0.5 * ((pred - target) ** 2)
                self.train_step(x, y, target)

            mse = total_loss / len(dataset)
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                print(f"Epoch {epoch:4d}/{epochs} | MSE Loss: {mse:.6f}")

    def predict(self, x: float, y: float) -> float:
        pred, _, _ = self.forward(x, y)
        return pred


def demo_anfis():
    print("=" * 70)
    print("    ADAPTIVE NEURO-FUZZY INFERENCE SYSTEM (ANFIS) MODEL       ")
    print("=" * 70)
    print("Architecture: 5 Layers (Jang, 1993)")
    print("Model Type  : First-Order Takagi-Sugeno-Kang (TSK) FIS")
    print("Approximating Non-linear Surface Function: z = 0.5 * x^2 + 0.3 * y + 0.2")

    # Generate Synthetic Training Dataset on [-1.0, 1.0] domain
    train_data = []
    for x_int in range(-4, 5, 2):
        for y_int in range(-4, 5, 2):
            x = x_int / 4.0
            y = y_int / 4.0
            z = 0.5 * (x ** 2) + 0.3 * y + 0.2
            train_data.append((x, y, z))

    anfis = ANFIS(num_rules=4, lr=0.1, seed=24)
    anfis.train(train_data, epochs=600, log_every=100)

    # Test on unseen evaluation points
    test_points = [
        (-0.75, 0.25),
        (0.25, -0.50),
        (0.50, 0.50),
        (0.00, 0.00)
    ]

    print("\n" + "=" * 70)
    print("                  ANFIS TEST SET EVALUATION                   ")
    print("=" * 70)
    print(f"{'Input (x, y)':<20} | {'True Target':<14} | {'ANFIS Output':<14} | {'Abs Error'}")
    print("-" * 70)

    for x_t, y_t in test_points:
        true_z = 0.5 * (x_t ** 2) + 0.3 * y_t + 0.2
        pred_z = anfis.predict(x_t, y_t)
        err = abs(true_z - pred_z)
        print(f"({x_t:+.2f}, {y_t:+.2f}){' ' * 8} | {true_z:<14.4f} | {pred_z:<14.4f} | {err:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    demo_anfis()
