"""
Experiment 6: Competitive Learning Networks
File: lvq.py
Description: Learning Vector Quantization (LVQ) implementation in pure Python.
             Supervised competitive learning algorithm using prototype reference vectors.
             Includes reward update (pull closer) and penalty update (push away).
"""

import math
from typing import List, Tuple


class LearningVectorQuantization:
    def __init__(self, weight_vectors: List[List[float]], class_labels: List[int], lr: float = 0.2):
        """
        :param weight_vectors: Initial reference codebook vectors
        :param class_labels: Target class associated with each reference vector
        :param lr: Initial learning rate (alpha)
        """
        self.weights = [list(w) for w in weight_vectors]
        self.classes = list(class_labels)
        self.initial_lr = lr
        self.lr = lr

    def euclidean_distance_sq(self, x: List[float], w: List[float]) -> float:
        return sum((xi - wi) ** 2 for xi, wi in zip(x, w))

    def find_winner(self, x: List[float]) -> Tuple[int, int, float]:
        """Finds closest prototype vector index J, its assigned class, and distance."""
        best_idx = 0
        min_dist = float('inf')

        for j, w in enumerate(self.weights):
            dist = self.euclidean_distance_sq(x, w)
            if dist < min_dist:
                min_dist = dist
                best_idx = j

        return best_idx, self.classes[best_idx], math.sqrt(min_dist)

    def train(self, X: List[List[float]], y: List[int], epochs: int = 50, decay_rate: float = 0.95):
        """
        Trains LVQ reference vectors.
        - If winning class == target: w_J = w_J + alpha * (x - w_J)
        - If winning class != target: w_J = w_J - alpha * (x - w_J)
        """
        print(f"\n--- Training LVQ Network (Initial Alpha={self.initial_lr}, Epochs={epochs}) ---")
        print("Initial Reference Vectors:")
        for j, (w, c) in enumerate(zip(self.weights, self.classes)):
            print(f"  Prototype W{j} (Class {c}): {[round(v, 4) for v in w]}")
        print("-" * 70)

        for epoch in range(1, epochs + 1):
            for x_vec, target_class in zip(X, y):
                winner_idx, winner_class, _ = self.find_winner(x_vec)

                # LVQ Update Rule
                if winner_class == target_class:
                    # Reward: pull prototype closer to input sample
                    for i in range(len(x_vec)):
                        self.weights[winner_idx][i] += self.lr * (x_vec[i] - self.weights[winner_idx][i])
                else:
                    # Penalty: push prototype away from wrongly classified sample
                    for i in range(len(x_vec)):
                        self.weights[winner_idx][i] -= self.lr * (x_vec[i] - self.weights[winner_idx][i])

            # Learning rate decay
            self.lr = self.initial_lr * (1.0 - (epoch / epochs))

        print("Final Trained Reference Vectors:")
        for j, (w, c) in enumerate(zip(self.weights, self.classes)):
            print(f"  Prototype W{j} (Class {c}): {[round(v, 4) for v in w]}")
        print("-" * 70)

    def predict(self, x: List[float]) -> int:
        _, pred_class, _ = self.find_winner(x)
        return pred_class


def demo_lvq():
    print("=" * 70)
    print("         LEARNING VECTOR QUANTIZATION (LVQ) CLASSIFICATION     ")
    print("=" * 70)

    # Training Data (4-dimensional feature vectors with binary target classes 0 or 1)
    X_train = [
        [0.0, 0.0, 1.0, 1.0],  # Class 0
        [1.0, 1.0, 0.0, 0.0],  # Class 1
        [0.0, 0.0, 0.0, 1.0],  # Class 0
        [1.0, 0.0, 0.0, 0.0],  # Class 1
        [0.0, 1.0, 1.0, 1.0],  # Class 0
        [1.0, 1.0, 1.0, 0.0]   # Class 1
    ]
    y_train = [0, 1, 0, 1, 0, 1]

    # Initial reference prototypes (one chosen for each class)
    initial_prototypes = [
        [0.0, 0.0, 1.0, 1.0],  # Prototype for Class 0
        [1.0, 1.0, 0.0, 0.0]   # Prototype for Class 1
    ]
    prototype_classes = [0, 1]

    lvq = LearningVectorQuantization(initial_prototypes, prototype_classes, lr=0.3)
    lvq.train(X_train, y_train, epochs=25)

    # Test samples
    X_test = [
        [0.0, 0.0, 1.0, 0.0],  # Expected Class 0
        [1.0, 0.5, 0.0, 0.0],  # Expected Class 1
        [0.2, 0.1, 0.9, 0.8],  # Expected Class 0
        [0.9, 0.8, 0.1, 0.2]   # Expected Class 1
    ]
    expected_test_y = [0, 1, 0, 1]

    print("\nTesting Trained LVQ Model on Unseen Samples:")
    print(f"{'Test Sample':<24} | {'Expected':<10} | {'Predicted':<10} | {'Status'}")
    print("-" * 70)
    for x_test, exp_y in zip(X_test, expected_test_y):
        pred_y = lvq.predict(x_test)
        status = "CORRECT" if pred_y == exp_y else "WRONG"
        print(f"{str(x_test):<24} | Class {exp_y:<4} | Class {pred_y:<4} | {status}")
    print("=" * 70)


if __name__ == "__main__":
    demo_lvq()
