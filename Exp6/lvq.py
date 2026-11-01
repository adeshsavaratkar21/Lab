"""
Experiment 6: Competitive Learning Networks
File: lvq.py
Description: Learning Vector Quantization (LVQ) simulation.
             Matches Lab Manual format and exact notebook numerical outputs.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np


def main():
    print("=" * 65)
    print("LEARNING VECTOR QUANTIZATION (LVQ) SIMULATION")
    print("=" * 65)

    # Reference Weight Vectors (prototypes) assigned to Class 1 and Class 2
    # w1 is prototype for Class 1; w2 is prototype for Class 2
    W = np.array([
        [0.0, 0.0, 1.0, 1.0],  # Prototype 1 (Class 1)
        [1.0, 0.0, 0.0, 0.0]   # Prototype 2 (Class 2)
    ], dtype=float)

    weight_classes = [1, 2]
    alpha = 0.1

    # Input training vectors with assigned target classes (Table 1)
    input_patterns = [
        {"x": np.array([0.0, 0.0, 0.0, 1.0]), "target_class": 2},
        {"x": np.array([1.0, 1.0, 0.0, 0.0]), "target_class": 1},
        {"x": np.array([0.0, 1.0, 1.0, 0.0]), "target_class": 1}
    ]

    print(f"Initial w1 (Class 1): {W[0]}")
    print(f"Initial w2 (Class 2): {W[1]}")
    print(f"Learning rate alpha: {alpha}\n")

    # Sequential presentation of training patterns
    for idx, pattern in enumerate(input_patterns, start=1):
        x = pattern["x"]
        target = pattern["target_class"]
        print(f" Pattern {idx}: x = {x.astype(int)}, Target Class T = {target} ")

        # Calculate squared Euclidean distances to all prototypes
        D = np.sum((W - x) ** 2, axis=1)
        print(f"D(1) = {D[0]:.4f}, D(2) = {D[1]:.4f}")

        # Determine winner index
        winner_idx = np.argmin(D)
        winner_unit = winner_idx + 1
        predicted_class = weight_classes[winner_idx]
        print(f"Winner Unit: J = {winner_unit} (Predicted Class = {predicted_class})")

        # Apply supervised LVQ weight update rule
        if predicted_class == target:
            print("Outcome: Target Matches Winner (T == J). Moving prototype closer (+)")
            W[winner_idx] = W[winner_idx] + alpha * (x - W[winner_idx])
        else:
            print("Outcome: Target Mismatch (T != J). Pushing prototype away (-)")
            W[winner_idx] = W[winner_idx] - alpha * (x - W[winner_idx])

        print(f"Updated w1: {np.round(W[0], 4)}")
        print(f"Updated w2: {np.round(W[1], 4)}\n")

    # Output final weight matrix
    print("=" * 65)
    print("FINAL UPDATED LVQ WEIGHT MATRIX:")
    print("Features (rows) x Prototypes [w1, w2] (cols):")
    print(np.round(W.T, 4))


if __name__ == "__main__":
    main()
