"""
Experiment 4: Single Layer Perceptron & Adaline Network
File: perceptron.py
Description: Implementation of Single Layer Perceptron Learning Algorithm for Bipolar AND Gate.
             Matches Lab Manual format and table outputs.
"""

import sys

# Configure stdout for UTF-8 on Windows with safe fallback
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Determine safe Delta symbol (uses Greek Delta if terminal supports it, else ASCII 'd')
try:
    "Δ".encode(sys.stdout.encoding or 'ascii')
    DELTA = "Δ"
except Exception:
    DELTA = "d"

try:
    import numpy as np
except ImportError:
    class _NP:
        def array(self, obj):
            return obj
    np = _NP()


# 1: Define the Bipolar Step Activation Function
def activation_function(yin):
    """
    Activation function:
    y =  1  if yin > 0
    y =  0  if yin == 0
    y = -1  if yin < 0
    """
    if yin > 0:
        return 1
    elif yin == 0:
        return 0
    else:
        return -1


def main():
    # 2: Training Data and Initialization
    # AND Gate with Bipolar Inputs (x1, x2) and Bipolar Target (t)
    X = np.array([
        [ 1,  1],
        [ 1, -1],
        [-1,  1],
        [-1, -1]
    ])
    # Desired Targets (t)
    T = np.array([1, -1, -1, -1])

    # Initial parameters
    w1 = 0
    w2 = 0
    b = 0
    alpha = 1

    print("Dataset initialized successfully:")
    print("x1  x2  |  t")
    print("-" * 14)
    for (x1_val, x2_val), t_val in zip(X, T):
        print(f"{x1_val:2d}  {x2_val:2d}  | {t_val:2d}")

    # 3: Training the Perceptron Network
    print("=" * 80)
    print("EXP: TO DESIGN SINGLE LAYER PERCEPTRON USING PYTHON")
    print("OUTPUT: AND GATE USING PERCEPTRON LEARNING RULE")
    print("=" * 80)
    header = f"{'EPOCH':<7} | {'x1':<3} {'x2':<3} {'t':<3} | {'yin':<5} {'y':<3} | {DELTA + 'w1':<4} {DELTA + 'w2':<4} {DELTA + 'b':<4} | {'w1':<4} {'w2':<4} {'b':<4}"
    print(header)
    print("-" * 80)

    epoch = 0
    converged = False
    while not converged:
        epoch += 1
        weight_changed = False
        for i in range(len(X)):
            x1_in = X[i][0]
            x2_in = X[i][1]
            t = T[i]

            # Step 1: Net input calculation: yin = b + x1*w1 + x2*w2
            yin = b + (x1_in * w1) + (x2_in * w2)

            # Step 2: Apply activation function
            y = activation_function(yin)

            # Step 3: Check if target equals calculated output
            if t != y:
                # Weight update rule
                delta_w1 = alpha * t * x1_in
                delta_w2 = alpha * t * x2_in
                delta_b  = alpha * t

                # New weights and bias
                w1 = w1 + delta_w1
                w2 = w2 + delta_w2
                b  = b + delta_b
                weight_changed = True
            else:
                # No change in weights
                delta_w1 = 0
                delta_w2 = 0
                delta_b  = 0

            # Display row matching lab manual table
            print(f"{epoch:<7} | {x1_in:3d} {x2_in:3d} {t:3d} | {yin:5d} {y:3d} | {delta_w1:4d} {delta_w2:4d} {delta_b:4d} | {w1:4d} {w2:4d} {b:4d}")

        print("-" * 80)
        # Stop condition: an epoch with zero weight updates
        if not weight_changed:
            converged = True

    print(f"\nTraining Converged in {epoch} Epochs!")
    print(f"Final Weights: w1 = {w1}, w2 = {w2}")
    print(f"Final Bias:    b = {b}")
    print(f"Decision Boundary Equation: {w1}*x1 + {w2}*x2 + ({b}) = 0")

    # 4: Testing the Trained Perceptron
    print("\n--- TESTING TRAINED PERCEPTRON FOR AND GATE ---")
    print(f"{'x1':<6} {'x2':<6} {'yin':<6} {'Calculated (y)':<16} {'Target (t)':<12} {'Status'}")
    print("-" * 65)
    for i in range(len(X)):
        x1_in = X[i][0]
        x2_in = X[i][1]
        t = T[i]
        yin = b + (x1_in * w1) + (x2_in * w2)
        y = activation_function(yin)
        status = "Correct" if y == t else "Error"
        print(f"{x1_in:<6} {x2_in:<6} {yin:<6} {y:<16} {t:<12} {status}")


if __name__ == "__main__":
    main()
