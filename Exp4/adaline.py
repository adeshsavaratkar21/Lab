"""
Experiment 4: Single Layer Perceptron & Adaline Network
File: adaline.py
Description: Implementation of Adaline (Adaptive Linear Neuron) Network using Delta Rule (Widrow-Hoff)
             for Bipolar OR Gate. Matches Lab Manual format and table outputs.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import numpy as np
except ImportError:
    class _NP:
        def array(self, obj):
            return obj
    np = _NP()


# 1: Adaline Quantizer / Bipolar Activation Function
def adaline_activation(yin):
    """
    Quantizer / Step function used for final classification:
    y = +1  if yin >= 0
    y = -1  if yin < 0
    """
    return 1 if yin >= 0 else -1


def main():
    # 2: Dataset and Initial Parameters
    # Bipolar OR Gate inputs: [x1, x2]
    X = np.array([
        [ 1,  1],
        [ 1, -1],
        [-1,  1],
        [-1, -1]
    ])
    # Bipolar Target outputs (t): Only (-1, -1) is -1; all others are +1
    T = np.array([1, 1, 1, -1])

    # Initial parameters matching the lab manual
    w1 = 0.1
    w2 = 0.1
    b = 0.1
    alpha = 0.1

    print("Adaline OR Gate Dataset:")
    print(f"{'x1':<5} {'x2':<5} | {'t':<4}")
    print("-" * 18)
    for (x1_val, x2_val), t_val in zip(X, T):
        print(f"{x1_val:<5} {x2_val:<5} | {t_val:<4}")

    # 3: Training Using Delta Rule (Widrow-Hoff Learning Rule)
    print("\n" + "=" * 115)
    print("OR GATE USING ADALINE LEARNING RULE")
    print("=" * 115)
    num_epochs = 2  # Set number of training epochs

    # Table header formatted according to lab manual specifications
    header = f"{'EPOCH':<7} | {'x1':<4} {'x2':<4} {'t':<4} | {'yin':<9} {'t - yin':<9} | {'Δw1':<8} {'Δw2':<8} {'Δb':<8} | {'w1':<8} {'w2':<8} {'b':<8} | {'(t - yin)^2':<10}"
    print(header)
    print("-" * 115)

    for epoch in range(1, num_epochs + 1):
        total_squared_error = 0.0
        for i in range(len(X)):
            x1_in = X[i][0]
            x2_in = X[i][1]
            t = T[i]

            # 1. Linear net input: yin = b + x1*w1 + x2*w2
            yin = b + (x1_in * w1) + (x2_in * w2)

            # 2. Continuous linear error (t - yin)
            error = t - yin
            squared_error = error ** 2
            total_squared_error += squared_error

            # 3. Delta rule parameter updates: delta_w = alpha * (t - yin) * x
            delta_w1 = alpha * error * x1_in
            delta_w2 = alpha * error * x2_in
            delta_b  = alpha * error

            # 4. Update weights and bias
            w1 += delta_w1
            w2 += delta_w2
            b  += delta_b

            # Print sample step details
            print(f"{epoch:<7} | {x1_in:<4} {x2_in:<4} {t:<4} | {yin:<9.4f} {error:<9.4f} | {delta_w1:<8.4f} {delta_w2:<8.4f} {delta_b:<8.4f} | {w1:<8.4f} {w2:<8.4f} {b:<8.4f} | {squared_error:<10.4f}")

        print("-" * 115)
        print(f"Total Mean Squared Error after Epoch {epoch}: {total_squared_error:.4f}")
        print("-" * 115)

    print(f"\nTraining completed for {num_epochs} Epochs.")
    print(f"Weights after Epoch {num_epochs}: w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")

    # 4: Testing the Trained Adaline Model
    print("\n--- TESTING TRAINED ADALINE NETWORK FOR OR GATE ---")
    print(f"{'x1':<5} {'x2':<5} {'Target (t)':<12} {'yin':<10} {'Predicted (y)':<15} {'Status'}")
    print("-" * 65)

    all_correct = True
    for i in range(len(X)):
        x1_in = X[i][0]
        x2_in = X[i][1]
        t = T[i]

        # Net input using final trained weights
        yin = b + (x1_in * w1) + (x2_in * w2)

        # Adaline quantizer output
        y = adaline_activation(yin)

        status = "Correct" if y == t else "Incorrect"
        if y != t:
            all_correct = False

        print(f"{x1_in:<5} {x2_in:<5} {t:<12} {yin:<10.4f} {y:<15} {status}")

    print("-" * 65)
    if all_correct:
        print("Verification Successful: The Adaline network correctly classifies all OR patterns.")


if __name__ == "__main__":
    main()
