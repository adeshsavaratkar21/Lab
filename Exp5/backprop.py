"""
Experiment 5: Backpropagation Neural Network (BPN)
File: backprop.py
Description: Implementation of Backpropagation Algorithm for a 2-Hidden-Unit, 1-Output-Unit BPN.
             Step-by-step trace of forward pass, error gradients, backward pass, and weight updates.
             Matches Lab Manual format and exact notebook numerical outputs.
"""

import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Fallback for numpy if not installed locally
try:
    import numpy as np
    exp_fn = np.exp
except ImportError:
    exp_fn = math.exp


# Define Binary Sigmoidal Activation Function: f(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1.0 / (1.0 + exp_fn(-x))


# Derivative of Sigmoidal Activation: f'(x) = f(x) * (1 - f(x))
def sigmoid_derivative(output):
    return output * (1.0 - output)


def main():
    # 1: Define activation functions and initialize given problem parameters
    # Given Input Pattern and Desired Target
    x1, x2 = 0.0, 1.0
    target = 1.0

    # Given Learning Rate (alpha)
    alpha = 0.25

    # Given Initial Weights and Biases (Input to Hidden Layer)
    v11, v21, v01 = 0.6, -0.1, 0.3
    v12, v22, v02 = -0.3, 0.4, 0.5

    # Given Initial Weights and Bias (Hidden to Output Layer)
    w1, w2, w0 = 0.4, 0.1, -0.2

    print(" INITIAL NETWORK SETUP ")
    print(f"Inputs: x1 = {x1}, x2 = {x2} | Target = {target} | Learning Rate alpha = {alpha}")
    print(f"Initial Hidden Unit 1 Weights: v01 = {v01}, v11 = {v11}, v21 = {v21}")
    print(f"Initial Hidden Unit 2 Weights: v02 = {v02}, v12 = {v12}, v22 = {v22}")
    print(f"Initial Output Unit Weights:   w0  = {w0}, w1  = {w1}, w2  = {w2}")

    # 2: Calculate net input and output for Hidden Neurons (Z1 and Z2)
    # Net input to Z1: zin1 = v01 + x1*v11 + x2*v21
    zin1 = v01 + (x1 * v11) + (x2 * v21)
    # Output of Z1: z1 = f(zin1)
    z1 = sigmoid(zin1)

    # Net input to Z2: zin2 = v02 + x1*v12 + x2*v22
    zin2 = v02 + (x1 * v12) + (x2 * v22)
    # Output of Z2: z2 = f(zin2)
    z2 = sigmoid(zin2)

    print("\n STEP 1: FORWARD PASS (HIDDEN LAYER) ")
    print(f"Hidden Unit Z1: zin1 = {zin1:.4f}, z1 = {z1:.4f}")
    print(f"Hidden Unit Z2: zin2 = {zin2:.4f}, z2 = {z2:.4f}")

    # 3: Calculate net input and output for Output Neuron (Y)
    # Net input to Y: yin = w0 + z1*w1 + z2*w2
    yin = w0 + (z1 * w1) + (z2 * w2)
    # Actual Output of Y: y = f(yin)
    y = sigmoid(yin)

    print("\n STEP 2: FORWARD PASS (OUTPUT LAYER) ")
    print(f"Net Input entering Output Layer (yin): {yin:.5f}")
    print(f"Calculated Network Output (y):          {y:.4f}")
    print(f"Error (Target - Calculated):           {target - y:.4f}")

    # 4: Calculate Output Error Gradient (delta_k) and changes in weights (delta_w)
    # Derivative of sigmoid at output unit: f'(yin) = y * (1 - y)
    f_prime_y = sigmoid_derivative(y)
    # Output error factor: delta1 = (target - y) * f'(yin)
    delta_output = (target - y) * f_prime_y

    # Weight correction terms between Hidden and Output layer
    delta_w1 = alpha * delta_output * z1
    delta_w2 = alpha * delta_output * z2
    delta_w0 = alpha * delta_output       # Bias correction

    print("\n STEP 3: BACKWARD PASS (OUTPUT LAYER) ")
    print(f"Derivative f'(yin) = {f_prime_y:.4f}")
    print(f"Output Error Term delta_k (delta1) = {delta_output:.4f}")
    print(f"Weight Changes: Delta w1 = {delta_w1:.5f}, Delta w2 = {delta_w2:.5f}, Delta w0 = {delta_w0:.5f}")

    # 5: Propagate error back to hidden units and calculate changes in weights (delta_v)
    # Net error coming into hidden units from output: delta_inj = delta_k * w_j
    delta_in1 = delta_output * w1
    delta_in2 = delta_output * w2

    # Derivatives for hidden units: f'(zin) = z * (1 - z)
    f_prime_z1 = sigmoid_derivative(z1)
    f_prime_z2 = sigmoid_derivative(z2)

    # Error terms for hidden units: delta_j = delta_inj * f'(zin_j)
    delta_h1 = delta_in1 * f_prime_z1
    delta_h2 = delta_in2 * f_prime_z2

    # Weight corrections for Hidden Unit Z1 links
    delta_v11 = alpha * delta_h1 * x1
    delta_v21 = alpha * delta_h1 * x2
    delta_v01 = alpha * delta_h1       # Bias correction for Z1

    # Weight corrections for Hidden Unit Z2 links
    delta_v12 = alpha * delta_h2 * x1
    delta_v22 = alpha * delta_h2 * x2
    delta_v02 = alpha * delta_h2       # Bias correction for Z2

    print("\n STEP 4: BACKWARD PASS (HIDDEN LAYER) ")
    print(f"Hidden Unit 1 Error (delta_1): {delta_h1:.5f}")
    print(f"Hidden Unit 2 Error (delta_2): {delta_h2:.5f}")
    print(f"Weight Changes for Z1: Delta v11 = {delta_v11:.5f}, Delta v21 = {delta_v21:.5f}, Delta v01 = {delta_v01:.5f}")
    print(f"Weight Changes for Z2: Delta v12 = {delta_v12:.7f}, Delta v22 = {delta_v22:.7f}, Delta v02 = {delta_v02:.7f}")

    # 6: Compute new weights and print comparison table
    # Update Hidden-to-Output weights
    w1_new = w1 + delta_w1
    w2_new = w2 + delta_w2
    w0_new = w0 + delta_w0

    # Update Input-to-Hidden weights
    v11_new = v11 + delta_v11
    v21_new = v21 + delta_v21
    v01_new = v01 + delta_v01

    v12_new = v12 + delta_v12
    v22_new = v22 + delta_v22
    v02_new = v02 + delta_v02

    print("\n" + "=" * 65)
    print("FINAL UPDATED WEIGHTS AND BIASES")
    print("=" * 65)
    print(f"{'Parameter':<15} | {'Old Value':<12} | {'Delta':<12} | {'New Value':<12}")
    print("-" * 65)
    print(f"{'w1 (Z1 -> Y)':<15} | {w1:<12.4f} | {delta_w1:<12.5f} | {w1_new:<12.4f}")
    print(f"{'w2 (Z2 -> Y)':<15} | {w2:<12.4f} | {delta_w2:<12.5f} | {w2_new:<12.5f}")
    print(f"{'w0 (Bias Y)':<15} | {w0:<12.4f} | {delta_w0:<12.5f} | {w0_new:<12.5f}")
    print("-" * 65)
    print(f"{'v11 (X1 -> Z1)':<15} | {v11:<12.4f} | {delta_v11:<12.5f} | {v11_new:<12.4f}")
    print(f"{'v21 (X2 -> Z1)':<15} | {v21:<12.4f} | {delta_v21:<12.5f} | {v21_new:<12.5f}")
    print(f"{'v01 (Bias Z1)':<15} | {v01:<12.4f} | {delta_v01:<12.5f} | {v01_new:<12.5f}")
    print("-" * 65)
    print(f"{'v12 (X1 -> Z2)':<15} | {v12:<12.4f} | {delta_v12:<12.7f} | {v12_new:<12.4f}")
    print(f"{'v22 (X2 -> Z2)':<15} | {v22:<12.4f} | {delta_v22:<12.7f} | {v22_new:<12.7f}")
    print(f"{'v02 (Bias Z2)':<15} | {v02:<12.4f} | {delta_v02:<12.7f} | {v02_new:<12.7f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
