"""
Experiment 6: Competitive Learning Networks
File: som.py
Description: Kohonen Self-Organizing Feature Map (KSOM) simulation.
             Matches Lab Manual format and exact notebook numerical outputs.
"""

import sys

# Configure UTF-8 stdout if possible
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np

# 1: Kohonen Self-Organizing Feature Map (KSOM)
# Initial Weight Matrix: rows = features (x1, x2), cols = cluster units (Y1 to Y5)
W = np.array([
    [0.3, 0.2, 0.1, 0.8, 0.4],
    [0.5, 0.6, 0.7, 0.9, 0.2]
], dtype=float)

print("=" * 65, flush=True)
print("KOHONEN SELF-ORGANIZING FEATURE MAP (KSOM) SIMULATION", flush=True)
print("=" * 65, flush=True)
print("Initial Weight Matrix W (2 x 5):\n", W, flush=True)

# Part (a): take input (0.2, 0.4) with alpha = 0.2
x_a = np.array([0.2, 0.4])
alpha_a = 0.2
print("\n Input Vector = (0.2, 0.4), alpha = 0.2 ", flush=True)

# Compute squared Euclidean distance D(j) for each cluster unit
D_a = np.sum((W - x_a[:, np.newaxis]) ** 2, axis=0)
for j in range(len(D_a)):
    print(f"D({j + 1}) = ({W[0, j]:.1f} - {x_a[0]:.1f})^2 + ({W[1, j]:.1f} - {x_a[1]:.1f})^2 = {D_a[j]:.4f}", flush=True)

# Identify winning cluster unit J
winner_a = np.argmin(D_a)  # 0-indexed: 0 corresponds to Y1
print(f"\nWinning Unit: Y{winner_a + 1} with minimum D = {D_a[winner_a]:.4f}", flush=True)

# Update weights of the winner
W[:, winner_a] = W[:, winner_a] + alpha_a * (x_a - W[:, winner_a])
print("Updated Weight Matrix after Part (a):\n", np.round(W, 4), flush=True)

# Part (b): take input (0.6, 0.6) with alpha = 0.1
W = np.array([
    [0.3, 0.2, 0.1, 0.8, 0.4],
    [0.5, 0.6, 0.7, 0.9, 0.2]
], dtype=float)

x_b = np.array([0.6, 0.6])
alpha_b = 0.1
print("\n Input Vector = (0.6, 0.6), alpha = 0.1 ", flush=True)

D_b = np.sum((W - x_b[:, np.newaxis]) ** 2, axis=0)
for j in range(len(D_b)):
    print(f"D({j + 1}) = ({W[0, j]:.1f} - {x_b[0]:.1f})^2 + ({W[1, j]:.1f} - {x_b[1]:.1f})^2 = {D_b[j]:.4f}", flush=True)

winner_b = np.argmin(D_b)
print(f"\nWinning Unit: Y{winner_b + 1} with minimum D = {D_b[winner_b]:.4f}", flush=True)

# Update weights of the winner
W[:, winner_b] = W[:, winner_b] + alpha_b * (x_b - W[:, winner_b])
print("Updated Weight Matrix after Part (b):\n", np.round(W, 4), flush=True)
