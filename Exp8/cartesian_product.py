"""
Experiment 8: Fuzzy Sets & Relations
File: cartesian_product.py

Topic: Fuzzy Cartesian Product (R = A x B)
Universe: X = {x1, x2, x3}, Y = {y1, y2}
Membership:
  A = 0.3/x1 + 0.7/x2 + 1.0/x3
  B = 0.4/y1 + 0.9/y2
Relation:
  mu_R(x, y) = min{mu_A(x), mu_B(y)}
"""

import sys

# Configure stdout to safely handle UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

x_labels = ['x1', 'x2', 'x3']
y_labels = ['y1', 'y2']
mu_A = np.array([0.3, 0.7, 1.0])
mu_B = np.array([0.4, 0.9])

print("=" * 65, flush=True)
print("PROBLEM 2: FUZZY CARTESIAN PRODUCT (R = A x B)", flush=True)
print("=" * 65, flush=True)

R = np.zeros((len(mu_A), len(mu_B)))

print("\nStep-by-step calculations: mu_R(x, y) = min{mu_A(x), mu_B(y)}", flush=True)
for i, xi in enumerate(x_labels):
    for j, yj in enumerate(y_labels):
        val = min(mu_A[i], mu_B[j])
        R[i, j] = val
        print(f"mu_R({xi}, {yj}) = min({mu_A[i]}, {mu_B[j]}) = {val}", flush=True)

print("\nFinal Cartesian Product Relation Matrix R:", flush=True)
if HAS_PANDAS:
    df_R = pd.DataFrame(R, index=x_labels, columns=y_labels)
    print(df_R, flush=True)
else:
    print(f"     {'  '.join(y_labels)}", flush=True)
    for xi, row in zip(x_labels, R):
        vals = "  ".join(f"{v:.1f}" for v in row)
        print(f"{xi}  {vals}", flush=True)
