"""
Experiment 8: Fuzzy Sets & Operations
File: fuzzy_operations.py (also available as fuzzy_operation.py)

Topic: Fuzzy Set Operations (Problem 9)
Universe: X = {1, 2, 3, 4}
Sets:
  A = 0.2/1 + 0.3/2 + 0.4/3 + 0.5/4
  B = 0.1/1 + 0.2/2 + 0.2/3 + 1.0/4
Operations:
  (a) Algebraic Sum: mu_A + mu_B - (mu_A * mu_B)
  (b) Algebraic Product: mu_A * mu_B
  (c) Bounded Sum: min(1, mu_A + mu_B)
  (d) Bounded Difference: max(0, mu_A - mu_B)
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

X = np.array([1, 2, 3, 4])
mu_A = np.array([0.2, 0.3, 0.4, 0.5])
mu_B = np.array([0.1, 0.2, 0.2, 1.0])

def display_fuzzy_set(name, mu, universe):
    terms = [f"{m:g} / {x}" for m, x in zip(mu, universe)]
    print(f"{name} = {{ {' + '.join(terms)} }}", flush=True)

print("." * 25, flush=True)
print("FUZZY SETS FOR PROBLEM 9", flush=True)
print("." * 25, flush=True)
display_fuzzy_set("A", mu_A, X)
display_fuzzy_set("B", mu_B, X)

print("." * 55, flush=True)
print("(a) ALGEBRAIC SUM: mu_A + mu_B - (mu_A * mu_B)", flush=True)
print("." * 55, flush=True)
mu_alg_sum = np.round(mu_A + mu_B - (mu_A * mu_B), 4)
for x, a, b, res in zip(X, mu_A, mu_B, mu_alg_sum):
    print(f"At x = {x}: {a} + {b} - ({a} * {b}) = {a + b:.2f} - {a*b:.2f} = {res}", flush=True)
display_fuzzy_set("\nResult (A + B)", mu_alg_sum, X)

print("\n" + "." * 40, flush=True)
print("(b) ALGEBRAIC PRODUCT: mu_A * mu_B", flush=True)
print("." * 40, flush=True)
mu_alg_prod = np.round(mu_A * mu_B, 4)
for x, a, b, res in zip(X, mu_A, mu_B, mu_alg_prod):
    print(f"At x = {x}: {a} * {b} = {res}", flush=True)
display_fuzzy_set("\nResult (A . B)", mu_alg_prod, X)

print("." * 40, flush=True)
print("(c) BOUNDED SUM: min(1, mu_A + mu_B)", flush=True)
print("." * 40, flush=True)
mu_bnd_sum = np.round(np.minimum(1.0, mu_A + mu_B), 4)
for x, a, b, res in zip(X, mu_A, mu_B, mu_bnd_sum):
    print(f"At x = {x}: min(1, {a} + {b}) = min(1, {a + b:.2f}) = {res}", flush=True)
display_fuzzy_set("\nResult (A (+) B)", mu_bnd_sum, X)

print("\n" + "." * 50, flush=True)
print("(d) BOUNDED DIFFERENCE: max(0, mu_A - mu_B)", flush=True)
print("." * 50, flush=True)
mu_bnd_diff = np.round(np.maximum(0.0, mu_A - mu_B), 4)
for x, a, b, res in zip(X, mu_A, mu_B, mu_bnd_diff):
    print(f"At x = {x}: max(0, {a} - {b}) = max(0, {a - b:.2f}) = {res}", flush=True)
display_fuzzy_set("\nResult (A (-) B)", mu_bnd_diff, X)

print("." * 99, flush=True)
print("FINAL SUMMARY TABLE", flush=True)
print("." * 99, flush=True)

if HAS_PANDAS:
    df_summary = pd.DataFrame({
        'Element (x)': X,
        'mu_A(x)': mu_A,
        'mu_B(x)': mu_B,
        'Algebraic Sum': mu_alg_sum,
        'Algebraic Product': mu_alg_prod,
        'Bounded Sum': mu_bnd_sum,
        'Bounded Difference': mu_bnd_diff
    })
    print(df_summary.to_string(index=False), flush=True)
else:
    header = f"{'Element (x)':>11}  {'mu_A(x)':>8}  {'mu_B(x)':>8}  {'Algebraic Sum':>14}  {'Algebraic Product':>18}  {'Bounded Sum':>12}  {'Bounded Difference':>19}"
    print(header, flush=True)
    for x, a, b, s1, s2, s3, s4 in zip(X, mu_A, mu_B, mu_alg_sum, mu_alg_prod, mu_bnd_sum, mu_bnd_diff):
        print(f"{x:>11}  {a:>8.1f}  {b:>8.1f}  {s1:>14.2f}  {s2:>18.2f}  {s3:>12.1f}  {s4:>19.1f}", flush=True)
