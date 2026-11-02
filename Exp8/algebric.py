import numpy as np
import pandas as pd

X = np.array([1, 2, 3, 4])
mu_A = np.array([0.2, 0.3, 0.4, 0.5])
mu_B = np.array([0.1, 0.2, 0.2, 1.0])

def display_fuzzy_set(name, mu, universe):
    terms = [f"{m:g} / {x}" for m, x in zip(mu, universe)]
    print(f"{name} = {{ {' + '.join(terms)} }}")

print("." * 25)
print("FUZZY SETS FOR PROBLEM 9")
print("." * 25)
display_fuzzy_set("A", mu_A, X)
display_fuzzy_set("B", mu_B, X)

print("." * 55)
print("(a) ALGEBRAIC SUM: mu_A + mu_B - (mu_A * mu_B)")
print("." * 55)
mu_alg_sum = np.round(mu_A + mu_B - (mu_A * mu_B), 4)
for x, a, b, res in zip(X, mu_A, mu_B, mu_alg_sum):
    print(f"At x = {x}: {a} + {b} - ({a} * {b}) = {a + b:.2f} - {a*b:.2f} = {res}")
display_fuzzy_set("\nResult (A + B)", mu_alg_sum, X)

print("\n" + "." * 40)
print("(b) ALGEBRAIC PRODUCT: mu_A * mu_B")
print("." * 40)
mu_alg_prod = np.round(mu_A * mu_B, 4)
for x, a, b, res in zip(X, mu_A, mu_B, mu_alg_prod):
    print(f"At x = {x}: {a} * {b} = {res}")
display_fuzzy_set("\nResult (A . B)", mu_alg_prod, X)

print("." * 40)
print("(c) BOUNDED SUM: min(1, mu_A + mu_B)")
print("." * 40)
mu_bnd_sum = np.round(np.minimum(1.0, mu_A + mu_B), 4)
for x, a, b, res in zip(X, mu_A, mu_B, mu_bnd_sum):
    print(f"At x = {x}: min(1, {a} + {b}) = min(1, {a + b:.2f}) = {res}")
display_fuzzy_set("\nResult (A (+) B)", mu_bnd_sum, X)

print("\n" + "." * 50)
print("(d) BOUNDED DIFFERENCE: max(0, mu_A - mu_B)")
print("." * 50)
mu_bnd_diff = np.round(np.maximum(0.0, mu_A - mu_B), 4)
for x, a, b, res in zip(X, mu_A, mu_B, mu_bnd_diff):
    print(f"At x = {x}: max(0, {a} - {b}) = max(0, {a - b:.2f}) = {res}")
display_fuzzy_set("\nResult (A (-) B)", mu_bnd_diff, X)

df_summary = pd.DataFrame({
    'Element (x)': X,
    'mu_A(x)': mu_A,
    'mu_B(x)': mu_B,
    'Algebraic Sum': mu_alg_sum,
    'Algebraic Product': mu_alg_prod,
    'Bounded Sum': mu_bnd_sum,
    'Bounded Difference': mu_bnd_diff
})
print("." * 99)
print("FINAL SUMMARY TABLE")
print("." * 99)
print(df_summary.to_string(index=False))