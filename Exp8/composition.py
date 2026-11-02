import numpy as np
import pandas as pd

x_labels = ['x1', 'x2']
y_labels = ['y1', 'y2']
z_labels = ['z1', 'z2', 'z3']

R = np.array([
    [0.6, 0.3],
    [0.2, 0.9]
])

S = np.array([
    [1.0, 0.5, 0.3],
    [0.8, 0.4, 0.7]
])

print("." * 45)
print("PROBLEM 3 (a): MAX-MIN COMPOSITION (T = R o S)")
print("." * 45)

T_max_min = np.zeros((R.shape[0], S.shape[1]))

print("\nStep-by-step calculations: mu_T(x, z) = max_y { min[mu_R(x, y), mu_S(y, z)] }")
for i, xi in enumerate(x_labels):
    for k, zk in enumerate(z_labels):
        terms = [min(R[i, j], S[j, k]) for j in range(len(y_labels))]
        result = max(terms)
        T_max_min[i, k] = result
        print(f"mu_T({xi}, {zk}) = max[min({R[i,0]}, {S[0,k]}), min({R[i,1]}, {S[1,k]})] = max({terms[0]}, {terms[1]}) = {result}")

df_T_max_min = pd.DataFrame(T_max_min, index=x_labels, columns=z_labels)
print("\nFinal Max-Min Composition Matrix T:")
print(df_T_max_min)

print("\n" + "." * 55)
print("PROBLEM 3 (b): MAX-PRODUCT COMPOSITION (T = R . S)")
print("." * 55)

T_max_prod = np.zeros((R.shape[0], S.shape[1]))

print("\nStep-by-step calculations: mu_T(x, z) = max_y { mu_R(x, y) * mu_S(y, z) }")
for i, xi in enumerate(x_labels):
    for k, zk in enumerate(z_labels):
        terms = [round(R[i, j] * S[j, k], 4) for j in range(len(y_labels))]
        result = max(terms)
        T_max_prod[i, k] = result
        print(f"mu_T({xi}, {zk}) = max[({R[i,0]} * {S[0,k]}), ({R[i,1]} * {S[1,k]})] = max({terms[0]}, {terms[1]}) = {result}")

df_T_max_prod = pd.DataFrame(T_max_prod, index=x_labels, columns=z_labels)
print("\nFinal Max-Product Composition Matrix T:")
print(df_T_max_prod)