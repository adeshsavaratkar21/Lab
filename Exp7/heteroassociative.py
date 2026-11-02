"""
Experiment 7: Associative Memory Networks
Part 2: Heteroassociative Memory Network

Topic: Heteroassociative Network for Pattern Association (S -> T)
Training Pairs:
Pair 1: s(1) = [1, 0, 0, 0]  -->  t(1) = [0, 1]
Pair 2: s(2) = [1, 1, 0, 0]  -->  t(2) = [0, 1]
Pair 3: s(3) = [0, 0, 0, 1]  -->  t(3) = [1, 0]
Pair 4: s(4) = [0, 0, 1, 1]  -->  t(4) = [1, 0]
"""

import sys

# Configure stdout to safely handle UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np

# Training Pairs (S -> T)
S = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1]
])

T = np.array([
    [0, 1],
    [0, 1],
    [1, 0],
    [1, 0]
])

P = len(S)

print(" TRAINING PAIRS (S -> T) ", flush=True)
for p in range(P):
    print(f"Pair {p+1}: s({p+1}) = {S[p]}  -->  t({p+1}) = {T[p]}", flush=True)

n = S.shape[1]
m = T.shape[1]

W = np.zeros((n, m), dtype=int)

print("\n STEP-BY-STEP OUTER PRODUCT ACCUMULATION ", flush=True)
for p in range(P):
    s_col = S[p].reshape(-1, 1)
    t_row = T[p].reshape(1, -1)
    W_p = np.dot(s_col, t_row)
    W += W_p
    print(f"\nOuter Product W_{p+1} for pair {p+1}:", flush=True)
    print(W_p, flush=True)

print("\n" + "." * 42, flush=True)
print("FINAL ACCUMULATED WEIGHT MATRIX W (4x2):", flush=True)
print(W, flush=True)
print("." * 42, flush=True)

def binary_step_activation(yin):
    """
    Standard binary step activation function for heteroassociative net:
    f(yin) = 1 if yin > 0
    f(yin) = 0 if yin <= 0
    """
    output = np.zeros_like(yin)
    for i in range(len(yin)):
        if yin[i] > 0:
            output[i] = 1
        else:
            output[i] = 0
    return output

def evaluate_hetero_net(x_test, target_expected, description):
    """
    Performs forward pass and verifies retrieved association.
    """
    y_in = np.dot(x_test, W)
    y_out = binary_step_activation(y_in)
    
    is_correct = np.array_equal(y_out, target_expected) if target_expected is not None else False
    status = "SUCCESS (Correctly Recalled)" if is_correct else "FAILURE / MISMATCH"
    
    print(f"\n {description} ", flush=True)
    print(f"Test Input (x)       : {x_test}", flush=True)
    print(f"Net Input (y_in)     : {y_in}", flush=True)
    print(f"Retrieved Output (y) : {y_out}", flush=True)
    if target_expected is not None:
        print(f"Expected Target (t)  : {target_expected}", flush=True)
        print(f"Verification Result  : {status}", flush=True)
        
    return y_out

print("\n TESTING WITH ORIGINAL TRAINING INPUTS ", flush=True)
for p in range(P):
    evaluate_hetero_net(
        x_test=S[p],
        target_expected=T[p],
        description=f"Testing with Training Pattern {p+1}"
    )

