"""
Experiment 7: Associative Memory Networks
Part 1: Autoassociative Memory Network

Topic: Autoassociative Network for Pattern Storage and Noise-Tolerant Recall
Stored Vector: s = [-1, 1, 1, 1]
"""

import sys

# Configure stdout to safely handle UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np

# Training Phase
s = np.array([-1, 1, 1, 1])

print(" TRAINING PHASE ", flush=True)
print(f"Stored input vector s: {s}", flush=True)

s_col = s.reshape(-1, 1)   
s_row = s.reshape(1, -1)   

W = np.dot(s_col, s_row)

print("\nWeight Matrix W (4x4) computed via outer product:", flush=True)
print(W, flush=True)

# Activation Function and Verification Function
def bipolar_activation(yin):
    """
    f(yin) = +1 if yin > 0
    f(yin) = -1 if yin < 0
    f(yin) =  0 if yin == 0
    """
    output = np.zeros_like(yin)
    for i in range(len(yin)):
        if yin[i] > 0:
            output[i] = 1
        elif yin[i] < 0:
            output[i] = -1
        else:
            output[i] = 0  
    return output

def test_pattern(x_test, description):
    """
    Evaluates net input, applies activation, and checks recall.
    """
    y_in = np.dot(x_test, W)
    y_out = bipolar_activation(y_in)
    
    is_recalled = np.array_equal(y_out, s)
    status = "SUCCESS (Correctly Recalled)" if is_recalled else "FAILURE (Not Recognized)"
    
    print(f"\n {description} ", flush=True)
    print(f"Test Vector x      : {x_test}", flush=True)
    print(f"Net Input y_in     : {y_in}", flush=True)
    print(f"Retrieved Output y : {y_out}", flush=True)
    print(f"Outcome            : {status}", flush=True)
    
    return y_out

# Testing with Original Vector and Missing / Mistaken Entries
test_pattern(np.array([-1, 1, 1, 1]), "Case 1: Same Input Vector")
test_pattern(np.array([0, 1, 1, 1]), "Case 2A: One Missing Entry (Index 0 = 0)")
test_pattern(np.array([-1, 1, 0, 1]), "Case 2B: One Missing Entry (Index 2 = 0)")
test_pattern(np.array([-1, -1, 1, 1]), "Case 3: One Mistake Entry (Index 1 flipped to -1)")
test_pattern(np.array([0, 0, 1, 1]), "Case 4A: Two Missing Entries (Indices 0, 1 = 0)")
test_pattern(np.array([-1, 0, 0, 1]), "Case 4B: Two Missing Entries (Indices 1, 2 = 0)")
test_pattern(np.array([-1, -1, -1, 1]), "Case 5: Two Mistaken Entries (Indices 1, 2 flipped to -1)")

