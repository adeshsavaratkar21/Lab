"""
Experiment 7: Associative Memory Networks
Part 3: Bidirectional Associative Memory (BAM) Network

Topic: BAM Network for Pattern Association using 5x3 Bipolar Character Grids
Pattern 'E' (15-dim bipolar vector)  --> Target: [-1,  1]
Pattern 'F' (15-dim bipolar vector)  --> Target: [ 1,  1]
"""

import sys

# Configure stdout to safely handle UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np

# Training Pattern Specifications (5x3 Grids for Characters 'E' and 'F')
grid_E = np.array([
    [ 1,  1,  1],
    [ 1, -1, -1],
    [ 1,  1,  1],
    [ 1, -1, -1],
    [ 1,  1,  1]
])

grid_F = np.array([
    [ 1,  1,  1],
    [ 1, -1, -1],
    [ 1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1, -1]
])

s1 = grid_E.flatten()  
s2 = grid_F.flatten()  

t1 = np.array([-1,  1])  
t2 = np.array([ 1,  1]) 

print(" TRAINING PATTERN SPECIFICATIONS ", flush=True)
print(f"Pattern 'E' vector s1 (shape {s1.shape}):\n{s1}", flush=True)
print(f"Target 'E' vector t1: {t1}\n", flush=True)
print(f"Pattern 'F' vector s2 (shape {s2.shape}):\n{s2}", flush=True)
print(f"Target 'F' vector t2: {t2}", flush=True)

# Weight Matrix calculation
s1_col = s1.reshape(-1, 1)  
t1_row = t1.reshape(1, -1)  
s2_col = s2.reshape(-1, 1)  
t2_row = t2.reshape(1, -1)  

W1 = np.dot(s1_col, t1_row)  
W2 = np.dot(s2_col, t2_row)  
W = W1 + W2

print("\nTotal BAM Weight Matrix W (Shape: 15 x 2):", flush=True)
print("Row: [w_i1, w_i2] for i = 1 to 15", flush=True)
print(W, flush=True)

def bam_activation(net_input, prev_state=None):
    """
    Standard bipolar activation function with threshold theta = 0:
    f(net) = +1 if net > 0
    f(net) = -1 if net < 0
    f(net) = prev_state if net == 0 (retains previous state)
    """
    output = np.zeros_like(net_input, dtype=int)
    for i in range(len(net_input)):
        if net_input[i] > 0:
            output[i] = 1
        elif net_input[i] < 0:
            output[i] = -1
        else:
            # Retain previous state if net input is zero
            output[i] = prev_state[i] if prev_state is not None else 0
    return output

def test_forward_x_to_y(x_input, expected_target, pattern_name):
    y_in = np.dot(x_input, W)
    y_out = bam_activation(y_in)
    
    is_correct = np.array_equal(y_out, expected_target)
    status = "SUCCESS (Correct Target Recalled)" if is_correct else "FAILURE"
    
    print(f"\n Forward Test: Pattern '{pattern_name}' (X -> Y) ", flush=True)
    print(f"Net Input y_in     : {y_in}", flush=True)
    print(f"Retrieved Target y : {y_out}", flush=True)
    print(f"Expected Target t  : {expected_target}", flush=True)
    print(f"Result             : {status}", flush=True)
    return y_out

# Test with Pattern 'E'
test_forward_x_to_y(s1, t1, "E")

# Test with Pattern 'F'
test_forward_x_to_y(s2, t2, "F")
