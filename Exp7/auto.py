import numpy as np

# =====================================================================
# EXPERIMENT 7A: AUTOASSOCIATIVE MEMORY NETWORK
# Stored Vector: s = [-1, 1, 1, 1]
# =====================================================================

# 1. Training Phase: Weight Matrix Computation via Outer Products Rule
s = np.array([-1, 1, 1, 1])

print("=" * 65)
print("             AUTOASSOCIATIVE NETWORK SIMULATION")
print("=" * 65)
print(f"Stored input vector s: {s}")

s_col = s.reshape(-1, 1)   # Shape (4, 1)
s_row = s.reshape(1, -1)   # Shape (1, 4)

W = np.dot(s_col, s_row)   # Outer product: W = s^T * s

print("\nWeight Matrix W (4x4) computed via Outer Products Rule:")
print(W)
print("=" * 65)

# 2. Activation Function & Testing Pipeline
def bipolar_activation(yin):
    """
    Standard bipolar threshold activation:
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
    # Net input: y_in = x * W
    y_in = np.dot(x_test, W)
    y_out = bipolar_activation(y_in)
    
    is_recalled = np.array_equal(y_out, s)
    status = "SUCCESS (Correctly Recalled)" if is_recalled else "FAILURE (Not Recognized)"
    
    print(f"\n--- {description} ---")
    print(f"Test Vector x      : {x_test}")
    print(f"Net Input y_in     : {y_in}")
    print(f"Retrieved Output y : {y_out}")
    print(f"Outcome            : {status}")
    return y_out

# 3. Testing Conditions
# Case 1: Original Pattern
test_pattern(np.array([-1, 1, 1, 1]), "Case 1: Same Input Vector")

# Case 2: One Missing Entry (represented by 0)
test_pattern(np.array([0, 1, 1, 1]), "Case 2A: One Missing Entry (Index 0 = 0)")
test_pattern(np.array([-1, 1, 0, 1]), "Case 2B: One Missing Entry (Index 2 = 0)")

# Case 3: One Mistake Entry (bit flipped)
test_pattern(np.array([-1, -1, 1, 1]), "Case 3: One Mistake Entry (Index 1 flipped to -1)")

# Case 4: Two Missing Entries
test_pattern(np.array([0, 0, 1, 1]), "Case 4A: Two Missing Entries (Indices 0, 1 = 0)")
test_pattern(np.array([-1, 0, 0, 1]), "Case 4B: Two Missing Entries (Indices 1, 2 = 0)")

# Case 5: Two Mistaken Entries
test_pattern(np.array([-1, -1, -1, 1]), "Case 5: Two Mistaken Entries (Indices 1, 2 flipped to -1)")

print("\n" + "=" * 65)
print("OBSERVATIONS:")
print("1. Missing entries (0) degrade signal margin gently (y_in: 4 -> 3 -> 2).")
print("2. Mistake entries (-1) actively cancel signals twice as fast (4 -> 2 -> 0).")
print("3. Two mistakes cause y_in = [0, 0, 0, 0], resulting in recall failure.")
print("=" * 65)