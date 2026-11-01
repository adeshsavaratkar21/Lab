"""
Experiment 8: Fuzzy Sets & Relations
File: fuzzy_operations.py
Description: Implementation of classical and algebraic operations on Fuzzy Sets,
             Cartesian Product, Fuzzy Relation Compositions (Max-Min, Max-Product),
             and De Morgan's Law verification.
"""

from typing import Dict, List


class FuzzySet:
    def __init__(self, elements: Dict[str, float]):
        """
        :param elements: Dictionary mapping universe element -> membership value [0.0, 1.0]
        """
        self.memberships = {k: max(0.0, min(1.0, float(v))) for k, v in elements.items()}

    def __getitem__(self, item: str) -> float:
        return self.memberships.get(item, 0.0)

    def universe(self) -> List[str]:
        return list(self.memberships.keys())

    def __str__(self) -> str:
        items = [f"{mu:.2f}/{x}" for x, mu in self.memberships.items()]
        return "{ " + " + ".join(items) + " }"


# -------------------------------------------------------------
# 1. Standard Fuzzy Set Operations
# -------------------------------------------------------------

def fuzzy_union(A: FuzzySet, B: FuzzySet) -> FuzzySet:
    """Union: mu_AUB(x) = max(mu_A(x), mu_B(x))"""
    keys = set(A.universe()) | set(B.universe())
    return FuzzySet({k: max(A[k], B[k]) for k in keys})


def fuzzy_intersection(A: FuzzySet, B: FuzzySet) -> FuzzySet:
    """Intersection: mu_AnB(x) = min(mu_A(x), mu_B(x))"""
    keys = set(A.universe()) | set(B.universe())
    return FuzzySet({k: min(A[k], B[k]) for k in keys})


def fuzzy_complement(A: FuzzySet) -> FuzzySet:
    """Complement: mu_A'(x) = 1 - mu_A(x)"""
    return FuzzySet({k: 1.0 - A[k] for k in A.universe()})


# -------------------------------------------------------------
# 2. Algebraic Operations
# -------------------------------------------------------------

def algebraic_sum(A: FuzzySet, B: FuzzySet) -> FuzzySet:
    """Algebraic Sum: mu_A+B(x) = mu_A(x) + mu_B(x) - mu_A(x)*mu_B(x)"""
    keys = set(A.universe()) | set(B.universe())
    return FuzzySet({k: A[k] + B[k] - (A[k] * B[k]) for k in keys})


def algebraic_product(A: FuzzySet, B: FuzzySet) -> FuzzySet:
    """Algebraic Product: mu_A*B(x) = mu_A(x) * mu_B(x)"""
    keys = set(A.universe()) | set(B.universe())
    return FuzzySet({k: A[k] * B[k] for k in keys})


def bounded_sum(A: FuzzySet, B: FuzzySet) -> FuzzySet:
    """Bounded Sum: mu(x) = min(1, mu_A(x) + mu_B(x))"""
    keys = set(A.universe()) | set(B.universe())
    return FuzzySet({k: min(1.0, A[k] + B[k]) for k in keys})


def bounded_difference(A: FuzzySet, B: FuzzySet) -> FuzzySet:
    """Bounded Difference: mu(x) = max(0, mu_A(x) - mu_B(x))"""
    keys = set(A.universe()) | set(B.universe())
    return FuzzySet({k: max(0.0, A[k] - B[k]) for k in keys})


# -------------------------------------------------------------
# 3. Fuzzy Relations & Compositions
# -------------------------------------------------------------

def cartesian_product(A: FuzzySet, B: FuzzySet) -> Dict[str, Dict[str, float]]:
    """Cartesian Product: R(x, y) = min(mu_A(x), mu_B(y))"""
    relation = {}
    for x in A.universe():
        relation[x] = {}
        for y in B.universe():
            relation[x][y] = min(A[x], B[y])
    return relation


def max_min_composition(R: List[List[float]], S: List[List[float]]) -> List[List[float]]:
    """
    Max-Min Composition: (R o S)[i][j] = max_k (min(R[i][k], S[k][j]))
    """
    rows_R, cols_R = len(R), len(R[0])
    rows_S, cols_S = len(S), len(S[0])
    assert cols_R == rows_S, "Matrix dimensions must match for composition!"

    result = [[0.0] * cols_S for _ in range(rows_R)]
    for i in range(rows_R):
        for j in range(cols_S):
            result[i][j] = max(min(R[i][k], S[k][j]) for k in range(cols_R))
    return result


def max_product_composition(R: List[List[float]], S: List[List[float]]) -> List[List[float]]:
    """
    Max-Product Composition: (R . S)[i][j] = max_k (R[i][k] * S[k][j])
    """
    rows_R, cols_R = len(R), len(R[0])
    rows_S, cols_S = len(S), len(S[0])
    assert cols_R == rows_S, "Matrix dimensions must match for composition!"

    result = [[0.0] * cols_S for _ in range(rows_R)]
    for i in range(rows_R):
        for j in range(cols_S):
            result[i][j] = max(R[i][k] * S[k][j] for k in range(cols_R))
    return result


def print_matrix(mat: List[List[float]], title: str):
    print(f"\n{title}:")
    for row in mat:
        print("  [" + ", ".join(f"{val:.2f}" for val in row) + "]")


def verify_demorgans(A: FuzzySet, B: FuzzySet):
    print("\n" + "=" * 65)
    print("           DE MORGAN'S LAWS VERIFICATION                      ")
    print("=" * 65)
    # Law 1: (A U B)' = A' n B'
    lhs1 = fuzzy_complement(fuzzy_union(A, B))
    rhs1 = fuzzy_intersection(fuzzy_complement(A), fuzzy_complement(B))
    law1_passed = all(abs(lhs1[k] - rhs1[k]) < 1e-6 for k in A.universe())

    print("1. Law 1: Compl(A U B) == Compl(A) n Compl(B)")
    print(f"   LHS Compl(A U B)     : {lhs1}")
    print(f"   RHS Compl(A) n Compl(B): {rhs1}")
    print(f"   --> Verification Status: {'VERIFIED (True)' if law1_passed else 'FAILED'}")

    # Law 2: (A n B)' = A' U B'
    lhs2 = fuzzy_complement(fuzzy_intersection(A, B))
    rhs2 = fuzzy_union(fuzzy_complement(A), fuzzy_complement(B))
    law2_passed = all(abs(lhs2[k] - rhs2[k]) < 1e-6 for k in A.universe())

    print("\n2. Law 2: Compl(A n B) == Compl(A) U Compl(B)")
    print(f"   LHS Compl(A n B)     : {lhs2}")
    print(f"   RHS Compl(A) U Compl(B): {rhs2}")
    print(f"   --> Verification Status: {'VERIFIED (True)' if law2_passed else 'FAILED'}")
    print("=" * 65)


def demo_fuzzy():
    print("=" * 65)
    print("                 FUZZY SET OPERATIONS EXPERIMENT              ")
    print("=" * 65)

    A = FuzzySet({'x1': 0.2, 'x2': 0.7, 'x3': 0.4, 'x4': 1.0})
    B = FuzzySet({'x1': 0.5, 'x2': 0.3, 'x3': 0.8, 'x4': 0.6})

    print(f"Fuzzy Set A = {A}")
    print(f"Fuzzy Set B = {B}")

    print("\n--- Standard Operations ---")
    print(f"Union (A U B)        : {fuzzy_union(A, B)}")
    print(f"Intersection (A n B) : {fuzzy_intersection(A, B)}")
    print(f"Complement of A (A') : {fuzzy_complement(A)}")
    print(f"Complement of B (B') : {fuzzy_complement(B)}")

    print("\n--- Algebraic & Bounded Operations ---")
    print(f"Algebraic Sum (A + B)       : {algebraic_sum(A, B)}")
    print(f"Algebraic Product (A * B)   : {algebraic_product(A, B)}")
    print(f"Bounded Sum (A (+) B)       : {bounded_sum(A, B)}")
    print(f"Bounded Difference (A (-) B): {bounded_difference(A, B)}")

    # De Morgan's Law Verification
    verify_demorgans(A, B)

    # Fuzzy Relation Composition
    print("\n" + "=" * 65)
    print("              FUZZY RELATION COMPOSITIONS                     ")
    print("=" * 65)
    # Relation R (2x3) and Relation S (3x2)
    R = [
        [0.6, 0.3, 0.8],
        [0.2, 0.7, 0.4]
    ]
    S = [
        [0.1, 0.9],
        [0.4, 0.5],
        [0.8, 0.2]
    ]

    print_matrix(R, "Relation Matrix R (2x3)")
    print_matrix(S, "Relation Matrix S (3x2)")

    max_min_res = max_min_composition(R, S)
    print_matrix(max_min_res, "Max-Min Composition (R o S) [2x2]")

    max_prod_res = max_product_composition(R, S)
    print_matrix(max_prod_res, "Max-Product Composition (R . S) [2x2]")
    print("=" * 65)


if __name__ == "__main__":
    demo_fuzzy()
