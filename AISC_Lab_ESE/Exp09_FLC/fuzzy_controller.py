"""
Experiment 9: Fuzzy Logic Controller (FLC)
File: fuzzy_controller.py
Description: Implementation of a Mamdani Fuzzy Logic Controller (FLC).
             Application: Washing Machine Wash Time Controller.
             Inputs : Dirt Level (0-100%), Grease/Fabric Sensitivity (0-100%)
             Output : Wash Time (0-60 minutes)
             Methods: Triangular Membership Functions, Min-Max Inference, Centroid Defuzzification (COG).
"""

from typing import Dict, List, Tuple


def trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular Membership Function:
    0 for x <= a or x >= c
    (x - a) / (b - a) for a < x <= b
    (c - x) / (c - b) for b < x < c
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal Membership Function."""
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


class FuzzyController:
    def __init__(self):
        # Output Fuzzy Sets for Wash Time (minutes 0 to 60)
        self.output_sets = {
            'Very_Short': (0.0, 0.0, 10.0),
            'Short'     : (0.0, 15.0, 30.0),
            'Medium'    : (15.0, 30.0, 45.0),
            'Long'      : (30.0, 45.0, 60.0),
            'Very_Long' : (45.0, 60.0, 60.0)
        }

    def fuzzify_dirt(self, x: float) -> Dict[str, float]:
        """Fuzzifies Dirt (0-100%): Low, Medium, High"""
        return {
            'Low'   : trapmf(x, -10.0, 0.0, 20.0, 50.0),
            'Medium': trimf(x, 20.0, 50.0, 80.0),
            'High'  : trapmf(x, 50.0, 80.0, 100.0, 110.0)
        }

    def fuzzify_grease(self, y: float) -> Dict[str, float]:
        """Fuzzifies Grease/Sensitivity (0-100%): Low, Medium, High"""
        return {
            'Low'   : trapmf(y, -10.0, 0.0, 20.0, 50.0),
            'Medium': trimf(y, 20.0, 50.0, 80.0),
            'High'  : trapmf(y, 50.0, 80.0, 100.0, 110.0)
        }

    def evaluate_rules(self, dirt_deg: Dict[str, float], grease_deg: Dict[str, float]) -> Dict[str, float]:
        """
        9 Mamdani Fuzzy Rules:
        1. IF Dirt is Low    AND Grease is Low    THEN Wash Time is Very_Short
        2. IF Dirt is Low    AND Grease is Medium THEN Wash Time is Short
        3. IF Dirt is Low    AND Grease is High   THEN Wash Time is Medium
        4. IF Dirt is Medium AND Grease is Low    THEN Wash Time is Short
        5. IF Dirt is Medium AND Grease is Medium THEN Wash Time is Medium
        6. IF Dirt is Medium AND Grease is High   THEN Wash Time is Long
        7. IF Dirt is High   AND Grease is Low    THEN Wash Time is Medium
        8. IF Dirt is High   AND Grease is Medium THEN Wash Time is Long
        9. IF Dirt is High   AND Grease is High   THEN Wash Time is Very_Long
        """
        rule_outputs = {
            'Very_Short': min(dirt_deg['Low'], grease_deg['Low']),
            'Short'     : max(min(dirt_deg['Low'], grease_deg['Medium']),
                              min(dirt_deg['Medium'], grease_deg['Low'])),
            'Medium'    : max(min(dirt_deg['Low'], grease_deg['High']),
                              min(dirt_deg['Medium'], grease_deg['Medium']),
                              min(dirt_deg['High'], grease_deg['Low'])),
            'Long'      : max(min(dirt_deg['Medium'], grease_deg['High']),
                              min(dirt_deg['High'], grease_deg['Medium'])),
            'Very_Long' : min(dirt_deg['High'], grease_deg['High'])
        }
        return rule_outputs

    def defuzzify_centroid(self, rule_weights: Dict[str, float], resolution: int = 600) -> float:
        """
        Centroid / Center of Gravity (COG) Defuzzification:
        z* = sum(z * mu(z)) / sum(mu(z))
        """
        numerator = 0.0
        denominator = 0.0

        step = 60.0 / resolution
        for i in range(resolution + 1):
            z = i * step

            # Find aggregated membership at z: mu(z) = max_k (min(rule_weight[k], set_membership(z)))
            mu_z = 0.0
            for term, (a, b, c) in self.output_sets.items():
                w = rule_weights.get(term, 0.0)
                if w > 0:
                    val = min(w, trimf(z, a, b, c))
                    if val > mu_z:
                        mu_z = val

            numerator += z * mu_z
            denominator += mu_z

        if denominator == 0.0:
            return 30.0  # Default middle value

        return numerator / denominator

    def compute(self, dirt: float, grease: float) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, float]]:
        dirt_deg = self.fuzzify_dirt(dirt)
        grease_deg = self.fuzzify_grease(grease)
        rule_weights = self.evaluate_rules(dirt_deg, grease_deg)
        wash_time = self.defuzzify_centroid(rule_weights)
        return wash_time, dirt_deg, grease_deg, rule_weights


def demo_flc():
    print("=" * 70)
    print("   MAMDANI FUZZY LOGIC CONTROLLER (WASHING MACHINE SYSTEM)    ")
    print("=" * 70)
    print("Inputs : Dirt Level (0 to 100%), Grease Level (0 to 100%)")
    print("Output : Recommended Wash Time (0 to 60 Minutes)")

    controller = FuzzyController()

    test_cases = [
        (20.0, 15.0, "Lightly soiled gentle clothes"),
        (65.0, 70.0, "Heavily soiled greasy workwear"),
        (45.0, 50.0, "Average daily laundry load"),
        (85.0, 90.0, "Extremely muddy garments"),
    ]

    for dirt_val, grease_val, scenario in test_cases:
        wash_time, d_deg, g_deg, r_weights = controller.compute(dirt_val, grease_val)

        print("\n" + "-" * 70)
        print(f"Scenario: {scenario}")
        print(f"Inputs  : Dirt = {dirt_val}%, Grease = {grease_val}%")
        print("1. Fuzzification:")
        print(f"   Dirt Memberships   : Low={d_deg['Low']:.2f}, Medium={d_deg['Medium']:.2f}, High={d_deg['High']:.2f}")
        print(f"   Grease Memberships : Low={g_deg['Low']:.2f}, Medium={g_deg['Medium']:.2f}, High={g_deg['High']:.2f}")
        print("2. Active Rule Consequent Strengths (Mamdani Implication):")
        for term, strength in r_weights.items():
            if strength > 0:
                print(f"   - {term:<12}: firing strength = {strength:.3f}")
        print("3. Defuzzified Result (Centroid Method - COG):")
        print(f"   --> Recommended Wash Time: {wash_time:.2f} Minutes")
    print("=" * 70)


if __name__ == "__main__":
    demo_flc()
