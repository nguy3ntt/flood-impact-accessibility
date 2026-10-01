import math
import unittest
from flood_access.network import Edge, shortest_cost, synthetic_demo

class NetworkTests(unittest.TestCase):
    def test_exposure_requires_a_declared_disruption_assumption(self):
        edges = [Edge("A", "H", 2, "exposed"), Edge("A", "B", 3), Edge("B", "H", 4)]
        self.assertEqual(shortest_cost(edges, "A", "H"), 2)
        self.assertEqual(shortest_cost(edges, "A", "H", "exposed_only"), 7)

    def test_unknown_is_not_implicitly_clear_in_conservative_case(self):
        edges = [Edge("A", "H", 5, "unknown")]
        self.assertEqual(shortest_cost(edges, "A", "H", "exposed_only"), 5)
        self.assertIsNone(shortest_cost(edges, "A", "H", "conservative"))

    def test_direction_is_preserved(self):
        self.assertIsNone(shortest_cost([Edge("A", "H", 1)], "H", "A"))

    def test_costs_reject_negative_nonfinite_values(self):
        for value in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                Edge("A", "H", value)

    def test_missing_nodes_and_policies_fail_explicitly(self):
        edges = [Edge("A", "H", 1)]
        with self.assertRaises(ValueError):
            shortest_cost(edges, "missing", "H")
        with self.assertRaises(ValueError):
            shortest_cost(edges, "A", "H", "safe")

    def test_scenario_can_disconnect_without_zero_cost(self):
        result = synthetic_demo("conservative")
        self.assertEqual(result["data_mode"], "synthetic")
        self.assertIsNone(result["results"][0]["scenario_cost"])
        self.assertTrue(result["results"][0]["newly_disconnected"])
        self.assertEqual(result["results"][2]["scenario_cost"], 6)

    def test_baseline_disconnected_is_not_newly_disconnected(self):
        edges = [Edge("A", "B", 1), Edge("H", "C", 1)]
        self.assertIsNone(shortest_cost(edges, "A", "H"))

    def test_no_disruption_equals_baseline_and_zero_is_valid(self):
        edges = [Edge("A", "B", 0), Edge("B", "H", 2)]
        self.assertEqual(shortest_cost(edges, "A", "H", "conservative"), 2)
        self.assertEqual(shortest_cost(edges, "H", "H"), 0)

if __name__ == "__main__":
    unittest.main()
