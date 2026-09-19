"""
Automated unit tests for Statutory Unit Sale Price (USP) Calculation & Verification
under the Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(m).

Covers 14 specific scenarios:
1.  Weight-based product < 1 kg (₹X.XX/g)
2.  Weight-based product >= 1 kg (₹X.XX/kg)
3.  Weight-based product declared in grams >= 1000g normalized to kg
4.  Volume-based product < 1 L (₹X.XX/ml)
5.  Volume-based product >= 1 L (₹X.XX/L)
6.  Volume-based product declared in ml >= 1000ml normalized to L
7.  Commodities sold by count / units (₹X.XX/unit)
8.  Strict rounding to 2 decimal places (e.g. 100 / 300 = 0.33)
9.  Separate numeric magnitude & unit normalization without string division
10. Missing / invalid MRP -> NOT_CALCULABLE, requires_review=True
11. Missing / invalid Net Quantity -> NOT_CALCULABLE, requires_review=True
12. Promotional free quantity handling (excludes free quantity from statutory base)
13. Statutory exemptions (Rule 6(1)(m) Proviso for 1kg/1L/1m/1unit, and Rule 26(a) for <=10g/<=10ml)
14. Divergence / mismatch detection between declared printed USP and calculated USP
"""

import unittest
from validation.field_validator import calculate_and_validate_usp, validate_net_quantity, validate_mrp
from compliance_rules import evaluate_unit_sale_price, evaluate_compliance


class TestUSPCalculation(unittest.TestCase):

    def test_1_weight_under_1kg(self):
        """1. Net quantity 500 g, MRP ₹100 -> USP = ₹0.20/g."""
        res = calculate_and_validate_usp(mrp=100.0, quantity=500.0, unit="g")
        self.assertEqual(res["status"], "CALCULATED")
        self.assertEqual(res["calculated_usp"], 0.20)
        self.assertEqual(res["calculated_unit"], "g")
        self.assertEqual(res["calculated_usp_formatted"], "₹0.20/g")
        self.assertEqual(res["evidence_string"], "USP = ₹100 ÷ 500 g = ₹0.20/g")
        self.assertFalse(res["has_mismatch"])

    def test_2_weight_over_1kg(self):
        """2. Net quantity 2 kg, MRP ₹250 -> USP = ₹125.00/kg."""
        res = calculate_and_validate_usp(mrp=250.0, quantity=2.0, unit="kg")
        self.assertEqual(res["status"], "CALCULATED")
        self.assertEqual(res["calculated_usp"], 125.00)
        self.assertEqual(res["calculated_unit"], "kg")
        self.assertEqual(res["calculated_usp_formatted"], "₹125.00/kg")
        self.assertEqual(res["evidence_string"], "USP = ₹250 ÷ 2 kg = ₹125.00/kg")
        self.assertFalse(res["has_mismatch"])

    def test_3_weight_grams_over_1000g(self):
        """3. Net quantity 1500 g, MRP ₹150 -> Normalized to 1.5 kg, USP = ₹100.00/kg."""
        res = calculate_and_validate_usp(mrp=150.0, quantity=1500.0, unit="g")
        self.assertEqual(res["status"], "CALCULATED")
        self.assertEqual(res["calculated_usp"], 100.00)
        self.assertEqual(res["calculated_unit"], "kg")
        self.assertEqual(res["calculated_usp_formatted"], "₹100.00/kg")
        self.assertEqual(res["evidence_string"], "USP = ₹150 ÷ 1.5 kg = ₹100.00/kg")

    def test_4_volume_under_1litre(self):
        """4. Net quantity 500 ml, MRP ₹50 -> USP = ₹0.10/ml."""
        res = calculate_and_validate_usp(mrp=50.0, quantity=500.0, unit="ml")
        self.assertEqual(res["status"], "CALCULATED")
        self.assertEqual(res["calculated_usp"], 0.10)
        self.assertEqual(res["calculated_unit"], "ml")
        self.assertEqual(res["calculated_usp_formatted"], "₹0.10/ml")
        self.assertEqual(res["evidence_string"], "USP = ₹50 ÷ 500 ml = ₹0.10/ml")

    def test_5_volume_over_1litre(self):
        """5. Net quantity 2 L, MRP ₹180 -> USP = ₹90.00/L."""
        res = calculate_and_validate_usp(mrp=180.0, quantity=2.0, unit="L")
        self.assertEqual(res["status"], "CALCULATED")
        self.assertEqual(res["calculated_usp"], 90.00)
        self.assertEqual(res["calculated_unit"], "L")
        self.assertEqual(res["calculated_usp_formatted"], "₹90.00/L")
        self.assertEqual(res["evidence_string"], "USP = ₹180 ÷ 2 L = ₹90.00/L")

    def test_6_volume_ml_over_1000ml(self):
        """6. Net quantity 2000 ml, MRP ₹200 -> Normalized to 2 L, USP = ₹100.00/L."""
        res = calculate_and_validate_usp(mrp=200.0, quantity=2000.0, unit="ml")
        self.assertEqual(res["status"], "CALCULATED")
        self.assertEqual(res["calculated_usp"], 100.00)
        self.assertEqual(res["calculated_unit"], "L")
        self.assertEqual(res["calculated_usp_formatted"], "₹100.00/L")
        self.assertEqual(res["evidence_string"], "USP = ₹200 ÷ 2 L = ₹100.00/L")

    def test_7_count_units(self):
        """7. Net quantity 10 units / pieces / N, MRP ₹150 -> USP = ₹15.00/unit."""
        for u in ["units", "unit", "pieces", "piece", "number", "u", "n"]:
            res = calculate_and_validate_usp(mrp=150.0, quantity=10.0, unit=u)
            self.assertEqual(res["status"], "CALCULATED")
            self.assertEqual(res["calculated_usp"], 15.00)
            self.assertEqual(res["calculated_unit"], "unit")
            self.assertEqual(res["calculated_usp_formatted"], "₹15.00/unit")
            self.assertEqual(res["evidence_string"], "USP = ₹150 ÷ 10 units = ₹15.00/unit")

    def test_8_rounding_strictly_two_decimals(self):
        """8. MRP ₹100, Net quantity 300 g -> 100 / 300 = 0.3333... -> strictly ₹0.33/g."""
        res = calculate_and_validate_usp(mrp=100.0, quantity=300.0, unit="g")
        self.assertEqual(res["calculated_usp"], 0.33)
        self.assertEqual(res["calculated_usp_formatted"], "₹0.33/g")

        # Also test round-half behavior e.g. 100 / 600 = 0.1666... -> 0.17
        res2 = calculate_and_validate_usp(mrp=100.0, quantity=600.0, unit="g")
        self.assertEqual(res2["calculated_usp"], 0.17)
        self.assertEqual(res2["calculated_usp_formatted"], "₹0.17/g")

    def test_9_unit_normalization_no_string_division(self):
        """9. Parsing magnitude and unit separately without string division."""
        # 500 g must evaluate as 500 g, not 0.5 g
        res_500g = calculate_and_validate_usp(mrp=100.0, quantity=500.0, unit="g")
        self.assertEqual(res_500g["calculated_unit"], "g")
        self.assertEqual(res_500g["calculated_usp"], 0.20)

        # 2 kg must evaluate as 2 kg
        res_2kg = calculate_and_validate_usp(mrp=200.0, quantity=2.0, unit="kg")
        self.assertEqual(res_2kg["calculated_unit"], "kg")
        self.assertEqual(res_2kg["calculated_usp"], 100.00)

        # 500 ml must evaluate as 500 ml, not 0.5 L
        res_500ml = calculate_and_validate_usp(mrp=50.0, quantity=500.0, unit="ml")
        self.assertEqual(res_500ml["calculated_unit"], "ml")
        self.assertEqual(res_500ml["calculated_usp"], 0.10)

        # 2 L must evaluate as 2 L
        res_2l = calculate_and_validate_usp(mrp=180.0, quantity=2.0, unit="L")
        self.assertEqual(res_2l["calculated_unit"], "L")
        self.assertEqual(res_2l["calculated_usp"], 90.00)

    def test_10_missing_or_invalid_mrp(self):
        """10. Missing / invalid MRP -> USP = NOT_CALCULABLE, requires_review = True."""
        for invalid_mrp in [None, 0.0, -10.0]:
            res = calculate_and_validate_usp(mrp=invalid_mrp, quantity=500.0, unit="g")
            self.assertEqual(res["status"], "NOT_CALCULABLE")
            self.assertIsNone(res["calculated_usp"])
            self.assertEqual(res["calculated_usp_formatted"], "NOT_CALCULABLE")
            self.assertTrue(res["requires_review"])
            self.assertIn("NOT_CALCULABLE", res["evidence_string"])

    def test_11_missing_or_invalid_net_quantity(self):
        """11. Missing / invalid Net Quantity -> USP = NOT_CALCULABLE, requires_review = True."""
        for invalid_qty in [None, 0.0, -5.0]:
            res = calculate_and_validate_usp(mrp=100.0, quantity=invalid_qty, unit="g")
            self.assertEqual(res["status"], "NOT_CALCULABLE")
            self.assertIsNone(res["calculated_usp"])
            self.assertEqual(res["calculated_usp_formatted"], "NOT_CALCULABLE")
            self.assertTrue(res["requires_review"])

        # Unrecognized unit
        res_bad_unit = calculate_and_validate_usp(mrp=100.0, quantity=500.0, unit="widgets")
        self.assertEqual(res_bad_unit["status"], "NOT_CALCULABLE")
        self.assertEqual(res_bad_unit["calculated_usp_formatted"], "NOT_CALCULABLE")
        self.assertTrue(res_bad_unit["requires_review"])

    def test_12_promotional_free_quantity_handling(self):
        """12. Promotional free quantity handling under PCR 2011."""
        # "100 g + 20 g Free", MRP ₹60 -> calculates on statutory base 100g = ₹0.60/g
        q_eval = validate_net_quantity("100 g + 20 g Free")
        self.assertTrue(q_eval["has_free_quantity"])
        self.assertEqual(q_eval["free_magnitude"], 20.0)
        self.assertEqual(q_eval["base_numeric_value"], 100.0)

        res = calculate_and_validate_usp(
            mrp=60.0,
            quantity=q_eval["base_numeric_value"],
            unit=q_eval["normalized_unit"],
            raw_quantity_str="100 g + 20 g Free"
        )
        self.assertEqual(res["status"], "CALCULATED")
        self.assertEqual(res["calculated_usp"], 0.60)
        self.assertEqual(res["calculated_unit"], "g")
        self.assertEqual(res["calculated_usp_formatted"], "₹0.60/g")
        self.assertIn("promotional 20 g free excluded under PCR 2011", res["evidence_string"])

    def test_13_statutory_exemptions(self):
        """13. Statutory exemptions under Rule 6(1)(m) Proviso and Rule 26(a)."""
        # A. Proviso to Rule 6(1)(m): 1 kg, 1 L, 1 m, 1 unit (where RSP equals USP)
        for unit in ["kg", "L", "m", "unit"]:
            res = calculate_and_validate_usp(mrp=100.0, quantity=1.0, unit=unit)
            self.assertEqual(res["status"], "EXEMPT")
            self.assertTrue(res["is_exempt"])
            self.assertTrue(res["is_exempt_equal_mrp"])
            self.assertIn("Proviso to Rule 6(1)(m)", res["exempt_reason"])

        # Also 1000g or 1000ml equals 1kg / 1L
        res_1000g = calculate_and_validate_usp(mrp=100.0, quantity=1000.0, unit="g")
        self.assertTrue(res_1000g["is_exempt"])

        # B. Rule 26(a): Small packages <= 10g or <= 10ml
        res_small_g = calculate_and_validate_usp(mrp=5.0, quantity=10.0, unit="g")
        self.assertTrue(res_small_g["is_exempt"])
        self.assertTrue(res_small_g["is_small_pack_exempt"])
        self.assertIn("Rule 26(a)", res_small_g["exempt_reason"])

        res_small_ml = calculate_and_validate_usp(mrp=5.0, quantity=8.0, unit="ml")
        self.assertTrue(res_small_ml["is_exempt"])
        self.assertTrue(res_small_ml["is_small_pack_exempt"])

    def test_14_divergence_mismatch_flag(self):
        """14. Divergence between printed OCR USP and calculated statutory USP."""
        # Case A: Mismatch (e.g. Printed Rs. 0.90 per g vs Calculated ₹0.56/g)
        res_mismatch = calculate_and_validate_usp(
            mrp=28.0,
            quantity=50.0,
            unit="g",
            declared_usp_str="Rs. 0.90 per g"
        )
        self.assertTrue(res_mismatch["has_mismatch"])
        self.assertFalse(res_mismatch["matches"])
        self.assertEqual(res_mismatch["printed_usp"], "Rs. 0.90 per g")
        self.assertEqual(res_mismatch["printed_usp_num"], 0.90)
        self.assertEqual(res_mismatch["calculated_usp"], 0.56)
        self.assertIn("USP mismatch — Inspector Review Required", res_mismatch["mismatch_message"])
        self.assertTrue(res_mismatch["requires_review"])

        # Case B: Match (Printed Rs. 0.56 per g vs Calculated ₹0.56/g)
        res_match = calculate_and_validate_usp(
            mrp=28.0,
            quantity=50.0,
            unit="g",
            declared_usp_str="Rs. 0.56 per g"
        )
        self.assertFalse(res_match["has_mismatch"])
        self.assertTrue(res_match["matches"])
        self.assertEqual(res_match["printed_usp_num"], 0.56)
        self.assertEqual(res_match["calculated_usp"], 0.56)
        self.assertFalse(res_match["requires_review"])

        # Case C: End-to-end statutory rule evaluation confirms NON_COMPLIANT status on mismatch
        rule_eval = evaluate_unit_sale_price(
            mrp_str="MRP Rs. 28.00 (incl. of all taxes)",
            net_quantity_str="50 g",
            declared_usp_str="Rs. 0.90 per g"
        )
        self.assertEqual(rule_eval["status"], "NON_COMPLIANT")
        self.assertIn("USP mismatch — Inspector Review Required", rule_eval["reason"])


if __name__ == "__main__":
    unittest.main()
