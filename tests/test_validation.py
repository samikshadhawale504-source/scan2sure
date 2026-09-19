"""
Unit tests for Field Validation Layer (validation/field_validator.py).
Tests statutory metric units, prohibited symbols, MRP formatting, USP calculations, and dates.
"""

import unittest
from validation.field_validator import (
    validate_mrp,
    validate_net_quantity,
    calculate_and_validate_usp,
    validate_date,
    validate_consumer_care,
    validate_manufacturer_address,
    validate_all_fields
)
from pipeline.hybrid_extractor import detect_extraction_conflicts, normalize_field_for_comparison


class TestFieldValidation(unittest.TestCase):

    def test_valid_mrp(self):
        res = validate_mrp("Rs. 28.00 (incl. of all taxes)")
        self.assertTrue(res["valid"])
        self.assertEqual(res["numeric_price"], 28.0)
        self.assertTrue(res["has_tax_clause"])
        self.assertEqual(len(res["issues"]), 0)

    def test_mrp_without_tax_clause(self):
        res = validate_mrp("₹50")
        self.assertTrue(res["valid"])
        self.assertFalse(res["has_tax_clause"])
        self.assertTrue(any("inclusive of all taxes" in w for w in res["warnings"]))

    def test_invalid_mrp(self):
        res = validate_mrp("Free Sample")
        self.assertFalse(res["valid"])
        self.assertIsNone(res["numeric_price"])

    def test_legal_metric_units(self):
        valid_cases = ["50 g", "1 kg", "500 ml", "2 l", "10 units", "1 number"]
        for case in valid_cases:
            res = validate_net_quantity(case)
            self.assertTrue(res["valid"], f"Failed for valid case: {case}")
            self.assertFalse(res["is_illegal_unit"])

    def test_prohibited_unit_abbreviations(self):
        prohibited_cases = [("50 gms", "g"), ("2 kgs", "kg"), ("500 mls", "ml"), ("1 ltr", "l")]
        for case, expected_standard in prohibited_cases:
            res = validate_net_quantity(case)
            self.assertTrue(res["is_illegal_unit"], f"Should flag prohibited unit: {case}")
            self.assertEqual(res["normalized_unit"], expected_standard)

    def test_unit_sale_price_calculation_under_1kg(self):
        # 50g at Rs. 28 -> Expected USP = 28 / 50 = 0.56 per g
        res = calculate_and_validate_usp(28.0, 50.0, "g", "Rs. 0.56 per g")
        self.assertTrue(res["applicable"])
        self.assertEqual(res["expected_usp"], 0.56)
        self.assertEqual(res["expected_unit"], "g")
        self.assertTrue(res["matches"])

    def test_unit_sale_price_calculation_over_1kg(self):
        # 2 kg at Rs. 200 -> Expected USP = 200 / 2 = 100 per kg
        res = calculate_and_validate_usp(200.0, 2.0, "kg", "Rs. 100.00 / kg")
        self.assertTrue(res["applicable"])
        self.assertEqual(res["expected_usp"], 100.0)
        self.assertEqual(res["expected_unit"], "kg")
        self.assertTrue(res["matches"])

    def test_unit_sale_price_divergence(self):
        # 50g at Rs. 28 -> Declared 0.90 per g (diverges from 0.56)
        res = calculate_and_validate_usp(28.0, 50.0, "g", "Rs. 0.90 per g")
        self.assertFalse(res["matches"])
        self.assertTrue(len(res["warnings"]) > 0)

    def test_valid_dates(self):
        res1 = validate_date("15/10/2025")
        self.assertTrue(res1["valid"])
        self.assertEqual(res1["parsed_month"], 10)
        self.assertEqual(res1["parsed_year"], 2025)

        res2 = validate_date("Oct 2025")
        self.assertTrue(res2["valid"])
        self.assertEqual(res2["parsed_month"], 10)
        self.assertEqual(res2["parsed_year"], 2025)

    def test_invalid_date_month(self):
        res = validate_date("15/18/2025")
        self.assertFalse(res["valid"])
        self.assertTrue(any("Invalid calendar month" in i for i in res["issues"]))

    def test_consumer_care_validation(self):
        # Multi-channel
        res1 = validate_consumer_care("Toll-free: +91 2248253651 | Email: care@monaliz.com")
        self.assertTrue(res1["valid"])
        self.assertEqual(res1["channel_count"], 2)

        # Missing completely
        res2 = validate_consumer_care("")
        self.assertFalse(res2["valid"])

    def test_metric_unit_case_sensitivity(self):
        # Legal Metrology Rule 13 mandates lower-case symbols for mass and length
        res_upper_g = validate_net_quantity("50 G")
        self.assertTrue(res_upper_g["is_case_violation"])
        self.assertFalse(res_upper_g["valid"])
        self.assertTrue(any("Formatting Advisory under Rule 13" in w for w in res_upper_g["warnings"]))

        res_upper_kg = validate_net_quantity("2 KG")
        self.assertTrue(res_upper_kg["is_case_violation"])
        self.assertFalse(res_upper_kg["valid"])

        res_correct_g = validate_net_quantity("50 g")
        self.assertFalse(res_correct_g["is_case_violation"])
        self.assertTrue(res_correct_g["valid"])

        res_correct_kg = validate_net_quantity("2 kg")
        self.assertFalse(res_correct_kg["is_case_violation"])
        self.assertTrue(res_correct_kg["valid"])

        # Volume permits capital 'L' under international SI & Rule 13 standard
        res_volume_ml = validate_net_quantity("500 mL")
        self.assertFalse(res_volume_ml["is_case_violation"])
        self.assertTrue(res_volume_ml["valid"])

        res_volume_l = validate_net_quantity("1 L")
        self.assertFalse(res_volume_l["is_case_violation"])
        self.assertTrue(res_volume_l["valid"])

    def test_prohibited_quantity_expressions(self):
        # Rule 11(2) strictly prohibits qualifying words like 'approximate' for all commodities
        res1 = validate_net_quantity("approx. 50 g")
        self.assertTrue(res1["has_prohibited_qualifier"])
        self.assertFalse(res1["valid"])
        self.assertTrue(any("Rule 11(2)" in i for i in res1["issues"]))

        res2 = validate_net_quantity("minimum 500 ml")
        self.assertTrue(res2["has_prohibited_qualifier"])
        self.assertFalse(res2["valid"])

        res3 = validate_net_quantity("gross 1 kg")
        self.assertTrue(res3["has_prohibited_qualifier"])
        self.assertFalse(res3["valid"])

    def test_rule11_proviso_climatic_variation(self):
        # Under Proviso to Rule 11(2), commodities susceptible to climatic variation (soaps, camphor)
        # are legally permitted to use 'when packed' / 'net weight when packed'.
        res_soap = validate_net_quantity("125 g when packed", commodity_name="Lux Bathing Soap")
        self.assertFalse(res_soap["has_prohibited_qualifier"])
        self.assertTrue(res_soap["is_permitted_climatic_qualifier"])
        self.assertTrue(res_soap["valid"])
        self.assertTrue(any("Permitted under Proviso to Rule 11(2)" in w for w in res_soap["warnings"]))

        # Non-climatic commodities (e.g. baking powder, flour) cannot use 'when packed'
        res_baking = validate_net_quantity("50 g when packed", commodity_name="Monaliz Baking Powder")
        self.assertTrue(res_baking["has_prohibited_qualifier"])
        self.assertFalse(res_baking["valid"])

        # 'approx.' is never permitted even for soaps
        res_soap_approx = validate_net_quantity("approx 125 g", commodity_name="Bathing Soap")
        self.assertTrue(res_soap_approx["has_prohibited_qualifier"])
        self.assertFalse(res_soap_approx["valid"])

    def test_rule13_spacing_advisory(self):
        # Rule 13 formatting: missing space between number and unit produces advisory, not non-compliance
        res_no_space = validate_net_quantity("50g")
        self.assertTrue(res_no_space["valid"])
        self.assertTrue(res_no_space["has_spacing_advisory"])
        self.assertTrue(any("Formatting Recommendation under Rule 13" in w for w in res_no_space["warnings"]))

        res_with_space = validate_net_quantity("50 g")
        self.assertTrue(res_with_space["valid"])
        self.assertFalse(res_with_space["has_spacing_advisory"])

    def test_vulgar_fractions_prohibited(self):
        # Rule 12(3) prohibits vulgar fractions (e.g. '1/2 kg')
        res_frac = validate_net_quantity("1/2 kg")
        self.assertTrue(res_frac["is_vulgar_fraction"])
        self.assertFalse(res_frac["valid"])
        self.assertTrue(any("Rule 12(3)" in i for i in res_frac["issues"]))

    def test_invalid_magnitudes(self):
        # Zero or negative net quantity
        res_zero = validate_net_quantity("0 g")
        self.assertFalse(res_zero["valid"])
        self.assertTrue(any("must be positive" in i for i in res_zero["issues"]))

        res_neg = validate_net_quantity("-50 ml")
        self.assertFalse(res_neg["valid"])

    def test_mrp_missing_prefix(self):
        # Rule 6(1)(da) prescribes the prefix words 'MRP' or 'Maximum Retail Price'
        res_with_prefix = validate_mrp("MRP Rs. 28.00 (incl. of all taxes)")
        self.assertTrue(res_with_prefix["has_mrp_prefix"])
        self.assertEqual(len(res_with_prefix["warnings"]), 0)

        res_without_prefix = validate_mrp("Rs. 28.00 (incl. of all taxes)")
        self.assertFalse(res_without_prefix["has_mrp_prefix"])
        self.assertTrue(any("Statutory prefix 'MRP'" in w for w in res_without_prefix["warnings"]))

    def test_validate_all_fields_with_fields_detail(self):
        # Test confidence propagation and conflict warnings into validate_all_fields
        fields = {
            "Product Name": "Monaliz Baking Powder",
            "Manufacturer": "Monaliz Foods Pvt Ltd, Mumbai 400001",
            "Net Quantity": "50 g",
            "MRP": "MRP Rs. 28.00 (incl. of all taxes)",
            "Unit Sale Price": "Rs. 0.56 per g",
            "Manufacturing Date": "10/2025",
            "Consumer Care": "Helpline: 022-28253651 | Email: care@monaliz.com"
        }
        fields_detail = {
            "Net Quantity": {
                "confidence": 45.0,  # Below 70% threshold
                "source": "Local OCR",
                "has_conflict": False,
                "conflict_detail": None
            },
            "MRP": {
                "confidence": 95.0,
                "source": "Gemini Vision AI",
                "has_conflict": True,
                "conflict_detail": "Gemini detected 'Rs. 28.00' vs OCR detected 'Rs. 88.00'"
            }
        }
        res = validate_all_fields(fields, fields_detail)
        self.assertEqual(res["quantity"]["confidence"], 45.0)
        self.assertTrue(res["quantity"]["requires_review"])
        self.assertTrue(any("Low extraction confidence" in w for w in res["quantity"]["warnings"]))

        self.assertEqual(res["mrp"]["confidence"], 95.0)
        self.assertTrue(res["mrp"]["has_conflict"])
        self.assertTrue(any("Extraction discrepancy detected" in w for w in res["mrp"]["warnings"]))

    def test_detect_extraction_conflicts(self):
        # Semantic divergence detection
        gemini_fields = {
            "Product Name": "Monaliz Double Action Baking Powder",
            "Net Quantity": "50 g",
            "MRP": "MRP Rs. 28.00 (incl. of all taxes)",
            "Manufacturing Date": "15/10/2025",
            "Batch Number": "AB1025"
        }
        # Disagreeing OCR results on Net Quantity and Batch Number
        ocr_conflicting = {
            "Product Name": "Baking Powder",  # Substring match -> NOT a conflict
            "Net Quantity": "100 g",          # Semantic disagreement -> CONFLICT!
            "MRP": "Rs. 28.00",               # Same numeric price -> NOT a conflict
            "Manufacturing Date": "10/2025",  # Same month/year -> NOT a conflict
            "Batch Number": "XY999"           # Divergent batch -> CONFLICT!
        }
        conflicts = detect_extraction_conflicts(gemini_fields, ocr_conflicting)
        self.assertIn("Net Quantity", conflicts)
        self.assertIn("Batch Number", conflicts)
        self.assertNotIn("Product Name", conflicts)
        self.assertNotIn("MRP", conflicts)
        self.assertNotIn("Manufacturing Date", conflicts)


if __name__ == "__main__":
    unittest.main()
