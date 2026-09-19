"""
Unit tests for Legal Metrology PCR 2011 Compliance Engine (compliance_rules.py).
Tests Rule 6(1)(a)-(m) evaluations, explainable scoring, and critical failure detection.
"""

import unittest
from config import get_extraction_confidence_threshold, set_extraction_confidence_threshold
from compliance_rules import evaluate_compliance, evaluate_net_quantity


class TestComplianceEngine(unittest.TestCase):

    def setUp(self):
        self.compliant_sample = {
            "Product Name": "Monaliz Double Action Baking Powder",
            "Manufacturer": "Monaliz Food Products Pvt Ltd, Plot 14, MIDC Industrial Area, Mumbai 400093",
            "Net Quantity": "50 g",
            "MRP": "MRP Rs. 28.00 (incl. of all taxes)",
            "Unit Sale Price": "Rs. 0.56 per g",
            "Manufacturing Date": "15/10/2025",
            "Batch Number": "AB1025",
            "Best Before / Expiry": "14/10/2027",
            "Consumer Care": "Helpline: +91 2248253651 | Email: grievance@monalizfoods.com",
            "Country of Origin": "India"
        }

    def test_fully_compliant_product(self):
        res = evaluate_compliance(self.compliant_sample)
        self.assertEqual(res["status"], "COMPLIANT")
        self.assertGreaterEqual(res["score"], 90.0)
        self.assertEqual(res["failed_count"], 0)
        self.assertIn("Mandatory PDP", res["category_scores"])
        self.assertIn("Price & USP", res["category_scores"])

    def test_missing_generic_commodity_name(self):
        # Missing OCR text should result in NOT_DETECTED and REQUIRES_REVIEW, not an affirmative legal conviction
        sample = dict(self.compliant_sample)
        sample["Product Name"] = ""
        res = evaluate_compliance(sample)
        self.assertEqual(res["status"], "REQUIRES_REVIEW")
        self.assertTrue(any(e["rule"] == "Rule 6(1)(a)" and e["status"] == "NOT_DETECTED" for e in res["audit_table"]))

    def test_prohibited_unit_abbreviation(self):
        # Affirmative legal violation (prohibited abbreviation 'gms') must result in NON-COMPLIANT
        sample = dict(self.compliant_sample)
        sample["Net Quantity"] = "50 gms"
        res = evaluate_compliance(sample)
        self.assertEqual(res["status"], "NON-COMPLIANT")
        rule3 = next(e for e in res["audit_table"] if e["rule"] == "Rule 6(1)(c)")
        self.assertEqual(rule3["status"], "NON_COMPLIANT")
        self.assertIn("gms", rule3["reason"])

    def test_missing_mrp_failure(self):
        # Missing MRP declaration results in NOT_DETECTED with physical review required
        sample = dict(self.compliant_sample)
        sample["MRP"] = ""
        res = evaluate_compliance(sample)
        self.assertEqual(res["status"], "REQUIRES_REVIEW")
        rule4 = next(e for e in res["audit_table"] if e["rule"] == "Rule 6(1)(da)")
        self.assertEqual(rule4["status"], "NOT_DETECTED")

    def test_usp_exemption_1kg_1litre(self):
        # Proviso to Rule 6(1)(m): When MRP = USP (1 kg, 1 litre, 1 metre, 1 unit), USP declaration is exempt
        sample = dict(self.compliant_sample)
        sample["Net Quantity"] = "1 kg"
        sample["MRP"] = "MRP Rs. 120.00 (incl. of all taxes)"
        sample["Unit Sale Price"] = ""  # No separate USP declared
        res = evaluate_compliance(sample)
        rule_usp = next(e for e in res["audit_table"] if e["rule"] == "Rule 6(1)(m)")
        self.assertEqual(rule_usp["status"], "NOT_APPLICABLE")
        self.assertIn("Proviso to Rule 6(1)(m)", rule_usp["reason"])
        self.assertEqual(res["status"], "COMPLIANT")
        self.assertEqual(res["score"], 100.0)

    def test_rule26_small_package_exemption(self):
        # Rule 26(a): Packages <= 10g or <= 10ml are exempt from Unit Sale Price declaration
        sample = dict(self.compliant_sample)
        sample["Net Quantity"] = "5 g"
        sample["MRP"] = "MRP Rs. 5.00 (incl. of all taxes)"
        sample["Unit Sale Price"] = ""
        res = evaluate_compliance(sample)
        rule_usp = next(e for e in res["audit_table"] if e["rule"] == "Rule 6(1)(m)")
        self.assertEqual(rule_usp["status"], "NOT_APPLICABLE")
        self.assertIn("Rule 26(a)", rule_usp["reason"])

    def test_non_food_expiry_exemption(self):
        # Rule 6(1)(g): Expiry / Best Before is mandatory ONLY for food/perishables; exempt for non-food
        non_food_sample = {
            "Product Name": "Philips LED Bulb 9W",
            "Manufacturer": "Signify Innovations India Ltd, Gurgaon 122002",
            "Net Quantity": "1 number",
            "MRP": "MRP Rs. 149.00 (incl. of all taxes)",
            "Unit Sale Price": "",
            "Manufacturing Date": "01/2026",
            "Batch Number": "PH2026",
            "Best Before / Expiry": "",  # Non-food commodities do not bear expiry
            "Consumer Care": "Helpline: 1800-102-2929 | Email: care@signify.com",
            "Country of Origin": "India"
        }
        res = evaluate_compliance(non_food_sample, commodity_category="non_food")
        rule_expiry = next(e for e in res["audit_table"] if e["rule"] == "Rule 6(1)(g)")
        self.assertEqual(rule_expiry["status"], "NOT_APPLICABLE")
        self.assertIn("non-perishable", rule_expiry["reason"])
        self.assertEqual(res["status"], "COMPLIANT")

    def test_not_detected_vs_non_compliant(self):
        # Differentiates unreadable OCR text from an affirmative illegal declaration
        # 1. Missing field in partial crop -> NOT_DETECTED -> overall REQUIRES_REVIEW
        partial_sample = dict(self.compliant_sample)
        partial_sample["Batch Number"] = ""
        partial_res = evaluate_compliance(partial_sample)
        rule_batch = next(e for e in partial_res["audit_table"] if e["rule"] == "PCR Traceability")
        self.assertEqual(rule_batch["status"], "NOT_DETECTED")
        self.assertEqual(partial_res["status"], "REQUIRES_REVIEW")

        # 2. Prohibited illegal unit symbol -> NON_COMPLIANT -> overall NON-COMPLIANT
        illegal_sample = dict(self.compliant_sample)
        illegal_sample["Net Quantity"] = "50 gms"
        illegal_res = evaluate_compliance(illegal_sample)
        rule_qty = next(e for e in illegal_res["audit_table"] if e["rule"] == "Rule 6(1)(c)")
        self.assertEqual(rule_qty["status"], "NON_COMPLIANT")
        self.assertEqual(illegal_res["status"], "NON-COMPLIANT")

    def test_unit_sale_price_evaluation(self):
        sample = dict(self.compliant_sample)
        sample["Unit Sale Price"] = "Rs. 0.56 per g"
        res = evaluate_compliance(sample)
        rule_usp = next(e for e in res["audit_table"] if e["rule"] == "Rule 6(1)(m)")
        self.assertEqual(rule_usp["status"], "COMPLIANT")

    def test_category_scores_structure(self):
        res = evaluate_compliance(self.compliant_sample)
        cat_scores = res["category_scores"]
        required_cats = [
            "Mandatory PDP", "Manufacturer & Address", "Quantity & Metric Units",
            "Price & USP", "Dates & Shelf Life", "Consumer Grievance", "Origin & Traceability"
        ]
        for cat in required_cats:
            self.assertIn(cat, cat_scores)
            self.assertIn("percentage", cat_scores[cat])
            self.assertIn("passed", cat_scores[cat])


    def test_compliant_50g_and_2kg(self):
        # 1. 50 g mass under 1 kg
        res_50g = evaluate_compliance(self.compliant_sample)
        self.assertEqual(res_50g["status"], "COMPLIANT")
        rule_qty_50g = next(e for e in res_50g["audit_table"] if e["rule"] == "Rule 6(1)(c)")
        self.assertEqual(rule_qty_50g["status"], "COMPLIANT")

        # 2. 2 kg mass over 1 kg
        sample_2kg = dict(self.compliant_sample)
        sample_2kg["Net Quantity"] = "2 kg"
        sample_2kg["MRP"] = "MRP Rs. 200.00 (incl. of all taxes)"
        sample_2kg["Unit Sale Price"] = "Rs. 100.00 per kg"
        res_2kg = evaluate_compliance(sample_2kg)
        self.assertEqual(res_2kg["status"], "COMPLIANT")
        rule_qty_2kg = next(e for e in res_2kg["audit_table"] if e["rule"] == "Rule 6(1)(c)")
        self.assertEqual(rule_qty_2kg["status"], "COMPLIANT")

    def test_case_sensitive_symbol_advisory_requires_review(self):
        # Upper-case mass symbol '50 G' or '2 KG' is a formatting advisory under Rule 13,
        # which must result in REQUIRES_REVIEW rather than an affirmative NON-COMPLIANT conviction
        sample_upper = dict(self.compliant_sample)
        sample_upper["Net Quantity"] = "50 G"
        res = evaluate_compliance(sample_upper)
        rule_qty = next(e for e in res["audit_table"] if e["rule"] == "Rule 6(1)(c)")
        self.assertEqual(rule_qty["status"], "REQUIRES_REVIEW")
        self.assertIn("Formatting Advisory under Rule 13", rule_qty["reason"])
        self.assertEqual(res["status"], "REQUIRES_REVIEW")

    def test_rule11_soap_when_packed_permitted(self):
        # Under Proviso to Rule 11(2), 'when packed' is permitted for soaps/camphor
        soap_sample = dict(self.compliant_sample)
        soap_sample["Product Name"] = "Lux Bathing Soap"
        soap_sample["Net Quantity"] = "125 g when packed"
        soap_sample["MRP"] = "MRP Rs. 50.00 (incl. of all taxes)"
        soap_sample["Unit Sale Price"] = "Rs. 0.40 per g"
        res = evaluate_compliance(soap_sample)
        rule_qty = next(e for e in res["audit_table"] if e["rule"] == "Rule 6(1)(c)")
        self.assertEqual(rule_qty["status"], "COMPLIANT")
        self.assertIn("Proviso to Rule 11(2)", rule_qty["reason"])
        self.assertEqual(res["status"], "COMPLIANT")

    def test_rule11_non_soap_when_packed_non_compliant(self):
        # Baking powder is not susceptible to climatic variations; 'when packed' is illegal
        baking_sample = dict(self.compliant_sample)
        baking_sample["Product Name"] = "Monaliz Baking Powder"
        baking_sample["Net Quantity"] = "50 g when packed"
        res = evaluate_compliance(baking_sample)
        rule_qty = next(e for e in res["audit_table"] if e["rule"] == "Rule 6(1)(c)")
        self.assertEqual(rule_qty["status"], "NON_COMPLIANT")
        self.assertEqual(res["status"], "NON-COMPLIANT")

    def test_rule11_approx_is_always_non_compliant(self):
        # 'approx' is strictly prohibited across all commodities, even soaps
        soap_approx = dict(self.compliant_sample)
        soap_approx["Product Name"] = "Lux Bathing Soap"
        soap_approx["Net Quantity"] = "approx 125 g"
        res = evaluate_compliance(soap_approx)
        rule_qty = next(e for e in res["audit_table"] if e["rule"] == "Rule 6(1)(c)")
        self.assertEqual(rule_qty["status"], "NON_COMPLIANT")
        self.assertEqual(res["status"], "NON-COMPLIANT")

    # =========================================================================
    # STAGE 3: Extraction Confidence & Status Integrity Automated Tests
    # =========================================================================

    def test_unreadable_field_is_not_detected(self):
        """1. Missing or unreadable field produces NOT_DETECTED, not an affirmative NON_COMPLIANT conviction."""
        sample = dict(self.compliant_sample)
        sample["Batch Number"] = ""
        res = evaluate_compliance(sample)
        rule_batch = next(e for e in res["audit_table"] if e["rule_id"] == "PCR-10")
        self.assertEqual(rule_batch["status"], "NOT_DETECTED")
        self.assertEqual(rule_batch["score"], 0)
        self.assertEqual(res["failed_count"], 0)
        self.assertGreaterEqual(res["not_detected_count"], 1)
        self.assertEqual(res["status"], "REQUIRES_REVIEW")

    def test_detected_invalid_is_non_compliant(self):
        """2. Detected declaration with affirmative legal violation produces NON_COMPLIANT and fails product."""
        sample = dict(self.compliant_sample)
        sample["Net Quantity"] = "50 gms"
        fields_detail = {
            "Net Quantity": {
                "value": "50 gms",
                "confidence": 95.0,
                "source": "Local OCR (Tesseract)"
            }
        }
        res = evaluate_compliance(sample, fields_detail=fields_detail)
        rule_qty = next(e for e in res["audit_table"] if e["rule_id"] == "PCR-03")
        self.assertEqual(rule_qty["status"], "NON_COMPLIANT")
        self.assertEqual(res["status"], "NON-COMPLIANT")
        self.assertGreaterEqual(res["failed_count"], 1)

    def test_low_confidence_extraction_requires_review(self):
        """3. Low confidence extraction (< threshold) produces REQUIRES_REVIEW for both valid and invalid text."""
        sample = dict(self.compliant_sample)
        fields_detail = {
            "Net Quantity": {
                "value": "50 g",
                "confidence": 50.0,  # Below default 70.0% threshold
                "source": "Local OCR (Tesseract)"
            }
        }
        res = evaluate_compliance(sample, fields_detail=fields_detail)
        rule_qty = next(e for e in res["audit_table"] if e["rule_id"] == "PCR-03")
        self.assertEqual(rule_qty["status"], "REQUIRES_REVIEW")
        self.assertIn("Low extraction confidence", rule_qty["reason"])
        self.assertEqual(res["status"], "REQUIRES_REVIEW")

        # Crucial Legal Safeguard: Do not convict on low confidence / OCR noise
        sample_low_defect = dict(self.compliant_sample)
        sample_low_defect["Net Quantity"] = "50 gms"
        fields_detail_defect = {
            "Net Quantity": {
                "value": "50 gms",
                "confidence": 45.0,  # Low confidence reading of potential violation
                "source": "Local OCR (Tesseract)"
            }
        }
        res_defect = evaluate_compliance(sample_low_defect, fields_detail=fields_detail_defect)
        rule_qty_defect = next(e for e in res_defect["audit_table"] if e["rule_id"] == "PCR-03")
        self.assertEqual(rule_qty_defect["status"], "REQUIRES_REVIEW")
        self.assertIn("Potential statutory irregularity detected", rule_qty_defect["reason"])
        self.assertEqual(res_defect["failed_count"], 0)

    def test_high_confidence_valid_is_compliant(self):
        """4. High confidence valid declaration (>= threshold) produces COMPLIANT."""
        sample = dict(self.compliant_sample)
        fields_detail = {
            "Net Quantity": {
                "value": "50 g",
                "confidence": 95.0,
                "source": "Gemini Vision AI"
            }
        }
        res = evaluate_compliance(sample, fields_detail=fields_detail)
        rule_qty = next(e for e in res["audit_table"] if e["rule_id"] == "PCR-03")
        self.assertEqual(rule_qty["status"], "COMPLIANT")
        self.assertEqual(rule_qty["confidence"], 95.0)

    def test_conflicting_gemini_ocr_values_requires_review(self):
        """5. Conflicting extraction results between engines flag REQUIRES_REVIEW with explanation."""
        sample = dict(self.compliant_sample)
        fields_detail = {
            "Net Quantity": {
                "value": "50 g",
                "confidence": 85.0,
                "has_conflict": True,
                "conflict_detail": "Gemini detected '50 g' vs Local OCR detected '100 g'"
            }
        }
        res = evaluate_compliance(sample, fields_detail=fields_detail)
        rule_qty = next(e for e in res["audit_table"] if e["rule_id"] == "PCR-03")
        self.assertEqual(rule_qty["status"], "REQUIRES_REVIEW")
        self.assertTrue(rule_qty["has_conflict"])
        self.assertIn("Extraction discrepancy", rule_qty["reason"])
        self.assertIn("Gemini detected '50 g' vs Local OCR detected '100 g'", rule_qty["reason"])
        self.assertEqual(res["status"], "REQUIRES_REVIEW")

    def test_confidence_propagation(self):
        """6. Measurable confidence and extraction source are propagated to the audit table."""
        sample = dict(self.compliant_sample)
        fields_detail = {
            "Net Quantity": {
                "value": "50 g",
                "confidence": 92.5,
                "source": "Gemini Vision AI"
            },
            "MRP": {
                "value": "MRP Rs. 28.00 (incl. of all taxes)",
                "confidence": 88.0,
                "source": "Local OCR (Tesseract)"
            },
            "Batch Number": {
                "value": "AB1025",
                "confidence": None,
                "source": "Manual Input"
            }
        }
        res = evaluate_compliance(sample, fields_detail=fields_detail)
        audit = {e["rule_id"]: e for e in res["audit_table"]}

        self.assertEqual(audit["PCR-03"]["confidence"], 92.5)
        self.assertEqual(audit["PCR-03"]["extraction_source"], "Gemini Vision AI")

        self.assertEqual(audit["PCR-04"]["confidence"], 88.0)
        self.assertEqual(audit["PCR-04"]["extraction_source"], "Local OCR (Tesseract)")

        self.assertIsNone(audit["PCR-10"]["confidence"])
        self.assertEqual(audit["PCR-10"]["extraction_source"], "Manual Input")

    def test_configurable_confidence_threshold(self):
        """7. Dynamically changing the confidence threshold adjusts the compliance evaluation boundary."""
        original_threshold = get_extraction_confidence_threshold()
        try:
            sample = dict(self.compliant_sample)
            fields_detail = {
                "Net Quantity": {
                    "value": "50 g",
                    "confidence": 75.0,
                    "source": "Local OCR (Tesseract)"
                }
            }
            # At 70% threshold, 75% confidence is COMPLIANT
            set_extraction_confidence_threshold(70.0)
            res_70 = evaluate_compliance(sample, fields_detail=fields_detail)
            rule_qty_70 = next(e for e in res_70["audit_table"] if e["rule_id"] == "PCR-03")
            self.assertEqual(rule_qty_70["status"], "COMPLIANT")

            # At 80% threshold, 75% confidence is below threshold -> REQUIRES_REVIEW
            set_extraction_confidence_threshold(80.0)
            res_80 = evaluate_compliance(sample, fields_detail=fields_detail)
            rule_qty_80 = next(e for e in res_80["audit_table"] if e["rule_id"] == "PCR-03")
            self.assertEqual(rule_qty_80["status"], "REQUIRES_REVIEW")
            self.assertIn("75.0% < 80.0% threshold", rule_qty_80["reason"])
        finally:
            set_extraction_confidence_threshold(original_threshold)

    def test_compliance_score_treatment_of_uncertain_fields(self):
        """8. Compliance score explainability treats NOT_DETECTED/REQUIRES_REVIEW distinct from NON_COMPLIANT."""
        # Product with unreadable field: status is REQUIRES_REVIEW, 0 legal violations
        sample_missing = dict(self.compliant_sample)
        sample_missing["Batch Number"] = ""
        res_missing = evaluate_compliance(sample_missing)
        self.assertEqual(res_missing["status"], "REQUIRES_REVIEW")
        self.assertEqual(res_missing["failed_count"], 0)
        self.assertEqual(res_missing["not_detected_count"], 1)
        self.assertIn("score_explanation", res_missing)
        self.assertEqual(res_missing["score_explanation"]["statutory_violations"], 0)
        self.assertEqual(res_missing["score_explanation"]["not_detected_count"], 1)

        # Product with affirmative violation: status is NON-COMPLIANT, failed_count >= 1
        sample_illegal = dict(self.compliant_sample)
        sample_illegal["Net Quantity"] = "50 gms"
        res_illegal = evaluate_compliance(sample_illegal)
        self.assertEqual(res_illegal["status"], "NON-COMPLIANT")
        self.assertGreaterEqual(res_illegal["failed_count"], 1)
        self.assertEqual(res_illegal["score_explanation"]["statutory_violations"], 1)


if __name__ == "__main__":
    unittest.main()
