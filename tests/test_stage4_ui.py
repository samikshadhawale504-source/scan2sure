"""
Comprehensive integration tests for Stage 4: UI & Inspector Report Updates.
Verifies field provenance indicators, final inspection summary card,
statutory declaration audit table (7 columns), official inspection report certificate,
and historical inspection log.
"""

import unittest
import json
import io
from app import app
from models.database import save_inspection, get_inspection, list_inspections


class TestStage4InspectorUI(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_field_provenance_rendering_in_edit_form(self):
        """1. Field provenance indicators render properly in the Inspector Verification Console."""
        # Simulate an audit check with fields_detail containing multiple sources & confidence values
        fields_detail = {
            "Product Name": {
                "value": "Monaliz Baking Powder",
                "evidence": "Front label text: Monaliz Baking Powder",
                "confidence": 94.5,
                "source": "Gemini Vision AI",
                "has_conflict": True,
                "conflict_detail": "Gemini saw 'Monaliz Baking Powder', Local OCR saw 'Baking Powder'"
            },
            "Net Quantity": {
                "value": "50 g",
                "evidence": "OCR detected 50 g",
                "confidence": 62.0,  # Low confidence < 70%
                "source": "Local OCR",
                "has_conflict": False,
                "conflict_detail": None
            },
            "MRP": {
                "value": "MRP Rs. 28.00 (incl. of all taxes)",
                "evidence": "Inspector verified price",
                "confidence": None,  # User edited / unmeasured
                "source": "Inspector Refined",
                "has_conflict": False,
                "conflict_detail": None
            }
        }

        response = self.client.post("/", data={
            "action": "check",
            "product_name": "Monaliz Baking Powder",
            "manufacturer": "Monaliz Food Product, 12 Industrial Area, Mumbai 400001",
            "net_quantity": "50 g",
            "mrp": "MRP Rs. 28.00 (incl. of all taxes)",
            "unit_sale_price": "Rs. 0.56 / g",
            "manufacturing_date": "10/2025",
            "consumer_care": "Phone: 9820012345 | Email: care@monaliz.com",
            "fields_detail_json": json.dumps(fields_detail)
        })

        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")

        # Provenance tags
        self.assertIn("source-tag-gemini", html)
        self.assertIn("Gemini Vision AI", html)
        self.assertIn("source-tag-ocr", html)
        self.assertIn("Local OCR", html)
        self.assertIn("source-tag-user", html)
        self.assertIn("User Verified / Edited", html)

        # Confidence pills
        self.assertIn("conf-pill-high", html)
        self.assertIn("94.5%", html)
        self.assertIn("conf-pill-low", html)
        self.assertIn("62.0%", html)
        self.assertIn("conf-pill-na", html)
        self.assertIn("Unmeasured", html)

        # Conflict badge
        self.assertIn("conflict-badge", html)
        self.assertIn("Conflict", html)

    def test_final_inspection_summary_card_and_explainability(self):
        """2. Final inspection summary card displays overall status, explainable score note, and next steps."""
        response = self.client.post("/", data={
            "action": "check",
            "product_name": "Compliant Test Commodity",
            "manufacturer": "Standard Manufacturer, Mumbai 400001",
            "net_quantity": "500 g",
            "mrp": "MRP Rs. 100.00 (incl. of all taxes)",
            "unit_sale_price": "Rs. 0.20 / g",
            "manufacturing_date": "01/2026",
            "consumer_care": "care@example.com | 1800-00-1122",
            "country_of_origin": "India",
            "batch_number": "B1001",
            "expiry_or_best_before": "01/2028"
        })

        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")

        # Summary header & badge
        self.assertIn("badge-compliant", html)
        self.assertIn("COMPLIANT", html)

        # Score & explainability
        self.assertIn("Compliance Index", html)
        self.assertIn("Statutory Scoring Transparency", html)
        self.assertIn("not represent a statistical probability of legal compliance", html)

        # Actionable next steps for inspector
        self.assertIn("inspector-actions-box", html)
        self.assertIn("Actionable Compliance Directives", html)

    def test_statutory_audit_table_7_columns(self):
        """3. Statutory declaration audit table renders complete 7 columns with explicit statuses."""
        response = self.client.post("/", data={
            "action": "check",
            "product_name": "Sample Audit Product",
            "manufacturer": "Sample Manufacturer, Delhi 110001",
            "net_quantity": "100 g",
            "mrp": "Rs. 50.00 (incl. of all taxes)",  # Missing 'MRP' prefix -> REQUIRES_REVIEW
            "unit_sale_price": "",  # Not detected -> NOT_DETECTED
            "manufacturing_date": "11/2025",
            "consumer_care": "care@test.com"
        })

        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")

        # 7 Column Headers
        self.assertIn("Rule ID &amp; Requirement", html)
        self.assertIn("PCR 2011 Rule", html)
        self.assertIn("Extracted Evidence &amp; Value", html)
        self.assertIn("Confidence &amp; Source", html)
        self.assertIn("Compliance Status", html)
        self.assertIn("Inspector Finding &amp; Reason", html)
        self.assertIn("Recommended Action", html)

        # Explicit statuses present
        self.assertIn("badge-compliant", html)
        self.assertIn("badge-requiresreview", html)
        self.assertIn("badge-notdetected", html)

    def test_not_applicable_statutory_exemption_rendering(self):
        """4. Statutory exemptions (NOT_APPLICABLE) render with distinct badge and legal citation."""
        # 1 kg package is exempt from Rule 6(1)(m) Unit Sale Price under Proviso to Rule 6(1)(m)
        response = self.client.post("/", data={
            "action": "check",
            "product_name": "Wheat Flour",
            "manufacturer": "Flour Mills Ltd, Kanpur 208001",
            "net_quantity": "1 kg",
            "mrp": "MRP Rs. 60.00 (incl. of all taxes)",
            "manufacturing_date": "12/2025",
            "consumer_care": "care@flour.com"
        })

        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")

        # Must render NOT_APPLICABLE with distinct badge
        self.assertIn("badge-notapplicable", html)
        self.assertIn("NOT_APPLICABLE", html)
        self.assertIn("status-row-notapplicable", html)
        self.assertIn("Proviso to Rule 6(1)(m)", html)

    def test_official_inspection_report_certificate(self):
        """5. Official report.html certificate renders complete government format."""
        # Create an inspection record
        insp_id = save_inspection(
            product_name="Certified Packaged Tea",
            image_filename="test_label.jpg",
            engine_used="Gemini Vision AI",
            confidence=91.5,
            overall_status="NON-COMPLIANT",
            compliance_score=65.0,
            input_source="image",
            fields={
                "Product Name": "Certified Packaged Tea",
                "Net Quantity": "500 gms",  # Illegal symbol
                "MRP": "MRP Rs. 250 (incl. of all taxes)"
            },
            validation_results={"is_valid": False, "issues": ["Illegal unit symbol 'gms'"]},
            compliance_results={
                "status": "NON-COMPLIANT",
                "score": 65.0,
                "passed_count": 5,
                "failed_count": 1,
                "warning_count": 2,
                "not_detected_count": 2,
                "not_applicable_count": 0,
                "commodity_category": "food",
                "score_explanation": {
                    "summary": "Affirmative statutory violations detected. Package is legally non-compliant.",
                    "statutory_violations": 1,
                    "requires_review_count": 2,
                    "not_detected_count": 2,
                    "exempt_count": 0,
                    "compliant_count": 5
                },
                "audit_table": [
                    {
                        "rule_id": "PCR-03",
                        "rule": "Rule 6(1)(c)",
                        "requirement": "Net Quantity Declaration & Metric Units",
                        "category": "Quantity & Metric Units",
                        "extracted_value": "500 gms",
                        "extracted_evidence": "500 gms",
                        "extraction_source": "Gemini Vision AI",
                        "confidence": 91.5,
                        "status": "NON_COMPLIANT",
                        "reason": "Illegal unit symbol 'gms' violates Rule 13(2).",
                        "recommended_action": "Correct packaging to use standard legal metric symbol 'g' under Rule 13."
                    }
                ]
            }
        )

        response = self.client.get(f"/inspection/{insp_id}/report")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")

        # Government Emblem & Title
        self.assertIn("GOVERNMENT OF INDIA", html)
        self.assertIn("DIRECTORATE OF LEGAL METROLOGY", html)
        self.assertIn("STATUTORY COMPLIANCE INSPECTION REPORT", html)

        # Dossier details
        self.assertIn(insp_id, html)
        self.assertIn("Certified Packaged Tea", html)
        self.assertIn("Gemini Vision AI", html)
        self.assertIn("91.5%", html)

        # Image thumbnail container
        self.assertIn("test_label.jpg", html)
        self.assertIn("cert-image-container", html)

        # Full audit table
        self.assertIn("Rule 6(1)(c)", html)
        self.assertIn("500 gms", html)
        self.assertIn("badge-noncompliant", html)
        self.assertIn("NON-COMPLIANT", html)

        # Statutory Notice & Authority Clause
        self.assertIn("Section 36(1) of the Legal Metrology Act, 2009", html)
        self.assertIn("Section 15", html)

        # Signature & Seal Block
        self.assertIn("cert-signature-block", html)
        self.assertIn("Inspecting Enforcement Officer", html)
        self.assertIn("Authorized Seal &amp; Endorsement", html)

    def test_history_log_rendering(self):
        """6. Inspection log history.html renders formatted status rows and certificates."""
        # Ensure at least one record in history
        save_inspection(
            product_name="History Test Item",
            image_filename=None,
            engine_used="Local OCR",
            confidence=85.0,
            overall_status="REQUIRES_REVIEW",
            compliance_score=75.0,
            input_source="manual",
            fields={"Product Name": "History Test Item"},
            validation_results={},
            compliance_results={"status": "REQUIRES_REVIEW", "score": 75.0, "audit_table": []}
        )

        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")

        self.assertIn("Inspection Audit Log", html)
        self.assertIn("History Test Item", html)
        self.assertIn("badge-requiresreview", html)
        self.assertIn("REQUIRES REVIEW", html)
        self.assertIn("Certificate", html)


if __name__ == "__main__":
    unittest.main()
