"""
Unit and integration tests for Hybrid Extraction Pipeline (pipeline/hybrid_extractor.py)
and Gemini Multimodal Vision (gemini_vision.py).

Guarantees 100% mocked offline execution: NO live API calls or network connections.
Covers all 7 statutory and architectural cases:
1. Valid structured Gemini response
2. Malformed Gemini response (invalid JSON / syntax error) -> safe error and auto-fallback
3. Missing fields in Gemini response -> safe defaulting and NOT_DETECTED status
4. Low-confidence Gemini response -> properly propagated as REQUIRES_REVIEW
5. Conflicting OCR and Gemini extraction -> conflict detection and audit review flag
6. Network / API failure -> graceful auto-fallback to Local OCR
7. Missing API key in auto mode -> seamless routing to Local OCR
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import json

from gemini_vision import extract_with_gemini, is_gemini_available, get_api_key
from pipeline.hybrid_extractor import run_hybrid_extraction, detect_extraction_conflicts, normalize_field_for_comparison
from compliance_rules import evaluate_compliance


class TestHybridExtractionAndGemini(unittest.TestCase):

    def setUp(self):
        self.sample_image = os.path.join(os.path.dirname(os.path.dirname(__file__)), "product.jpg")
        self.patcher_mistral = patch("pipeline.hybrid_extractor.get_mistral_api_key", return_value="")
        self.patcher_mistral.start()
        self.mock_ocr_data = {
            "product_name": "Monaliz Baking Powder",
            "manufacturer": "Monaliz Food Product",
            "net_quantity": "50 g",
            "mrp": "Rs. 28",
            "unit_sale_price": "Rs. 0.56 / g",
            "manufacturing_date": "15/10/2025",
            "batch_number": "AB1025",
            "expiry_or_best_before": "14/10/2027",
            "consumer_care": "care@monaliz.com",
            "country_of_origin": "India",
            "confidence": 72.0,
            "processing_time_sec": 1.2,
            "raw_text": "Monaliz Baking Powder 50g Rs 28 MFD 15/10/2025"
        }

    def tearDown(self):
        self.patcher_mistral.stop()

    # =========================================================================
    # CASE 1: Valid structured Gemini response
    # =========================================================================
    def test_case1_valid_structured_gemini_response(self):
        """1. Valid structured Gemini response parses correctly into all statutory packaging fields."""
        gemini_json = {
            "product_name": {"value": "Monaliz Double Action Baking Powder", "evidence": "Monaliz Double Action Baking Powder", "confidence": 99.0, "requires_review": False},
            "manufacturer": {"value": "Monaliz Food Products Pvt Ltd", "evidence": "Monaliz Food Products Pvt Ltd", "confidence": 98.0, "requires_review": False},
            "manufacturer_address": {"value": "Plot 14, MIDC, Mumbai 400093", "evidence": "Plot 14, MIDC, Mumbai 400093", "confidence": 97.0, "requires_review": False},
            "net_quantity": {"value": "50 g", "evidence": "Net Qty: 50 g", "confidence": 99.0, "requires_review": False},
            "mrp": {"value": "MRP Rs. 28.00 (incl. of all taxes)", "evidence": "MRP Rs. 28.00 incl. of all taxes", "confidence": 98.0, "requires_review": False},
            "unit_sale_price": {"value": "Rs. 0.56 per g", "evidence": "USP: Rs. 0.56 per g", "confidence": 96.0, "requires_review": False},
            "manufacturing_date": {"value": "15/10/2025", "evidence": "MFD 15/10/2025", "confidence": 97.0, "requires_review": False},
            "batch_number": {"value": "AB1025", "evidence": "B.No. AB1025", "confidence": 95.0, "requires_review": False},
            "best_before": {"value": "14/10/2027", "evidence": "Best Before 14/10/2027", "confidence": 96.0, "requires_review": False},
            "consumer_care_phone": {"value": "+91 2248253651", "evidence": "Ph: +91 2248253651", "confidence": 97.0, "requires_review": False},
            "consumer_care_email": {"value": "care@monaliz.com", "evidence": "care@monaliz.com", "confidence": 98.0, "requires_review": False},
            "country_of_origin": {"value": "India", "evidence": "Made in India", "confidence": 99.0, "requires_review": False}
        }

        mock_resp = MagicMock()
        mock_resp.text = json.dumps(gemini_json)

        with patch("gemini_vision.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            res = extract_with_gemini(self.sample_image, api_key="valid-mock-key")

            self.assertTrue(res["success"])
            self.assertEqual(res["engine"], "gemini")
            self.assertGreaterEqual(res["confidence"], 90.0)
            self.assertEqual(res["fields"]["Product Name"], "Monaliz Double Action Baking Powder")
            self.assertIn("Mumbai 400093", res["fields"]["Manufacturer"])
            self.assertEqual(res["fields"]["Net Quantity"], "50 g")
            self.assertIn("care@monaliz.com", res["fields"]["Consumer Care"])
            self.assertIn("+91 2248253651", res["fields"]["Consumer Care"])

            # Verify evidence and field-level metadata
            self.assertEqual(res["fields_detail"]["Net Quantity"]["evidence"], "Net Qty: 50 g")
            self.assertFalse(res["fields_detail"]["Net Quantity"]["requires_review"])

    # =========================================================================
    # CASE 2: Malformed Gemini response (invalid JSON / syntax error)
    # =========================================================================
    def test_case2_malformed_gemini_response_handled_cleanly(self):
        """2a. extract_with_gemini catches invalid JSON string without throwing an unhandled exception."""
        mock_resp = MagicMock()
        mock_resp.text = "MALFORMED_OUTPUT_NOT_JSON {{{: invalid"

        with patch("gemini_vision.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            res = extract_with_gemini(self.sample_image, api_key="valid-mock-key")

            self.assertFalse(res["success"])
            self.assertIn("error", res)
            self.assertIsNotNone(res["error"])

    def test_case2_malformed_gemini_triggers_auto_fallback(self):
        """2b. In 'auto' mode, a malformed Gemini response triggers automatic fallback to Local OCR."""
        with patch("pipeline.hybrid_extractor.get_api_key", return_value="valid-mock-key"):
            with patch("pipeline.hybrid_extractor.extract_with_gemini") as mock_gemini:
                mock_gemini.return_value = {
                    "success": False,
                    "error": "Expecting value: line 1 column 1 (char 0)",
                    "engine": "gemini",
                    "processing_time_sec": 0.4
                }
                with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=self.mock_ocr_data):
                    res = run_hybrid_extraction(self.sample_image, engine_mode="auto", gemini_api_key="valid-mock-key")

                    self.assertTrue(res["success"])
                    self.assertTrue(res["fallback_used"])
                    self.assertIn("Fallback", res["engine"])
                    self.assertIn("Expecting value", res["fallback_note"])
                    self.assertEqual(res["fields"]["Net Quantity"], "50 g")
                    self.assertEqual(res["fields_detail"]["Net Quantity"]["source"], "Local OCR (Tesseract)")

    # =========================================================================
    # CASE 3: Missing fields in Gemini response
    # =========================================================================
    def test_case3_missing_fields_in_gemini_response(self):
        """3. Incomplete Gemini response safely populates empty strings and produces NOT_DETECTED in audit."""
        sparse_json = {
            "product_name": {"value": "Loose Biscuits", "confidence": 90.0},
            "net_quantity": {"value": "200 g", "confidence": 92.0}
            # All other mandatory fields omitted
        }
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(sparse_json)

        with patch("gemini_vision.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            res = extract_with_gemini(self.sample_image, api_key="valid-mock-key")

            self.assertTrue(res["success"])
            # Omitted fields must default safely to empty string without raising KeyError
            self.assertEqual(res["fields"]["MRP"], "")
            self.assertEqual(res["fields"]["Manufacturing Date"], "")
            self.assertEqual(res["fields"]["Batch Number"], "")
            self.assertIsNone(res["fields_detail"]["MRP"]["confidence"])

            # Pass into Legal Metrology compliance engine
            compliance = evaluate_compliance(res["fields"], fields_detail=res["fields_detail"])
            self.assertEqual(compliance["status"], "REQUIRES_REVIEW")
            self.assertGreaterEqual(compliance["not_detected_count"], 1)
            # Crucial: Unreadable/missing fields must NOT be marked NON_COMPLIANT
            self.assertEqual(compliance["failed_count"], 0)
            mrp_rule = next(e for e in compliance["audit_table"] if e["rule_id"] == "PCR-04")
            self.assertEqual(mrp_rule["status"], "NOT_DETECTED")

    # =========================================================================
    # CASE 4: Low-confidence Gemini response
    # =========================================================================
    def test_case4_low_confidence_gemini_response(self):
        """4. Low confidence extraction (< 70%) is flagged as REQUIRES_REVIEW in fields_detail and compliance."""
        low_conf_json = {
            "product_name": {"value": "Ambiguous Flour", "confidence": 55.0, "requires_review": True},
            "manufacturer": {"value": "Some Flour Mill", "confidence": 50.0, "requires_review": True},
            "net_quantity": {"value": "500 g", "confidence": 48.0, "requires_review": True},
            "mrp": {"value": "MRP Rs. 40.00", "confidence": 52.0, "requires_review": True},
            "manufacturing_date": {"value": "01/2026", "confidence": 50.0, "requires_review": True},
            "consumer_care_phone": {"value": "1800123456", "confidence": 55.0, "requires_review": True},
            "country_of_origin": {"value": "India", "confidence": 60.0, "requires_review": True}
        }
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(low_conf_json)

        with patch("gemini_vision.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            res = extract_with_gemini(self.sample_image, api_key="valid-mock-key")

            self.assertTrue(res["success"])
            self.assertTrue(res["fields_detail"]["Net Quantity"]["requires_review"])
            self.assertEqual(res["fields_detail"]["Net Quantity"]["confidence"], 48.0)

            # In compliance engine, 48.0% < 70.0% threshold must force REQUIRES_REVIEW
            compliance = evaluate_compliance(res["fields"], fields_detail=res["fields_detail"])
            rule_qty = next(e for e in compliance["audit_table"] if e["rule_id"] == "PCR-03")
            self.assertEqual(rule_qty["status"], "REQUIRES_REVIEW")
            self.assertIn("Low extraction confidence", rule_qty["reason"])
            self.assertEqual(compliance["status"], "REQUIRES_REVIEW")

    # =========================================================================
    # CASE 5: Conflicting OCR and Gemini extraction
    # =========================================================================
    def test_case5_conflicting_gemini_and_local_ocr_extraction(self):
        """5. Conflicting values between Gemini and OCR in dual mode produce conflict alerts and REQUIRES_REVIEW."""
        gemini_result = {
            "success": True,
            "engine": "gemini",
            "model": "gemini-3.6-flash",
            "confidence": 95.0,
            "processing_time_sec": 1.5,
            "fields": {
                "Product Name": "Monaliz Baking Powder",
                "Manufacturer": "Monaliz Food Product, Mumbai",
                "Net Quantity": "50 g",
                "MRP": "MRP Rs. 28.00 (incl. of all taxes)",
                "Unit Sale Price": "Rs. 0.56 per g",
                "Manufacturing Date": "15/10/2025",
                "Batch Number": "AB1025",
                "Best Before / Expiry": "14/10/2027",
                "Consumer Care": "care@monaliz.com",
                "Country of Origin": "India"
            },
            "fields_detail": {
                "Net Quantity": {"value": "50 g", "confidence": 95.0, "requires_review": False, "source": "Gemini Vision AI", "has_conflict": False},
                "MRP": {"value": "MRP Rs. 28.00 (incl. of all taxes)", "confidence": 95.0, "requires_review": False, "source": "Gemini Vision AI", "has_conflict": False}
            },
            "raw_text": "Sample"
        }

        ocr_conflicting = dict(self.mock_ocr_data)
        ocr_conflicting["net_quantity"] = "500 g"  # Severe divergence: 50 g vs 500 g
        ocr_conflicting["batch_number"] = "XY9999"  # Divergent batch

        with patch("pipeline.hybrid_extractor.get_api_key", return_value="valid-mock-key"):
            with patch("pipeline.hybrid_extractor.extract_with_gemini", return_value=gemini_result):
                with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=ocr_conflicting):
                    res = run_hybrid_extraction(self.sample_image, engine_mode="dual", gemini_api_key="valid-mock-key")

                    self.assertTrue(res["success"])
                    self.assertIn("Dual-Engine", res["engine"])

                    # Verify conflict flag on Net Quantity
                    qty_detail = res["fields_detail"]["Net Quantity"]
                    self.assertTrue(qty_detail["has_conflict"])
                    self.assertTrue(qty_detail["requires_review"])
                    self.assertIn("Gemini detected '50 g' vs Local OCR detected '500 g'", qty_detail["conflict_detail"])

                    # Pass into Compliance Engine: must generate REQUIRES_REVIEW with explanation
                    compliance = evaluate_compliance(res["fields"], fields_detail=res["fields_detail"])
                    rule_qty = next(e for e in compliance["audit_table"] if e["rule_id"] == "PCR-03")
                    self.assertEqual(rule_qty["status"], "REQUIRES_REVIEW")
                    self.assertTrue(rule_qty["has_conflict"])
                    self.assertIn("Extraction discrepancy", rule_qty["reason"])
                    self.assertEqual(compliance["status"], "REQUIRES_REVIEW")

    # =========================================================================
    # CASE 6: Network / API failure -> fallback to local OCR
    # =========================================================================
    def test_case6_network_api_failure_triggers_auto_fallback(self):
        """6a. Network connection timeout or HTTP 503 error automatically routes to Local OCR in 'auto' mode."""
        with patch("pipeline.hybrid_extractor.get_api_key", return_value="valid-mock-key"):
            with patch("pipeline.hybrid_extractor.extract_with_gemini") as mock_gemini:
                mock_gemini.return_value = {
                    "success": False,
                    "error": "HTTPSConnectionPool: Connection timed out (504 Gateway Timeout)",
                    "engine": "gemini",
                    "processing_time_sec": 3.0
                }
                with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=self.mock_ocr_data):
                    res = run_hybrid_extraction(self.sample_image, engine_mode="auto", gemini_api_key="valid-mock-key")

                    self.assertTrue(res["success"])
                    self.assertTrue(res["fallback_used"])
                    self.assertIn("Fallback", res["engine"])
                    self.assertIn("504 Gateway Timeout", res["fallback_note"])
                    self.assertEqual(res["fields"]["Net Quantity"], "50 g")

    def test_case6_gemini_mode_fails_cleanly_on_network_error(self):
        """6b. When 'gemini' mode is explicitly selected, network failure returns a clean error dict, not a crash."""
        with patch("pipeline.hybrid_extractor.get_api_key", return_value="valid-mock-key"):
            with patch("pipeline.hybrid_extractor.extract_with_gemini") as mock_gemini:
                mock_gemini.return_value = {
                    "success": False,
                    "error": "Google API Quota Exceeded (ResourceExhausted)",
                    "engine": "gemini",
                    "processing_time_sec": 0.5
                }
                res = run_hybrid_extraction(self.sample_image, engine_mode="gemini", gemini_api_key="valid-mock-key")

                self.assertFalse(res["success"])
                self.assertFalse(res["fallback_used"])
                self.assertEqual(res["error"], "Google API Quota Exceeded (ResourceExhausted)")
                self.assertEqual(res["engine"], "Gemini Vision AI (Failed)")

    # =========================================================================
    # CASE 7: Missing API key in auto mode -> fallback to local OCR
    # =========================================================================
    def test_case7_missing_api_key_in_auto_mode(self):
        """7a. In 'auto' mode with empty or missing API key, system immediately routes to Local OCR."""
        with patch("pipeline.hybrid_extractor.get_api_key", return_value=""):
            with patch("pipeline.hybrid_extractor.get_mistral_api_key", return_value=""):
                with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=self.mock_ocr_data) as mock_ocr:
                    res = run_hybrid_extraction(self.sample_image, engine_mode="auto", gemini_api_key="", mistral_api_key="")

                    self.assertTrue(res["success"])
                    self.assertTrue(res["fallback_used"])
                    self.assertIn("No AI API keys configured", res["fallback_note"])
                    self.assertIn("Fallback", res["engine"])
                    self.assertEqual(res["fields"]["Net Quantity"], "50 g")
                    self.assertEqual(mock_ocr.call_count, 1)

    def test_case7_missing_api_key_in_gemini_mode(self):
        """7b. In explicit 'gemini' mode with missing API key, extract_with_gemini immediately returns error."""
        with patch("gemini_vision.get_api_key", return_value=""):
            res = extract_with_gemini(self.sample_image, api_key="")
            self.assertFalse(res["success"])
            self.assertIn("No Gemini API key provided", res["error"])

    # =========================================================================
    # CASE 8: Field Provenance and Engine Attributes
    # =========================================================================
    def test_case8_field_provenance_and_engine_attribute(self):
        """8. Every field preserves value, confidence, evidence, and engine attributes."""
        with patch("pipeline.hybrid_extractor.get_api_key", return_value=""):
            with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=self.mock_ocr_data):
                res = run_hybrid_extraction(self.sample_image, engine_mode="auto")
                self.assertTrue(res["success"])
                self.assertIn("fields_detail", res)
                for field_name, detail in res["fields_detail"].items():
                    self.assertIn("value", detail)
                    self.assertIn("evidence", detail)
                    self.assertIn("engine", detail)
                    self.assertEqual(detail["engine"], "Local OCR")

    # =========================================================================
    # CASE 9: Auto Mode Engine Labels ('Gemini Vision — Active' & 'Failed -> Fallback')
    # =========================================================================
    def test_case9_auto_mode_engine_active_and_fallback_labels(self):
        """9. Auto mode produces 'Gemini Vision — Active' on success, 'Failed -> Fallback' on failure."""
        gemini_success = {
            "success": True,
            "engine": "gemini",
            "model": "gemini-3.6-flash",
            "confidence": 96.0,
            "processing_time_sec": 1.1,
            "fields": {"Product Name": "Sample Prod"},
            "fields_detail": {"Product Name": {"value": "Sample Prod", "confidence": 96.0, "evidence": "Sample", "engine": "Gemini Vision"}},
            "raw_text": "Sample"
        }
        with patch("pipeline.hybrid_extractor.get_api_key", return_value="valid-key"):
            with patch("pipeline.hybrid_extractor.extract_with_gemini", return_value=gemini_success):
                res = run_hybrid_extraction(self.sample_image, engine_mode="auto", gemini_api_key="valid-key")
                self.assertIn("Gemini Vision — Active", res["engine"])

        gemini_fail = {
            "success": False,
            "error": "Timeout 504",
            "engine": "gemini",
            "processing_time_sec": 2.0
        }
        with patch("pipeline.hybrid_extractor.get_api_key", return_value="valid-key"):
            with patch("pipeline.hybrid_extractor.extract_with_gemini", return_value=gemini_fail):
                with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=self.mock_ocr_data):
                    res_fail = run_hybrid_extraction(self.sample_image, engine_mode="auto", gemini_api_key="valid-key")
                    self.assertEqual(res_fail["engine"], "Local OCR (OpenCV + Tesseract - Fallback)")
                    self.assertTrue(res_fail["fallback_used"])
                    self.assertIn("Local OCR fallback used", res_fail["fallback_note"])

    # =========================================================================
    # CASE 10: Telemetry Flags (gemini_configured, attempted, success)
    # =========================================================================
    def test_case10_telemetry_flags(self):
        """10. run_hybrid_extraction returns exact telemetry flags for Gemini state."""
        # A: Auto mode with key and success
        gemini_success = {
            "success": True,
            "engine": "gemini",
            "model": "gemini-3.6-flash",
            "confidence": 96.0,
            "processing_time_sec": 1.1,
            "fields": {"Product Name": "Sample Prod"},
            "fields_detail": {"Product Name": {"value": "Sample Prod", "confidence": 96.0, "evidence": "Sample", "engine": "Gemini Vision"}},
            "raw_text": "Sample"
        }
        with patch("pipeline.hybrid_extractor.get_api_key", return_value="mock-api-key-test-1234"):
            with patch("pipeline.hybrid_extractor.extract_with_gemini", return_value=gemini_success):
                res = run_hybrid_extraction(self.sample_image, engine_mode="auto", gemini_api_key="mock-api-key-test-1234")
                self.assertTrue(res["gemini_configured"])
                self.assertTrue(res["gemini_attempted"])
                self.assertTrue(res["gemini_success"])
                self.assertIsNone(res["gemini_failure_reason"])

        # B: Auto mode without key
        with patch("pipeline.hybrid_extractor.get_api_key", return_value=""):
            with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=self.mock_ocr_data):
                res_no_key = run_hybrid_extraction(self.sample_image, engine_mode="auto")
                self.assertFalse(res_no_key["gemini_configured"])
                self.assertFalse(res_no_key["gemini_attempted"])
                self.assertFalse(res_no_key["gemini_success"])

        # C: Auto mode with key but Gemini API failure
        gemini_fail = {"success": False, "error": "Quota exceeded", "engine": "gemini"}
        with patch("pipeline.hybrid_extractor.get_api_key", return_value="mock-api-key-test-1234"):
            with patch("pipeline.hybrid_extractor.extract_with_gemini", return_value=gemini_fail):
                with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=self.mock_ocr_data):
                    res_api_fail = run_hybrid_extraction(self.sample_image, engine_mode="auto", gemini_api_key="mock-api-key-test-1234")
                    self.assertTrue(res_api_fail["gemini_configured"])
                    self.assertTrue(res_api_fail["gemini_attempted"])
                    self.assertFalse(res_api_fail["gemini_success"])
                    self.assertEqual(res_api_fail["gemini_failure_reason"], "Quota exceeded")
                    self.assertTrue(res_api_fail["fallback_used"])

    # =========================================================================
    # CASE 11: Gemini Vision AI Only Mode Strictly Rejects Fallback
    # =========================================================================
    def test_case11_gemini_mode_does_not_fallback(self):
        """11. In 'gemini' mode, failure does NOT fall back to Local OCR."""
        gemini_fail = {"success": False, "error": "Network Timeout", "engine": "gemini"}
        with patch("pipeline.hybrid_extractor.get_api_key", return_value="mock-api-key-test-1234"):
            with patch("pipeline.hybrid_extractor.extract_with_gemini", return_value=gemini_fail):
                with patch("pipeline.hybrid_extractor.extract_local_ocr") as mock_ocr:
                    res = run_hybrid_extraction(self.sample_image, engine_mode="gemini", gemini_api_key="mock-api-key-test-1234")
                    self.assertFalse(res["success"])
                    self.assertFalse(res["fallback_used"])
                    self.assertIn("Failed", res["engine"])
                    self.assertIn("Network Timeout", res["error"])
                    mock_ocr.assert_not_called()

    # =========================================================================
    # CASE 12: Local OCR Only Mode Never Calls Gemini
    # =========================================================================
    def test_case12_local_mode_never_calls_gemini(self):
        """12. In 'local' mode, Gemini Vision is never attempted even if key is present."""
        with patch("pipeline.hybrid_extractor.extract_with_gemini") as mock_gemini:
            with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=self.mock_ocr_data):
                res = run_hybrid_extraction(self.sample_image, engine_mode="local", gemini_api_key="mock-api-key-test-1234")
                self.assertTrue(res["success"])
                self.assertFalse(res["gemini_attempted"])
                self.assertEqual(res["engine"], "Local OCR (OpenCV + Tesseract)")
                mock_gemini.assert_not_called()

    # =========================================================================
    # CASE 13: Error Sanitization Strips Sensitive API Keys
    # =========================================================================
    def test_case13_error_sanitization_strips_api_keys(self):
        """13. sanitize_gemini_error strips AIzaSy... tokens and categorizes errors cleanly."""
        from gemini_vision import sanitize_gemini_error
        dummy_raw_key = "AIzaSy" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123456"
        leaked_key_error = f"API call failed with {dummy_raw_key}: permission denied"
        cleaned = sanitize_gemini_error(leaked_key_error)
        self.assertNotIn(dummy_raw_key, cleaned)
        self.assertIn("Authentication Error", cleaned)

    # =========================================================================
    # CASE 14: Programmatic Model Resolution & Candidate Selection
    # =========================================================================
    def test_case14_resolve_supported_model(self):
        """14. resolve_supported_model selects gemini-3.6-flash and filters unsupported/preview models."""
        from gemini_vision import resolve_supported_model, DEFAULT_GEMINI_MODEL
        
        # When client is None, returns preferred/default
        self.assertEqual(resolve_supported_model(None), "gemini-3.6-flash")

        # Mock client with models list containing gemini-3.6-flash
        mock_client = MagicMock()
        m1 = MagicMock()
        m1.name = "models/gemini-3.6-flash"
        m2 = MagicMock()
        m2.name = "models/gemini-2.0-flash"  # Disallowed
        m3 = MagicMock()
        m3.name = "models/gemini-1.5-flash"  # Disallowed
        m4 = MagicMock()
        m4.name = "models/gemini-3.6-flash-preview"  # Disallowed
        mock_client.models.list.return_value = [m1, m2, m3, m4]

        resolved = resolve_supported_model(mock_client)
        self.assertEqual(resolved, "gemini-3.6-flash")

        # When gemini-3.6-flash is absent but gemini-3.5-flash is present
        mock_client_alt = MagicMock()
        ma1 = MagicMock()
        ma1.name = "models/gemini-3.5-flash"
        mock_client_alt.models.list.return_value = [ma1]
        resolved_alt = resolve_supported_model(mock_client_alt)
        self.assertEqual(resolved_alt, "gemini-3.5-flash")

    # =========================================================================
    # CASE 15: Startup Configuration Validation
    # =========================================================================
    def test_case15_validate_gemini_configuration(self):
        """15. validate_gemini_configuration verifies model before image scan."""
        from gemini_vision import validate_gemini_configuration

        # No key
        with patch("gemini_vision.get_api_key", return_value=""):
            res = validate_gemini_configuration()
            self.assertFalse(res["configured"])
            self.assertFalse(res["valid"])

        # Valid key and model
        with patch("gemini_vision.get_api_key", return_value="mock-valid-key-1234"):
            with patch("gemini_vision.resolve_supported_model", return_value="gemini-3.6-flash"):
                with patch("google.genai.Client"):
                    res = validate_gemini_configuration()
                    self.assertTrue(res["configured"])
                    self.assertTrue(res["valid"])
                    self.assertEqual(res["active_model"], "gemini-3.6-flash")


if __name__ == "__main__":
    unittest.main()
