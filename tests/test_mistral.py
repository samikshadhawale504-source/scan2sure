"""
Unit and integration tests for Mistral OCR Extraction Engine (mistral_ocr.py)
and 3-Tier Provider Architecture in Hybrid Extractor (pipeline/hybrid_extractor.py).

Guarantees 100% mocked offline execution: NO live network connections or API calls.
Covers:
1. Valid structured extraction from Mistral OCR markdown response
2. Anti-hallucination verification (missing packaging declarations safely marked NOT_DETECTED)
3. HTTP 401 / Authentication failure classification and key sanitization
4. HTTP 429 / Rate limit & quota failure classification
5. Mistral OCR Only mode failure strictly avoiding silent Local OCR fallback
6. Auto mode cascading: Gemini fails -> Mistral succeeds
7. Auto mode cascading: Both Gemini and Mistral fail -> Local OCR fallback
8. Auto mode cascading: Gemini unconfigured -> Mistral succeeds
9. Error sanitization ensuring zero API key exposure
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import json

from mistral_ocr import (
    extract_with_mistral,
    is_mistral_available,
    get_mistral_api_key,
    sanitize_mistral_error
)
from pipeline.hybrid_extractor import run_hybrid_extraction


class TestMistralOCRExtraction(unittest.TestCase):

    def setUp(self):
        self.sample_image = os.path.join(os.path.dirname(os.path.dirname(__file__)), "product.jpg")
        self.sample_markdown = (
            "MONALIZ BAKING POWDER\n"
            "DOUBLE ACTION\n"
            "Manufactured & Packed by:\n"
            "Monaliz Food Products Pvt Ltd\n"
            "Plot 14, MIDC Industrial Area, Mumbai 400093, Maharashtra\n"
            "Net Weight: 50 g\n"
            "MRP Rs. 28.00 (Inclusive of all taxes)\n"
            "Unit Sale Price: Rs. 0.56 / g\n"
            "Batch No: AB1025\n"
            "Mfg Date: 15/10/2025\n"
            "Best Before 24 months from manufacture\n"
            "Customer Care: care@monaliz.com | Helpline: 1800-222-333\n"
            "Country of Origin: India\n"
        )

    # =========================================================================
    # 1. Valid Structured Extraction
    # =========================================================================
    def test_mistral_success_structured_extraction(self):
        """Valid response from Mistral OCR API parses correctly into all statutory declarations."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "mistral-ocr-latest",
            "pages": [
                {
                    "index": 0,
                    "markdown": self.sample_markdown,
                    "confidence": 0.94
                }
            ],
            "usage_info": {"pages_processed": 1}
        }

        with patch("requests.post", return_value=mock_response):
            res = extract_with_mistral(self.sample_image, api_key="dummy_mock_key_12345")

            self.assertTrue(res["success"])
            self.assertEqual(res["engine"], "mistral")
            self.assertEqual(res["model"], "mistral-ocr-latest")
            self.assertGreaterEqual(res["confidence"], 90.0)

            # Check statutory fields
            self.assertIn("monaliz", res["fields"]["Product Name"].lower())
            self.assertIn("monaliz food products", res["fields"]["Manufacturer"].lower())
            self.assertEqual(res["fields"]["Net Quantity"], "50 g")
            self.assertIn("28", res["fields"]["MRP"])
            self.assertIn("0.56", res["fields"]["Unit Sale Price"])
            self.assertIn("15/10/2025", res["fields"]["Manufacturing Date"])
            self.assertIn("AB1025", res["fields"]["Batch Number"])
            self.assertIn("care@monaliz.com", res["fields"]["Consumer Care"])
            self.assertEqual(res["fields"]["Country of Origin"], "India")

            # Check fields_detail metadata and provenance
            self.assertEqual(res["fields_detail"]["Net Quantity"]["source"], "Mistral OCR")
            self.assertEqual(res["fields_detail"]["Net Quantity"]["status"], "DETECTED")
            self.assertFalse(res["fields_detail"]["Net Quantity"]["requires_review"])
            self.assertIn("50 g", res["fields_detail"]["Net Quantity"]["evidence"])

    # =========================================================================
    # 2. Anti-Hallucination Guardrail: Missing Fields
    # =========================================================================
    def test_mistral_anti_hallucination_missing_fields(self):
        """Packaging fields absent from OCR text must NEVER be hallucinated and safely marked NOT_DETECTED."""
        sparse_markdown = (
            "MONALIZ BAKING POWDER\n"
            "Net Wt: 100 g\n"
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "mistral-ocr-latest",
            "pages": [{"index": 0, "markdown": sparse_markdown}],
            "usage_info": {"pages_processed": 1}
        }

        with patch("requests.post", return_value=mock_response):
            res = extract_with_mistral(self.sample_image, api_key="dummy_mock_key_12345")

            self.assertTrue(res["success"])
            # Detected fields
            self.assertIn("monaliz", res["fields"]["Product Name"].lower())
            self.assertEqual(res["fields"]["Net Quantity"], "100 g")

            # Missing fields must remain empty strings
            self.assertEqual(res["fields"]["MRP"], "")
            self.assertEqual(res["fields"]["Manufacturing Date"], "")
            self.assertEqual(res["fields"]["Batch Number"], "")

            # Metadata must indicate NOT_DETECTED and requires_review
            self.assertEqual(res["fields_detail"]["MRP"]["status"], "NOT_DETECTED")
            self.assertTrue(res["fields_detail"]["MRP"]["requires_review"])
            self.assertEqual(res["fields_detail"]["Manufacturing Date"]["status"], "NOT_DETECTED")
            self.assertTrue(res["fields_detail"]["Manufacturing Date"]["requires_review"])

    # =========================================================================
    # 3. HTTP 401 / Authentication Error Classification
    # =========================================================================
    def test_mistral_api_failure_401_sanitized(self):
        """HTTP 401 response from Mistral API is categorized as Authentication Error without key leakage."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = '{"message": "Unauthorized: Invalid API key mistral_secret_key_abcdef12345"}'

        with patch("requests.post", return_value=mock_response):
            res = extract_with_mistral(self.sample_image, api_key="mistral_secret_key_abcdef12345")

            self.assertFalse(res["success"])
            self.assertIn("Authentication Error (401/403)", res["error"])
            # Ensure the secret key is NOT leaked in the error message
            self.assertNotIn("mistral_secret_key_abcdef12345", res["error"])

    # =========================================================================
    # 4. HTTP 429 / Rate Limit & Quota Classification
    # =========================================================================
    def test_mistral_api_failure_429_sanitized(self):
        """HTTP 429 response from Mistral API is categorized as Resource Exhausted / Quota Exceeded."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = '{"message": "Quota exceeded. Please upgrade your plan."}'

        with patch("requests.post", return_value=mock_response):
            res = extract_with_mistral(self.sample_image, api_key="dummy_mock_key_12345")

            self.assertFalse(res["success"])
            self.assertIn("429 RESOURCE_EXHAUSTED", res["error"])

    # =========================================================================
    # 5. Mistral-Only Mode: Strictly No Silent Fallback
    # =========================================================================
    def test_mistral_only_mode_no_fallback(self):
        """In explicit 'mistral' mode, API failure must NOT silently fall back to Local OCR."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = '{"message": "Forbidden"}'

        with patch("requests.post", return_value=mock_response):
            res = run_hybrid_extraction(
                self.sample_image,
                engine_mode="mistral",
                mistral_api_key="dummy_mock_key_12345"
            )

            self.assertFalse(res["success"])
            self.assertFalse(res["fallback_used"])
            self.assertEqual(res["engine"], "Mistral OCR (Failed)")
            self.assertTrue(res["mistral_attempted"])
            self.assertFalse(res["mistral_success"])
            self.assertIsNotNone(res["error"])

    # =========================================================================
    # 6. Cascade Test 1: Mistral configured -> Mistral is primary and succeeds without calling Gemini or Local OCR
    # =========================================================================
    def test_auto_cascade_mistral_primary_succeeds_without_calling_gemini_or_local(self):
        """When Mistral is configured, it runs as PRIMARY and does NOT invoke Gemini or Local OCR upon success."""
        mock_mistral_resp = MagicMock()
        mock_mistral_resp.status_code = 200
        mock_mistral_resp.json.return_value = {
            "model": "mistral-ocr-latest",
            "pages": [{"index": 0, "markdown": self.sample_markdown, "confidence": 0.95}],
            "usage_info": {"pages_processed": 1}
        }

        with patch("pipeline.hybrid_extractor.extract_with_gemini") as mock_gemini:
            with patch("pipeline.hybrid_extractor.extract_local_ocr") as mock_local:
                with patch("requests.post", return_value=mock_mistral_resp):
                    res = run_hybrid_extraction(
                        self.sample_image,
                        engine_mode="auto",
                        gemini_api_key="valid_gemini_key_12345",
                        mistral_api_key="valid_mistral_key_12345"
                    )

                    self.assertTrue(res["success"])
                    self.assertFalse(res["fallback_used"])
                    self.assertIn("Mistral OCR — Active", res["engine"])
                    self.assertTrue(res["mistral_attempted"])
                    self.assertTrue(res["mistral_success"])
                    self.assertFalse(res["gemini_attempted"])
                    self.assertFalse(res["local_attempted"])
                    mock_gemini.assert_not_called()
                    mock_local.assert_not_called()

    # =========================================================================
    # 7. Cascade Test 2: Mistral configured -> Mistral fails -> Gemini succeeds
    # =========================================================================
    def test_auto_cascade_mistral_fails_gemini_succeeds(self):
        """In Auto mode, when Mistral OCR fails, pipeline cascades to Gemini Vision."""
        mock_mistral_resp = MagicMock()
        mock_mistral_resp.status_code = 500
        mock_mistral_resp.text = "Internal Server Error"

        gemini_success_res = {
            "success": True,
            "engine": "gemini",
            "model": "gemini-3.6-flash",
            "confidence": 95.0,
            "fields": {
                "Product Name": "Monaliz Baking Powder",
                "Manufacturer": "Monaliz Food Products Pvt Ltd",
                "Net Quantity": "50 g",
                "MRP": "Rs. 28",
                "Unit Sale Price": "Rs. 0.56 / g",
                "Manufacturing Date": "15/10/2025",
                "Batch Number": "AB1025",
                "Best Before / Expiry": "14/10/2027",
                "Consumer Care": "care@monaliz.com",
                "Country of Origin": "India"
            },
            "fields_detail": {},
            "raw_text": self.sample_markdown
        }

        with patch("requests.post", return_value=mock_mistral_resp):
            with patch("pipeline.hybrid_extractor.extract_with_gemini", return_value=gemini_success_res):
                res = run_hybrid_extraction(
                    self.sample_image,
                    engine_mode="auto",
                    gemini_api_key="valid_gemini_key_12345",
                    mistral_api_key="valid_mistral_key_12345"
                )

                self.assertTrue(res["success"])
                self.assertTrue(res["fallback_used"])
                self.assertIn("Gemini Vision — Active", res["engine"])
                self.assertTrue(res["mistral_attempted"])
                self.assertFalse(res["mistral_success"])
                self.assertTrue(res["gemini_attempted"])
                self.assertTrue(res["gemini_success"])
                self.assertIn("Mistral OCR failed", res["fallback_note"])
                self.assertIn("Gemini Vision fallback used", res["fallback_note"])

    # =========================================================================
    # 8. Cascade Test 3: Mistral fails -> Gemini unavailable -> Local OCR fallback
    # =========================================================================
    def test_auto_cascade_mistral_fails_gemini_unavailable_local_fallback(self):
        """When Mistral fails and Gemini is unconfigured, fallback to Local OCR with explicit message."""
        mock_mistral_resp = MagicMock()
        mock_mistral_resp.status_code = 503
        mock_mistral_resp.text = "Service Unavailable"

        mock_local_res = {
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
            "confidence": 75.0,
            "processing_time_sec": 1.1,
            "raw_text": "Monaliz Baking Powder 50g Rs 28"
        }

        with patch("requests.post", return_value=mock_mistral_resp):
            with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=mock_local_res):
                res = run_hybrid_extraction(
                    self.sample_image,
                    engine_mode="auto",
                    gemini_api_key="",  # Gemini unavailable
                    mistral_api_key="valid_mistral_key_12345"
                )

                self.assertTrue(res["success"])
                self.assertTrue(res["fallback_used"])
                self.assertIn("Local OCR", res["engine"])
                self.assertTrue(res["mistral_attempted"])
                self.assertFalse(res["mistral_success"])
                self.assertFalse(res["gemini_attempted"])
                self.assertTrue(res["local_attempted"])
                self.assertIn("Mistral OCR failed", res["fallback_note"])
                self.assertIn("Gemini unavailable", res["fallback_note"])
                self.assertIn("Local OCR fallback used", res["fallback_note"])

    # =========================================================================
    # 9. Cascade Test 4: Mistral unavailable -> Gemini unavailable -> Local OCR engaged seamlessly
    # =========================================================================
    def test_auto_cascade_mistral_unavailable_gemini_unavailable_local_engaged(self):
        """When neither Mistral nor Gemini keys exist, Local OCR engaged seamlessly with clear message."""
        mock_local_res = {
            "product_name": "Monaliz Baking Powder",
            "manufacturer": "Monaliz Food Product",
            "net_quantity": "50 g",
            "mrp": "Rs. 28",
            "confidence": 75.0,
            "processing_time_sec": 0.8,
            "raw_text": "Monaliz Baking Powder 50g"
        }

        with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=mock_local_res):
            res = run_hybrid_extraction(
                self.sample_image,
                engine_mode="auto",
                gemini_api_key="",
                mistral_api_key=""
            )

            self.assertTrue(res["success"])
            self.assertTrue(res["fallback_used"])
            self.assertIn("Local OCR", res["engine"])
            self.assertFalse(res["mistral_attempted"])
            self.assertFalse(res["gemini_attempted"])
            self.assertTrue(res["local_attempted"])
            self.assertEqual(
                res["fallback_note"],
                "No AI API keys configured. Automatically using Local OCR (OpenCV + Tesseract)."
            )

    # =========================================================================
    # 10. Cascade Test 5: Gemini key missing -> application still works normally with Mistral
    # =========================================================================
    def test_auto_cascade_gemini_unconfigured_mistral_succeeds(self):
        """When Gemini API key is not configured, Auto mode seamlessly uses Mistral OCR as primary AI."""
        mock_mistral_resp = MagicMock()
        mock_mistral_resp.status_code = 200
        mock_mistral_resp.json.return_value = {
            "model": "mistral-ocr-latest",
            "pages": [{"index": 0, "markdown": self.sample_markdown, "confidence": 0.96}],
            "usage_info": {"pages_processed": 1}
        }

        with patch("requests.post", return_value=mock_mistral_resp):
            res = run_hybrid_extraction(
                self.sample_image,
                engine_mode="auto",
                gemini_api_key="",  # Not configured
                mistral_api_key="valid_mistral_key_12345"
            )

            self.assertTrue(res["success"])
            self.assertFalse(res["gemini_attempted"])
            self.assertTrue(res["mistral_attempted"])
            self.assertTrue(res["mistral_success"])
            self.assertIn("Mistral OCR — Active", res["engine"])
            self.assertFalse(res["fallback_used"])

    # =========================================================================
    # 11. Error Sanitization: Zero API Key Exposure
    # =========================================================================
    def test_mistral_error_sanitization_masks_keys(self):
        """sanitize_mistral_error must strip any Bearer tokens or secret keys."""
        raw_err = "Failed request with Bearer secret_token_xyz_98765432101234567890"
        sanitized = sanitize_mistral_error(raw_err)
        self.assertNotIn("secret_token_xyz_98765432101234567890", sanitized)


if __name__ == "__main__":
    unittest.main()
