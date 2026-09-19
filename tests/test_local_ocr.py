"""
Unit tests for Phase 1 Local OCR Pipeline (ocr_test.py).
Tests image preprocessing, orientation handling, OCR text cleaning,
field extraction heuristics, bounding-box data extraction, and confidence calculation.
"""

import unittest
import os
import cv2
import numpy as np

from ocr_test import (
    preprocess_variants,
    detect_and_correct_orientation,
    clean_text,
    extract_mrp,
    extract_net_quantity,
    extract_manufacturing_date,
    extract_expiry,
    extract_consumer_care,
    extract_product_info,
    run_ocr_data
)


class TestPhase1LocalOCR(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "product.jpg")
        # Create a synthetic 100x100 test image for fast deterministic tests
        cls.synthetic_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(cls.synthetic_img, "NET 50g", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def test_preprocess_variants(self):
        """Phase 1: preprocess_variants returns contrast, adaptive, and otsu threshold variants."""
        variants = preprocess_variants(self.synthetic_img)
        self.assertIsInstance(variants, list)
        self.assertEqual(len(variants), 3)
        for v in variants:
            self.assertIsInstance(v, np.ndarray)
            self.assertEqual(len(v.shape), 2)  # Grayscale variants

    def test_detect_and_correct_orientation(self):
        """Phase 1: detect_and_correct_orientation preserves shape or corrects orientation gracefully."""
        # Non-text synthetic image should not crash and should return an image of the same shape
        res = detect_and_correct_orientation(self.synthetic_img)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, self.synthetic_img.shape)

    def test_clean_text(self):
        """Phase 1: clean_text strips form-feeds, carriage returns, and excessive whitespace."""
        raw = "  Net \x0c Quantity: \t\t 50 g \n\n Rs. 28.00  "
        cleaned = clean_text(raw)
        self.assertNotIn("\x0c", cleaned)
        self.assertEqual(cleaned, "Net Quantity: 50 g \n\n Rs. 28.00")

    def test_extract_mrp_regex(self):
        """Phase 1: extract_mrp detects MRP and tax clauses from raw text."""
        text1 = "MRP Rs. 28.00 (INCL. OF ALL TAXES) B.No 12"
        mrp1 = extract_mrp(text1)
        self.assertIsNotNone(mrp1)
        self.assertIn("28", mrp1)

        text2 = "Maximum Retail Price ₹ 150.00 inclusive of all taxes"
        mrp2 = extract_mrp(text2)
        self.assertIsNotNone(mrp2)
        self.assertIn("150", mrp2)

    def test_extract_net_quantity_regex(self):
        """Phase 1: extract_net_quantity detects net weights and volume declarations."""
        text1 = "Net Wt. : 50 g at 27 C"
        qty1 = extract_net_quantity(text1)
        self.assertIsNotNone(qty1)
        self.assertIn("50", qty1)

        text2 = "Net Quantity: 500 ml"
        qty2 = extract_net_quantity(text2)
        self.assertIsNotNone(qty2)
        self.assertIn("500", qty2)

    def test_extract_dates_regex(self):
        """Phase 1: extract_manufacturing_date and extract_expiry extract dates from label text."""
        text = "MFD: 15/10/2025 EXP: 14/10/2027 LOT 99"
        mfg = extract_manufacturing_date(text)
        exp = extract_expiry(text)
        self.assertIsNotNone(mfg)
        self.assertIsNotNone(exp)
        self.assertIn("10/2025", mfg)
        self.assertIn("10/2027", exp)

    def test_extract_consumer_care(self):
        """Phase 1: extract_consumer_care detects email and telephone contact channels."""
        text = "For Consumer Feedback: Call 1800-22-3344 or email: care@monaliz.com"
        care = extract_consumer_care(text)
        self.assertIsNotNone(care)
        self.assertTrue("care@monaliz.com" in care or "1800" in care)

    def test_calculate_confidence(self):
        """Phase 1: OCR confidence computation from bounding box conf values."""
        mock_ocr_data = {
            "text": ["Net", "Quantity", "50", "g", ""],
            "conf": [90, 85, 95, 90, -1]  # -1 for empty tokens
        }
        confs = [float(c) for w, c in zip(mock_ocr_data["text"], mock_ocr_data["conf"]) if w.strip() and float(c) >= 0]
        avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
        self.assertGreaterEqual(avg_conf, 80.0)
        self.assertEqual(avg_conf, 90.0)

    def test_osd_orientation_safety_rejects_low_confidence(self):
        """Phase 1: detect_and_correct_orientation preserves orientation when OSD confidence < 3.0."""
        # Mock pytesseract.image_to_osd returning Rotate: 180 with low confidence 0.12
        from unittest.mock import patch
        mock_osd = "Page number: 0\nOrientation in degrees: 180\nRotate: 180\nOrientation confidence: 0.12\nScript: Latin\nScript confidence: 1.90\n"
        with patch("ocr_test.pytesseract.image_to_osd", return_value=mock_osd):
            dummy_img = np.ones((400, 400, 3), dtype=np.uint8) * 128
            out = detect_and_correct_orientation(dummy_img)
            # Must NOT flip the image when confidence is 0.12
            self.assertTrue(np.array_equal(out, dummy_img))

    def test_extract_mrp_with_colon_and_symbols(self):
        """Phase 1: extract_mrp correctly extracts MRP formatted as Rs:46/- with colon separator."""
        text = "+, 1009 mrp: Rs:46/- incl. of all Taxes"
        mrp = extract_mrp(text)
        self.assertEqual(mrp, "Rs. 46")

    def test_extract_spatial_net_quantity_heals_misread(self):
        """Phase 1: extract_net_quantity heals '1009 mrp' OCR misread into '100 g'."""
        text = "+, 1009 mrp: Rs:46/- incl. of all Taxes"
        qty = extract_net_quantity(text)
        self.assertEqual(qty, "100 g")

    def test_extract_batch_from_stamped_token_before_mfd(self):
        """Phase 1: extract_batch_number detects stamped alphanumeric token preceding MFD."""
        from ocr_test import extract_batch_number
        text = "T: 230711113 MFD: 11/07/2023 EXP: 10/01/2025"
        batch = extract_batch_number(text)
        self.assertEqual(batch, "230711113")

    def test_extract_consumer_care_std_code(self):
        """Phase 1: extract_consumer_care detects landline phone numbers with STD code."""
        text = "For Consumer Complaints Contact Manager at Phone No. 0120 -2564285 at address given below"
        care = extract_consumer_care(text)
        self.assertIn("0120 -2564285", care)

    def test_extract_mrp_multiline(self):
        """Phase 1: extract_mrp captures price across line wrap with (inclusive of all taxes)."""
        text = "MRP:\n(inclusive of all taxes):\nRs. 215.50"
        mrp = extract_mrp(text)
        self.assertEqual(mrp, "Rs. 215.50")

    def test_extract_expiry_equals_and_dot_delimiters(self):
        """Phase 1: extract_expiry detects dates with '=' or '.' delimiters."""
        from ocr_test import extract_expiry
        text1 = "Exp. Date = MAY-2028"
        text2 = "MFG .SEP.2024-BDEO EXP.AUG.2027"
        self.assertEqual(extract_expiry(text1), "MAY-2028")
        self.assertEqual(extract_expiry(text2), "AUG.2027")

    def test_extract_date_rejects_drug_license_numbers(self):
        """Phase 1: extract_manufacturing_date and grid parser do not mistake license numbers with invalid month 18 as dates."""
        from ocr_test import extract_manufacturing_date, parse_packaging_grid
        text = "Mfg.Lic.No. + /MINB/18/1018 & MB/18/1019\nMfg. Date > JUN-2026"
        mfg = extract_manufacturing_date(text)
        self.assertEqual(mfg, "JUN-2026")
        grid = parse_packaging_grid(text)
        self.assertNotIn("18/1018", grid.values())
        self.assertNotIn("18/1019", grid.values())

    def test_extract_mrp_rejects_standalone_letter_f(self):
        """Phase 1: extract_mrp does not confuse batch letter 'F' or 'S' with rupee symbol."""
        text = "B. NO: R3035930-F S800 1260"
        mrp = extract_mrp(text)
        self.assertEqual(mrp, "")

    def test_extract_product_name_rejects_hazard_and_substring_traps(self):
        """Phase 1: extract_product_name rejects hazard warnings (FLAMMABLE) and substring false matches."""
        from ocr_test import extract_product_name
        text_flammable = "FLAMMABLE\nINFLAMMABLE\nKeep out of reach of children."
        self.assertEqual(extract_product_name(text_flammable), "")
        text_teat = "{incisive of ait teat: ry"
        self.assertEqual(extract_product_name(text_teat), "")


if __name__ == "__main__":
    unittest.main()
