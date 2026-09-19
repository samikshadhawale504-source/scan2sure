"""
Unit tests for tabular grid parsing, currency normalization, and noise filtering in ocr_test.py.
"""

import unittest
from ocr_test import (
    parse_packaging_grid,
    extract_product_name,
    extract_net_quantity,
    extract_mrp,
    extract_unit_sale_price,
    extract_manufacturing_date,
    extract_batch_number,
    looks_like_garbage,
    clean_ocr_lines
)


class TestGridExtraction(unittest.TestCase):

    def test_tabular_grid_parsing(self):
        """Packaging grid parser correctly associates multi-line header/value declarations."""
        raw = """
        DATE OF PACKAGING : USE BY : BATCH NO : MRP : USP
        AUG26 JUL27 E190771633 ₹ 10.00 ₹ 1.25/g
        """
        grid = parse_packaging_grid(raw)
        self.assertEqual(grid.get("manufacturing_date"), "AUG26")
        self.assertEqual(grid.get("expiry_or_best_before"), "JUL27")
        self.assertEqual(grid.get("batch_number"), "E190771633")
        self.assertIn("10", grid.get("mrp", ""))
        self.assertIn("1.25", grid.get("unit_sale_price", ""))

    def test_currency_symbol_variations(self):
        """Handles Rupee symbol variations commonly read by OCR (₹, Rs., INR, F, €)."""
        texts = [
            ("MRP ₹ 150.00", "150"),
            ("MRP Rs. 150.00", "150"),
            ("MRP F 150.00", "150"),
            ("MRP € 150.00", "150"),
            ("MAX RETAIL PRICE INR 150.00", "150")
        ]
        for t, expected in texts:
            val = extract_mrp(t)
            self.assertIn(expected, val, f"Failed for text: {t}")

    def test_usp_variations(self):
        """Handles USP patterns with various unit declarations and currency symbols."""
        texts = [
            ("USP ₹ 1.25 / g", "1.25"),
            ("USP F 1.25/g", "1.25"),
            ("₹ 15.00 / kg", "15"),
            ("Unit Sale Price: Rs. 2.50 / ml", "2.50")
        ]
        for t, expected in texts:
            val = extract_unit_sale_price(t)
            self.assertIn(expected, val, f"Failed for USP: {t}")

    def test_garbage_lines_rejected(self):
        """Obvious noise, punctuation soup, and isolated fragments are rejected as garbage."""
        garbage_lines = [
            ": a a | one <",
            "~ oe",
            "* ] kr’, ory rt a .",
            "rN",
            "G",
            "Ft",
            "~2",
            "---===---",
            "s ro i 4 r"
        ]
        for g in garbage_lines:
            self.assertTrue(looks_like_garbage(g), f"Expected '{g}' to be classified as garbage")

    def test_statutory_declarations_not_rejected_as_garbage(self):
        """Legitimate packaging declarations with units or punctuation must not be rejected."""
        valid_lines = [
            "A WETWEIGHT: 8g 7 ]",
            "Net Weight: 50g",
            "MRP Rs. 45.00",
            "BATCH NO: B123",
            "DATE OF PACKAGING : AUG26",
            "USE BY : 12M",
            "USP: ₹ 1.25 / g"
        ]
        for v in valid_lines:
            self.assertFalse(looks_like_garbage(v), f"Expected '{v}' to NOT be classified as garbage")

    def test_product_name_rejects_numeric_value_rows(self):
        """Candidate product name extraction rejects lines containing tabular values, prices, or dates."""
        grid_row = "AUG26 IWL2? E190774633 F 10.00; # 1.2579"
        candidate = extract_product_name(grid_row)
        self.assertEqual(candidate, "", f"Expected empty product name for grid values, got: '{candidate}'")

    def test_product_name_detects_clean_brand(self):
        """Clean commodity and brand name is correctly extracted when present."""
        text = """
        Use 14 g of Monaliz Baking Powder for every 450 g of flour.
        Net Weight: 50g
        MRP: Rs. 45.00
        """
        name = extract_product_name(text)
        self.assertEqual(name, "Monaliz Baking Powder")


if __name__ == "__main__":
    unittest.main()
