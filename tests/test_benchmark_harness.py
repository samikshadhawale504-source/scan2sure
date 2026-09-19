"""
Automated Unit and Integration Tests for the Benchmark Evaluation Harness
(services/benchmark_harness.py).

Verifies:
1. Single image pipeline execution fidelity
2. Ground truth handling (unavailable vs evaluated)
3. Corrupt/missing image error isolation (resilience)
4. Multi-image batch execution
5. Machine-readable exports (JSON, CSV, and text summary)
6. Conflict and review state propagation
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import json
import csv
import shutil

from services.benchmark_harness import (
    process_single_image,
    run_benchmark,
    SUPPORTED_EXTENSIONS
)


class TestBenchmarkHarness(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_compliant_extraction = {
            "success": True,
            "engine": "Local OCR (OpenCV + Tesseract)",
            "confidence": 92.0,
            "processing_time_sec": 0.8,
            "fallback_used": False,
            "fallback_note": "",
            "fields": {
                "Product Name": "Tata Salt Vacuum Evaporated",
                "Manufacturer": "Tata Chemicals Ltd, Bombay House, Mumbai 400001",
                "Net Quantity": "1 kg",
                "MRP": "MRP Rs. 28.00 (incl. of all taxes)",
                "Unit Sale Price": "Rs. 28.00 per kg",
                "Manufacturing Date": "01/2026",
                "Batch Number": "TS1001",
                "Best Before / Expiry": "12/2027",
                "Consumer Care": "1800-22-4488 | care@tatachemicals.com",
                "Country of Origin": "India"
            },
            "fields_detail": {
                "Net Quantity": {"value": "1 kg", "confidence": 95.0, "source": "Local OCR", "has_conflict": False},
                "MRP": {"value": "MRP Rs. 28.00 (incl. of all taxes)", "confidence": 92.0, "source": "Local OCR", "has_conflict": False}
            },
            "raw_text": "Sample packaging raw text"
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_process_single_image_success(self):
        """1. process_single_image runs full pipeline and records comprehensive statutory metrics."""
        dummy_img = os.path.join(self.temp_dir, "sample.jpg")
        with open(dummy_img, "wb") as f:
            f.write(b"fake_image_bytes")

        with patch("services.benchmark_harness.run_hybrid_extraction", return_value=self.mock_compliant_extraction):
            record = process_single_image(dummy_img, engine_mode="local")

            self.assertTrue(record["image_processing_success"])
            self.assertEqual(record["filename"], "sample.jpg")
            self.assertEqual(record["final_compliance_status"], "COMPLIANT")
            self.assertEqual(record["compliance_score"], 100.0)
            self.assertEqual(len(record["failed_rules"]), 0)
            self.assertEqual(record["ground_truth_status"], "ground truth unavailable")
            self.assertIsNone(record["accuracy_metrics"])
            self.assertGreater(record["rule_status_counts"]["COMPLIANT"], 5)

    def test_ground_truth_annotation_evaluation(self):
        """2. When ground truth is supplied, computes field-level accuracy and status concordance."""
        dummy_img = os.path.join(self.temp_dir, "sample_gt.jpg")
        with open(dummy_img, "wb") as f:
            f.write(b"fake_image_bytes")

        gt_annotations = {
            "Product Name": "Tata Salt Vacuum Evaporated",
            "Net Quantity": "1 kg",
            "MRP": "MRP Rs. 28.00 (incl. of all taxes)",
            "expected_status": "COMPLIANT"
        }

        with patch("services.benchmark_harness.run_hybrid_extraction", return_value=self.mock_compliant_extraction):
            record = process_single_image(dummy_img, ground_truth=gt_annotations)

            self.assertEqual(record["ground_truth_status"], "evaluated")
            self.assertIsNotNone(record["accuracy_metrics"])
            self.assertEqual(record["accuracy_metrics"]["field_accuracy_pct"], 100.0)
            self.assertTrue(record["accuracy_metrics"]["status_matched"])

    def test_ground_truth_divergence_detection(self):
        """3. Field divergence against ground truth is accurately reported without crashing."""
        dummy_img = os.path.join(self.temp_dir, "sample_mismatch.jpg")
        with open(dummy_img, "wb") as f:
            f.write(b"fake_image_bytes")

        gt_divergent = {
            "Product Name": "Tata Salt Vacuum Evaporated",
            "Net Quantity": "500 g",  # Ground truth differs from extracted '1 kg'
            "expected_status": "NON-COMPLIANT"
        }

        with patch("services.benchmark_harness.run_hybrid_extraction", return_value=self.mock_compliant_extraction):
            record = process_single_image(dummy_img, ground_truth=gt_divergent)

            self.assertEqual(record["ground_truth_status"], "evaluated")
            metrics = record["accuracy_metrics"]
            self.assertFalse(metrics["field_breakdown"]["Net Quantity"]["matched"])
            self.assertFalse(metrics["status_matched"])
            self.assertLess(metrics["field_accuracy_pct"], 100.0)

    def test_corrupt_or_missing_image_error_isolation(self):
        """4. Corrupt, non-existent, or failing images record error dictionary without stopping batch."""
        missing_path = os.path.join(self.temp_dir, "nonexistent.jpg")
        record = process_single_image(missing_path)

        self.assertFalse(record["image_processing_success"])
        self.assertIn("not found", record["error_message"])
        self.assertEqual(record["final_compliance_status"], "UNKNOWN")

    def test_conflict_and_review_state_propagation(self):
        """5. Engine discrepancies propagate to conflicts dictionary and flag requires_review_state."""
        dummy_img = os.path.join(self.temp_dir, "conflict.png")
        with open(dummy_img, "wb") as f:
            f.write(b"fake_image_bytes")

        extraction_with_conflict = dict(self.mock_compliant_extraction)
        extraction_with_conflict["fields_detail"] = {
            "Net Quantity": {
                "value": "1 kg",
                "confidence": 85.0,
                "has_conflict": True,
                "conflict_detail": "Gemini detected '1 kg' vs Local OCR detected '2 kg'"
            }
        }

        with patch("services.benchmark_harness.run_hybrid_extraction", return_value=extraction_with_conflict):
            record = process_single_image(dummy_img)

            self.assertTrue(record["requires_review_state"])
            self.assertIn("Net Quantity", record["conflicts"])
            self.assertEqual(record["final_compliance_status"], "REQUIRES_REVIEW")

    def test_multi_image_batch_benchmark_and_exports(self):
        """6. run_benchmark processes multiple images, calculates aggregate scorecard, and writes JSON/CSV/TXT."""
        # Create 2 test image files
        img1 = os.path.join(self.temp_dir, "pack1.jpg")
        img2 = os.path.join(self.temp_dir, "pack2.png")
        with open(img1, "wb") as f:
            f.write(b"img1")
        with open(img2, "wb") as f:
            f.write(b"img2")

        # Mock extractions
        with patch("services.benchmark_harness.run_hybrid_extraction") as mock_extract:
            mock_extract.side_effect = [
                self.mock_compliant_extraction,
                {
                    "success": False,
                    "error": "Tesseract corrupted image error",
                    "engine": "Local OCR",
                    "confidence": 0.0,
                    "fields": {},
                    "fields_detail": {}
                }
            ]

            out_dir = os.path.join(self.temp_dir, "output")
            res = run_benchmark(self.temp_dir, output_dir=out_dir, engine_mode="local")

            self.assertEqual(res["benchmark_summary"]["total_images"], 2)
            self.assertEqual(res["benchmark_summary"]["successful_extractions"], 1)
            self.assertEqual(res["benchmark_summary"]["failed_extractions"], 1)

            # Verify files were generated
            json_file = os.path.join(out_dir, "benchmark_results.json")
            csv_file = os.path.join(out_dir, "benchmark_results.csv")
            txt_file = os.path.join(out_dir, "benchmark_summary.txt")

            self.assertTrue(os.path.exists(json_file))
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(txt_file))

            # Inspect JSON structure
            with open(json_file, "r", encoding="utf-8") as jf:
                jdata = json.load(jf)
                self.assertIn("benchmark_summary", jdata)
                self.assertEqual(len(jdata["image_results"]), 2)

            # Inspect CSV structure
            with open(csv_file, "r", encoding="utf-8") as cf:
                reader = list(csv.DictReader(cf))
                self.assertEqual(len(reader), 2)
                self.assertEqual(reader[0]["filename"], "pack1.jpg")
                self.assertEqual(reader[0]["success"], "True")


if __name__ == "__main__":
    unittest.main()
