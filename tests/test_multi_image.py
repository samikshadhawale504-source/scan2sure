"""
Unit and integration tests for Multi-Image Product Inspection in SIH26034.
Tests multi-image upload, deduplication, conflict detection, provenance,
cross-view USP calculation, failure resilience, and database persistence.
All tests run 100% offline with mocked AI responses.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import json
import io

from app import app, allowed_file
from pipeline.hybrid_extractor import (
    run_hybrid_extraction,
    run_multi_image_extraction,
    merge_multi_image_records,
    normalize_field_for_comparison
)
from compliance_rules import evaluate_compliance
from models.database import init_db, save_inspection, get_inspection


class TestMultiImageInspection(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        init_db()

    # -------------------------------------------------------------------------
    # 1. test_multi_image_upload_limit
    # -------------------------------------------------------------------------
    @patch("app.run_hybrid_extraction")
    def test_multi_image_upload_limit(self, mock_extract):
        """Verify that at most 10 images are processed, exceeding files discarded safely."""
        mock_extract.return_value = {
            "engine": "Mistral OCR",
            "confidence": 95.0,
            "processing_time_sec": 1.5,
            "fields": {"Product Name": "Test Product"},
            "fields_detail": {},
            "images_uploaded": 10,
            "images_processed": 10,
            "images_failed": 0,
            "fallback_used": False
        }

        # Create 12 dummy files
        data = {
            "action": "extract",
            "engine": "auto",
            "product_images": [
                (io.BytesIO(b"dummy image content"), f"view_{i}.jpg")
                for i in range(12)
            ]
        }
        res = self.client.post("/", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)

        # Assert run_hybrid_extraction was called with at most 10 paths
        self.assertTrue(mock_extract.called)
        call_args = mock_extract.call_args[1]
        passed_paths = call_args.get("image_path")
        self.assertIsInstance(passed_paths, list)
        self.assertLessEqual(len(passed_paths), 10)

    # -------------------------------------------------------------------------
    # 2. test_multi_image_file_types
    # -------------------------------------------------------------------------
    def test_multi_image_file_types(self):
        """Accept JPG, JPEG, PNG, WEBP, BMP (case-insensitive) and reject others."""
        allowed = ["front.jpg", "back.JPEG", "side1.png", "side2.WEBP", "pdp.bmp", "label.PNG"]
        disallowed = ["doc.pdf", "script.exe", "notes.txt", "data.csv", "image.gif"]

        for fname in allowed:
            self.assertTrue(allowed_file(fname), f"Expected {fname} to be allowed")

        for fname in disallowed:
            self.assertFalse(allowed_file(fname), f"Expected {fname} to be rejected")

    # -------------------------------------------------------------------------
    # 3. test_single_image_backwards_compatibility
    # -------------------------------------------------------------------------
    @patch("pipeline.hybrid_extractor.run_multi_image_extraction")
    def test_single_image_backwards_compatibility(self, mock_multi):
        """Passing a single image string to run_hybrid_extraction delegates correctly."""
        mock_multi.return_value = {
            "engine": "Mistral OCR",
            "confidence": 90.0,
            "processing_time_sec": 0.8,
            "fields": {"Product Name": "Single Product"},
            "fields_detail": {},
            "images_uploaded": 1,
            "images_processed": 1,
            "images_failed": 0
        }

        res = run_hybrid_extraction(image_path="uploads/test_single.jpg")
        self.assertTrue(mock_multi.called)
        called_paths = mock_multi.call_args[1].get("image_paths") or (mock_multi.call_args[0][0] if mock_multi.call_args[0] else None)
        self.assertEqual(called_paths, ["uploads/test_single.jpg"])
        self.assertEqual(res["fields"]["Product Name"], "Single Product")

    # -------------------------------------------------------------------------
    # 4. test_multi_image_all_succeed_merge
    # -------------------------------------------------------------------------
    def test_multi_image_all_succeed_merge(self):
        """2-3 images each contributing different fields merge into one complete product record."""
        view1 = {
            "image_label": "Image 1",
            "image_index": 1,
            "image_filename": "front.jpg",
            "engine": "Mistral OCR",
            "confidence": 95.0,
            "fields": {
                "Product Name": "Monaliz Baking Powder",
                "Net Quantity": "50 g",
                "MRP": "Rs. 28.00 (incl. of all taxes)"
            },
            "fields_detail": {
                "Product Name": {"value": "Monaliz Baking Powder", "evidence": "Monaliz Baking Powder", "confidence": 96.0},
                "Net Quantity": {"value": "50 g", "evidence": "Net Qty: 50 g", "confidence": 95.0},
                "MRP": {"value": "Rs. 28.00 (incl. of all taxes)", "evidence": "MRP Rs. 28.00", "confidence": 94.0}
            }
        }
        view2 = {
            "image_label": "Image 2",
            "image_index": 2,
            "image_filename": "back.jpg",
            "engine": "Mistral OCR",
            "confidence": 92.0,
            "fields": {
                "Manufacturer": "Monaliz Food Products, Plot 14, MIDC, Mumbai 400093",
                "Manufacturing Date": "15/10/2025",
                "Batch Number": "AB1025",
                "Best Before / Expiry": "14/10/2027"
            },
            "fields_detail": {
                "Manufacturer": {"value": "Monaliz Food Products, Plot 14, MIDC, Mumbai 400093", "evidence": "Mfd by: Monaliz", "confidence": 92.0},
                "Manufacturing Date": {"value": "15/10/2025", "evidence": "MFD: 15/10/2025", "confidence": 93.0},
                "Batch Number": {"value": "AB1025", "evidence": "B.No. AB1025", "confidence": 90.0},
                "Best Before / Expiry": {"value": "14/10/2027", "evidence": "EXP: 14/10/2027", "confidence": 91.0}
            }
        }
        view3 = {
            "image_label": "Image 3",
            "image_index": 3,
            "image_filename": "side.jpg",
            "engine": "Mistral OCR",
            "confidence": 90.0,
            "fields": {
                "Consumer Care": "care@monaliz.com, Phone: 022-28492000",
                "Country of Origin": "India"
            },
            "fields_detail": {
                "Consumer Care": {"value": "care@monaliz.com, Phone: 022-28492000", "evidence": "Customer Care", "confidence": 90.0},
                "Country of Origin": {"value": "India", "evidence": "Made in India", "confidence": 95.0}
            }
        }

        res = merge_multi_image_records(
            successful_results=[view1, view2, view3],
            total_images_uploaded=3,
            images_processed=3
        )
        merged_fields, merged_detail = res["fields"], res["fields_detail"]

        self.assertEqual(merged_fields["Product Name"], "Monaliz Baking Powder")
        self.assertEqual(merged_detail["Product Name"]["source_images"], ["Image 1"])
        self.assertEqual(merged_fields["Manufacturer"], "Monaliz Food Products, Plot 14, MIDC, Mumbai 400093")
        self.assertEqual(merged_detail["Manufacturer"]["source_images"], ["Image 2"])
        self.assertEqual(merged_fields["Consumer Care"], "care@monaliz.com, Phone: 022-28492000")
        self.assertEqual(merged_detail["Consumer Care"]["source_images"], ["Image 3"])
        # All merged fields have no conflict
        self.assertFalse(merged_detail["Product Name"]["has_conflict"])
        self.assertFalse(merged_detail["Manufacturer"]["has_conflict"])

    # -------------------------------------------------------------------------
    # 5. test_multi_image_deduplication_exact_match
    # -------------------------------------------------------------------------
    def test_multi_image_deduplication_exact_match(self):
        """Identical values across Image 1 and Image 2 deduplicate with source_images tracked."""
        view1 = {
            "image_label": "Image 1",
            "engine": "Mistral OCR",
            "fields": {"Product Name": "Tata Salt"},
            "fields_detail": {"Product Name": {"value": "Tata Salt", "confidence": 95.0}}
        }
        view2 = {
            "image_label": "Image 2",
            "engine": "Mistral OCR",
            "fields": {"Product Name": "Tata Salt"},
            "fields_detail": {"Product Name": {"value": "Tata Salt", "confidence": 97.0}}
        }

        res = merge_multi_image_records(
            successful_results=[view1, view2],
            total_images_uploaded=2,
            images_processed=2
        )
        merged_fields, merged_detail = res["fields"], res["fields_detail"]

        self.assertEqual(merged_fields["Product Name"], "Tata Salt")
        self.assertFalse(merged_detail["Product Name"]["has_conflict"])
        self.assertEqual(merged_detail["Product Name"]["source_images"], ["Image 1", "Image 2"])
        self.assertEqual(merged_detail["Product Name"]["confidence"], 97.0)

    # -------------------------------------------------------------------------
    # 6. test_multi_image_deduplication_normalized_match
    # -------------------------------------------------------------------------
    def test_multi_image_deduplication_normalized_match(self):
        """Descriptive fields with minor subset differences do not trigger false conflicts."""
        view1 = {
            "image_label": "Image 1",
            "engine": "Mistral OCR",
            "fields": {"Manufacturer": "Monaliz Food Products Pvt Ltd, Mumbai"},
            "fields_detail": {"Manufacturer": {"value": "Monaliz Food Products Pvt Ltd, Mumbai", "confidence": 90.0}}
        }
        view2 = {
            "image_label": "Image 2",
            "engine": "Mistral OCR",
            "fields": {"Manufacturer": "Monaliz Food Products Pvt Ltd"},
            "fields_detail": {"Manufacturer": {"value": "Monaliz Food Products Pvt Ltd", "confidence": 92.0}}
        }

        res = merge_multi_image_records(
            successful_results=[view1, view2],
            total_images_uploaded=2,
            images_processed=2
        )
        merged_fields, merged_detail = res["fields"], res["fields_detail"]

        self.assertFalse(merged_detail["Manufacturer"]["has_conflict"])
        self.assertIn("Monaliz Food Products", merged_fields["Manufacturer"])

    # -------------------------------------------------------------------------
    # 7. test_multi_image_conflict_detection_mrp
    # -------------------------------------------------------------------------
    def test_multi_image_conflict_detection_mrp(self):
        """Conflicting MRP values on Image 1 vs Image 2 flag conflict and retain both candidates."""
        view1 = {
            "image_label": "Image 1",
            "engine": "Mistral OCR",
            "fields": {"MRP": "Rs. 25.00"},
            "fields_detail": {"MRP": {"value": "Rs. 25.00", "confidence": 95.0}}
        }
        view2 = {
            "image_label": "Image 2",
            "engine": "Mistral OCR",
            "fields": {"MRP": "Rs. 30.00"},
            "fields_detail": {"MRP": {"value": "Rs. 30.00", "confidence": 93.0}}
        }

        res = merge_multi_image_records(
            successful_results=[view1, view2],
            total_images_uploaded=2,
            images_processed=2
        )
        merged_fields, merged_detail = res["fields"], res["fields_detail"]

        self.assertEqual(merged_fields["MRP"], "CONFLICT / REQUIRES REVIEW")
        self.assertTrue(merged_detail["MRP"]["has_conflict"])
        self.assertIn("Rs. 25.00", merged_detail["MRP"]["conflict_detail"])
        self.assertIn("Rs. 30.00", merged_detail["MRP"]["conflict_detail"])
        self.assertEqual(len(merged_detail["MRP"]["conflicting_candidates"]), 2)

    # -------------------------------------------------------------------------
    # 8. test_multi_image_conflict_detection_net_qty
    # -------------------------------------------------------------------------
    def test_multi_image_conflict_detection_net_qty(self):
        """Conflicting Net Quantity values on Image 1 vs Image 2 flag conflict."""
        view1 = {
            "image_label": "Image 1",
            "engine": "Mistral OCR",
            "fields": {"Net Quantity": "500 g"},
            "fields_detail": {"Net Quantity": {"value": "500 g", "confidence": 94.0}}
        }
        view2 = {
            "image_label": "Image 2",
            "engine": "Mistral OCR",
            "fields": {"Net Quantity": "1 kg"},
            "fields_detail": {"Net Quantity": {"value": "1 kg", "confidence": 96.0}}
        }

        res = merge_multi_image_records(
            successful_results=[view1, view2],
            total_images_uploaded=2,
            images_processed=2
        )
        merged_fields, merged_detail = res["fields"], res["fields_detail"]

        self.assertEqual(merged_fields["Net Quantity"], "CONFLICT / REQUIRES REVIEW")
        self.assertTrue(merged_detail["Net Quantity"]["has_conflict"])

    # -------------------------------------------------------------------------
    # 9. test_multi_image_conflict_audit_status
    # -------------------------------------------------------------------------
    def test_multi_image_conflict_audit_status(self):
        """When a field has conflict, compliance evaluation flags REQUIRES_REVIEW, not silent pass."""
        fields = {
            "Product Name": "Test Product",
            "Manufacturer": "Test Manufacturer, Mumbai 400001",
            "Net Quantity": "500 g",
            "MRP": "CONFLICT / REQUIRES REVIEW",
            "Unit Sale Price": "Rs. 0.10/g",
            "Manufacturing Date": "10/2025",
            "Batch Number": "B123",
            "Best Before / Expiry": "10/2027",
            "Consumer Care": "care@test.com",
            "Country of Origin": "India"
        }
        fields_detail = {
            "MRP": {
                "value": "CONFLICT / REQUIRES REVIEW",
                "has_conflict": True,
                "conflict_detail": "Conflicting declarations: Image 1 ('Rs. 50') vs Image 2 ('Rs. 60')",
                "confidence": None,
                "source_images": ["Image 1", "Image 2"]
            }
        }

        comp_res = evaluate_compliance(fields, fields_detail)
        mrp_row = next((r for r in comp_res["audit_table"] if r["rule_id"] == "PCR-04"), None)

        self.assertIsNotNone(mrp_row)
        self.assertEqual(mrp_row["status"], "REQUIRES_REVIEW")
        self.assertTrue(mrp_row["has_conflict"])
        self.assertIn("Extraction discrepancy detected", mrp_row["reason"])

    # -------------------------------------------------------------------------
    # 10. test_multi_image_provenance_tracking
    # -------------------------------------------------------------------------
    def test_multi_image_provenance_tracking(self):
        """Each merged field retains source_images, source_image, and confidence."""
        view1 = {
            "image_label": "Image 1",
            "engine": "Mistral OCR",
            "confidence": 88.0,
            "fields": {"Product Name": "Brand X Tea"},
            "fields_detail": {"Product Name": {"value": "Brand X Tea", "confidence": 88.0}}
        }
        res = merge_multi_image_records(
            successful_results=[view1],
            total_images_uploaded=1,
            images_processed=1
        )
        merged_fields, merged_detail = res["fields"], res["fields_detail"]

        fd = merged_detail["Product Name"]
        self.assertEqual(fd["source_images"], ["Image 1"])
        self.assertEqual(fd["source_image"], "Image 1")
        self.assertEqual(fd["confidence"], 88.0)

    # -------------------------------------------------------------------------
    # 11. test_multi_image_cross_view_usp_calculation
    # -------------------------------------------------------------------------
    def test_multi_image_cross_view_usp_calculation(self):
        """MRP on Image 1 + Net Qty on Image 2 automatically computes statutory USP."""
        view1 = {
            "image_label": "Image 1",
            "engine": "Mistral OCR",
            "fields": {"MRP": "Rs. 50.00 (incl. of all taxes)"},
            "fields_detail": {"MRP": {"value": "Rs. 50.00 (incl. of all taxes)", "confidence": 95.0}}
        }
        view2 = {
            "image_label": "Image 2",
            "engine": "Mistral OCR",
            "fields": {"Net Quantity": "500 g"},
            "fields_detail": {"Net Quantity": {"value": "500 g", "confidence": 95.0}}
        }

        res = merge_multi_image_records(
            successful_results=[view1, view2],
            total_images_uploaded=2,
            images_processed=2
        )
        merged_fields, merged_detail = res["fields"], res["fields_detail"]

        usp_detail = merged_detail["Unit Sale Price"]
        self.assertIsNotNone(usp_detail.get("calculated_usp"))
        # 50 / 500 = ₹0.10/g
        self.assertIn("0.10", usp_detail["calculated_usp"])
        self.assertIn("/g", usp_detail["calculated_usp"])
        self.assertEqual(merged_fields["Unit Sale Price"], usp_detail["calculated_usp"])

    # -------------------------------------------------------------------------
    # 12. test_multi_image_cross_view_usp_conflict_prevents_calc
    # -------------------------------------------------------------------------
    def test_multi_image_cross_view_usp_conflict_prevents_calc(self):
        """If MRP has a conflict, USP cannot be calculated and is marked for review."""
        view1 = {
            "image_label": "Image 1",
            "engine": "Mistral OCR",
            "fields": {"MRP": "Rs. 50.00", "Net Quantity": "500 g"},
            "fields_detail": {
                "MRP": {"value": "Rs. 50.00", "confidence": 95.0},
                "Net Quantity": {"value": "500 g", "confidence": 95.0}
            }
        }
        view2 = {
            "image_label": "Image 2",
            "engine": "Mistral OCR",
            "fields": {"MRP": "Rs. 60.00"},
            "fields_detail": {"MRP": {"value": "Rs. 60.00", "confidence": 93.0}}
        }

        res = merge_multi_image_records(
            successful_results=[view1, view2],
            total_images_uploaded=2,
            images_processed=2
        )
        merged_fields, merged_detail = res["fields"], res["fields_detail"]

        usp_detail = merged_detail["Unit Sale Price"]
        self.assertTrue(usp_detail.get("requires_review"))
        self.assertIsNone(usp_detail.get("calculated_usp"))
        self.assertEqual(merged_fields["Unit Sale Price"], "REQUIRES_REVIEW")

    # -------------------------------------------------------------------------
    # 13. test_multi_image_printed_vs_calculated_usp_cross_view
    # -------------------------------------------------------------------------
    def test_multi_image_printed_vs_calculated_usp_cross_view(self):
        """Calculated USP from Image 1 & 2 compared with printed USP on Image 2 flags mismatch."""
        view1 = {
            "image_label": "Image 1",
            "engine": "Mistral OCR",
            "fields": {"MRP": "Rs. 100.00 (incl. of all taxes)", "Net Quantity": "2 kg"},
            "fields_detail": {
                "MRP": {"value": "Rs. 100.00 (incl. of all taxes)", "confidence": 95.0},
                "Net Quantity": {"value": "2 kg", "confidence": 95.0}
            }
        }
        view2 = {
            "image_label": "Image 2",
            "engine": "Mistral OCR",
            "fields": {"Unit Sale Price": "Rs. 40.00 / kg"},  # Incorrect printed rate (should be 50.00)
            "fields_detail": {
                "Unit Sale Price": {"value": "Rs. 40.00 / kg", "confidence": 92.0}
            }
        }

        res = merge_multi_image_records(
            successful_results=[view1, view2],
            total_images_uploaded=2,
            images_processed=2
        )
        merged_fields, merged_detail = res["fields"], res["fields_detail"]

        usp_detail = merged_detail["Unit Sale Price"]
        self.assertTrue(usp_detail.get("has_mismatch"))
        self.assertEqual(usp_detail.get("printed_usp"), "Rs. 40.00 / kg")
        self.assertIn("50.00", usp_detail.get("calculated_usp"))

    # -------------------------------------------------------------------------
    # 14. test_multi_image_resilience_one_image_fails
    # -------------------------------------------------------------------------
    @patch("pipeline.hybrid_extractor._run_single_hybrid_extraction")
    def test_multi_image_resilience_one_image_fails(self, mock_single):
        """If 1 of 2 images fails with exception, pipeline succeeds on remaining image with telemetry."""
        # First image fails with RuntimeError, second image succeeds
        mock_single.side_effect = [
            RuntimeError("Corrupted image file / unreadable header"),
            {
                "success": True,
                "engine": "Mistral OCR",
                "confidence": 94.0,
                "fields": {"Product Name": "Resilient Tea", "Net Quantity": "250 g"},
                "fields_detail": {
                    "Product Name": {"value": "Resilient Tea", "confidence": 94.0},
                    "Net Quantity": {"value": "250 g", "confidence": 94.0}
                }
            }
        ]

        res = run_multi_image_extraction(
            image_paths=["corrupted.jpg", "valid.jpg"],
            engine_mode="auto"
        )

        self.assertEqual(res["images_uploaded"], 2)
        self.assertEqual(res["images_processed"], 1)
        self.assertEqual(res["images_failed"], 1)
        self.assertIn("Partial multi-view extraction", res.get("partial_success_note", ""))
        self.assertEqual(res["fields"]["Product Name"], "Resilient Tea")

    # -------------------------------------------------------------------------
    # 15. test_multi_image_all_images_fail
    # -------------------------------------------------------------------------
    @patch("pipeline.hybrid_extractor._run_single_hybrid_extraction")
    def test_multi_image_all_images_fail(self, mock_single):
        """If all images fail, returns safe error response without unhandled exception."""
        mock_single.side_effect = [
            RuntimeError("Failed Image 1"),
            RuntimeError("Failed Image 2")
        ]

        res = run_multi_image_extraction(
            image_paths=["fail1.jpg", "fail2.jpg"],
            engine_mode="auto"
        )

        self.assertEqual(res["images_uploaded"], 2)
        self.assertEqual(res["images_processed"], 0)
        self.assertEqual(res["images_failed"], 2)
        self.assertIsNotNone(res.get("error"))
        self.assertEqual(res["confidence"], 0.0)

    # -------------------------------------------------------------------------
    # 16. test_multi_image_empty_not_detected_ignored_if_other_detects
    # -------------------------------------------------------------------------
    def test_multi_image_empty_not_detected_ignored_if_other_detects(self):
        """NOT_DETECTED or empty value on Image 1 is ignored when Image 2 detects the field."""
        view1 = {
            "image_label": "Image 1",
            "engine": "Mistral OCR",
            "fields": {"Net Quantity": "NOT_DETECTED"},
            "fields_detail": {"Net Quantity": {"value": "NOT_DETECTED", "status": "NOT_DETECTED"}}
        }
        view2 = {
            "image_label": "Image 2",
            "engine": "Mistral OCR",
            "fields": {"Net Quantity": "500 g"},
            "fields_detail": {"Net Quantity": {"value": "500 g", "confidence": 95.0}}
        }

        res = merge_multi_image_records(
            successful_results=[view1, view2],
            total_images_uploaded=2,
            images_processed=2
        )
        merged_fields, merged_detail = res["fields"], res["fields_detail"]

        self.assertEqual(merged_fields["Net Quantity"], "500 g")
        self.assertFalse(merged_detail["Net Quantity"]["has_conflict"])
        self.assertEqual(merged_detail["Net Quantity"]["source_images"], ["Image 2"])

    # -------------------------------------------------------------------------
    # 17. test_multi_image_database_persistence
    # -------------------------------------------------------------------------
    def test_multi_image_database_persistence(self):
        """Inspection with multiple image filenames persists and retrieves correctly via SQLite and routes."""
        image_list = ["1789_front.jpg", "1789_back.jpg", "1789_side.jpg"]
        insp_id = save_inspection(
            product_name="Multi-View Biscuit Pack",
            image_filename=json.dumps(image_list),
            engine_used="Mistral OCR",
            confidence=95.0,
            overall_status="COMPLIANT",
            compliance_score=100.0,
            input_source="image",
            fields={"Product Name": "Multi-View Biscuit Pack", "Net Quantity": "200 g"},
            validation_results={},
            compliance_results={
                "status": "COMPLIANT",
                "score": 100.0,
                "passed_count": 10,
                "failed_count": 0,
                "warning_count": 0,
                "not_detected_count": 0,
                "audit_table": [
                    {
                        "rule_id": "PCR-01",
                        "rule": "Rule 6(1)(a)",
                        "requirement": "Product Name",
                        "category": "Identity",
                        "extracted_value": "Multi-View Biscuit Pack",
                        "extracted_evidence": "Biscuit Pack",
                        "extraction_source": "Mistral OCR",
                        "source_images": ["Image 1", "Image 2"],
                        "confidence": 95.0,
                        "status": "COMPLIANT",
                        "score": 10,
                        "weight": 10,
                        "reason": "Complies",
                        "recommended_action": "None"
                    }
                ]
            }
        )

        # Retrieve via get_inspection
        data = get_inspection(insp_id)
        self.assertIsNotNone(data)
        self.assertIn("1789_front.jpg", data["image_filename"])

        # Test view_inspection route
        res = self.client.get(f"/inspection/{insp_id}")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Packaging Evidence (3 Views)", res.data)
        self.assertIn(b"1789_front.jpg", res.data)

        # Test official_report route
        rep_res = self.client.get(f"/inspection/{insp_id}/report")
        self.assertEqual(rep_res.status_code, 200)
        self.assertIn(b"Packaging Views Evidence (3 views)", rep_res.data)


if __name__ == "__main__":
    unittest.main()
