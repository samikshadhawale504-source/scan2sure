"""
End-to-end integration tests for Flask web routes (app.py).
Tests home page, compliance check, e-commerce mode, history, report export, image upload, and error handling.
"""

import unittest
from unittest.mock import patch
import json
import io
import os
from app import app
from models.database import list_inspections


class TestFlaskRoutes(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page_get(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Legal Metrology Compliance System", response.data)
        self.assertIn(b"Upload Packaged Commodity Label", response.data)

    def test_compliance_check_post(self):
        response = self.client.post("/", data={
            "action": "check",
            "product_name": "Monaliz Baking Powder",
            "manufacturer": "Monaliz Food Product, 12 Industrial Area, Mumbai 400001",
            "net_quantity": "50 g",
            "mrp": "MRP Rs. 28.00 (incl. of all taxes)",
            "unit_sale_price": "Rs. 0.56 per g",
            "manufacturing_date": "15/10/2025",
            "batch_number": "AB1025",
            "expiry_or_best_before": "14/10/2027",
            "consumer_care": "Phone: 9820012345 | Email: care@monaliz.com",
            "country_of_origin": "India"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"COMPLIANT", response.data)
        self.assertIn(b"Statutory Declaration Audit Table", response.data)

    def test_ecommerce_routes(self):
        # GET
        get_res = self.client.get("/ecommerce")
        self.assertEqual(get_res.status_code, 200)
        self.assertIn(b"E-Commerce Product Information Audit", get_res.data)

        # POST
        sample_text = (
            "Product: Tata Tea Gold 250g\n"
            "Manufacturer: Tata Consumer Products Ltd, Mumbai 400001\n"
            "Net Quantity: 250 g\n"
            "MRP: Rs. 150 (inclusive of all taxes)\n"
            "Date of Mfg: 01/2026\n"
            "Consumer Care: care@tataconsumer.com | 1800-22-4488"
        )
        post_res = self.client.post("/ecommerce", data={"listing_text": sample_text})
        self.assertEqual(post_res.status_code, 200)
        self.assertIn(b"Tata Tea Gold", post_res.data)

    def test_history_route(self):
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Inspection Audit Log", response.data)

    def test_history_filter_by_status(self):
        """History page handles filtering by status parameter."""
        res_compliant = self.client.get("/history?status=COMPLIANT")
        self.assertEqual(res_compliant.status_code, 200)
        self.assertIn(b"Inspection Audit Log", res_compliant.data)

        res_non_compliant = self.client.get("/history?status=NON-COMPLIANT")
        self.assertEqual(res_non_compliant.status_code, 200)

    def test_json_api_export_and_report_view(self):
        post_res = self.client.post("/", data={
            "action": "check",
            "product_name": "Test Commodity",
            "manufacturer": "Test Manufacturer, Delhi 110001",
            "net_quantity": "100 g",
            "mrp": "Rs. 50.00 (incl. of all taxes)",
            "manufacturing_date": "11/2025",
            "consumer_care": "Phone: 9999999999 | Email: test@test.com"
        })
        self.assertEqual(post_res.status_code, 200)

        # Retrieve the latest inspection from history
        inspections = list_inspections(limit=1)
        self.assertTrue(len(inspections) > 0)
        latest_id = inspections[0]["id"]

        # Test JSON endpoint
        json_res = self.client.get(f"/api/inspection/{latest_id}/json")
        self.assertEqual(json_res.status_code, 200)
        data = json.loads(json_res.data)
        self.assertEqual(data["id"], latest_id)
        self.assertEqual(data["product_name"], "Test Commodity")

        # Test Official Certificate Report view
        report_res = self.client.get(f"/inspection/{latest_id}/report")
        self.assertEqual(report_res.status_code, 200)
        self.assertIn(b"STATUTORY COMPLIANCE INSPECTION REPORT", report_res.data)

    def test_nonexistent_inspection_handling(self):
        """Requesting non-existent inspection ID returns 404 for API, and redirects to history for HTML report."""
        json_res = self.client.get("/api/inspection/DOESNOTEXIST999/json")
        self.assertEqual(json_res.status_code, 404)

        report_res = self.client.get("/inspection/DOESNOTEXIST999/report")
        self.assertEqual(report_res.status_code, 302)
        self.assertIn("/history", report_res.headers.get("Location", ""))

    def test_upload_image_scan_workflow(self):
        """Uploading an image triggers hybrid extraction and loads review form with pre-populated fields."""
        mock_extract_result = {
            "success": True,
            "engine": "Local OCR (OpenCV + Tesseract)",
            "confidence": 78.5,
            "processing_time_sec": 1.1,
            "fallback_used": False,
            "fallback_note": "",
            "fields": {
                "Product Name": "Mocked Biscuit Pack",
                "Manufacturer": "Mocked Bakeries Ltd, Mumbai 400001",
                "Net Quantity": "100 g",
                "MRP": "MRP Rs. 20.00 (incl. of all taxes)",
                "Unit Sale Price": "Rs. 0.20 per g",
                "Manufacturing Date": "01/2026",
                "Batch Number": "B123",
                "Best Before / Expiry": "07/2026",
                "Consumer Care": "1800-000-0000",
                "Country of Origin": "India"
            },
            "fields_detail": {},
            "raw_text": "Sample"
        }

        with patch("app.run_hybrid_extraction", return_value=mock_extract_result):
            data = {
                "product_image": (io.BytesIO(b"fake image data"), "sample_test_label.jpg"),
                "action": "scan",
                "engine": "local"
            }
            res = self.client.post("/", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 200)
            self.assertIn(b"Mocked Biscuit Pack", res.data)
            self.assertIn(b"Mocked Bakeries Ltd", res.data)
            self.assertIn(b"Successfully extracted", res.data)

    def test_compliance_check_with_non_food_category(self):
        """Inspector specifying non-food category triggers Rule 6(1)(g) exemption."""
        res = self.client.post("/", data={
            "action": "check",
            "commodity_category": "non_food",
            "product_name": "Philips LED 9W",
            "manufacturer": "Signify India, Gurgaon 122002",
            "net_quantity": "1 number",
            "mrp": "MRP Rs. 149.00 (incl. of all taxes)",
            "unit_sale_price": "Rs. 149.00 per number",
            "manufacturing_date": "01/2026",
            "batch_number": "PH123",
            "consumer_care": "care@signify.com",
            "country_of_origin": "India"
            # No expiry provided - exempt under Rule 6(1)(g)
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"COMPLIANT", res.data)
        self.assertIn(b"NOT_APPLICABLE", res.data)


if __name__ == "__main__":
    unittest.main()
