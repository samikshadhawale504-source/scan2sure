"""
Automated Security, Sanitization, and Credential Safety Tests for SIH26034.

Covers:
1. File extension validation & whitelist enforcement
2. Path traversal attack sanitization on uploaded filenames
3. File upload payload size configuration
4. Secret scanning across all workspace source files for API key leaks
5. Verification of clean template in .env.example
6. Verification that user-provided API keys are never stored in the database or exposed via JSON APIs
"""

import unittest
from unittest.mock import patch
import os
import re
import io
import json
from app import app, allowed_file
from models.database import get_inspection, list_inspections


class TestApplicationSecurity(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        self.workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # =========================================================================
    # 1. File Extension Whitelisting & Upload Rejection
    # =========================================================================
    def test_allowed_file_extension_whitelist(self):
        """allowed_file must permit image formats and reject executable/script formats."""
        # Permitted formats
        self.assertTrue(allowed_file("sample.jpg"))
        self.assertTrue(allowed_file("sample.jpeg"))
        self.assertTrue(allowed_file("sample.png"))
        self.assertTrue(allowed_file("sample.webp"))
        self.assertTrue(allowed_file("sample.bmp"))
        self.assertTrue(allowed_file("SAMPLE.JPG"))
        self.assertTrue(allowed_file("sample.PnG"))

        # Dangerous or non-image formats must be rejected
        dangerous = [
            "exploit.exe", "script.sh", "backdoor.php", "payload.py",
            "notes.txt", "contract.pdf", "image.svg", "batch.bat",
            "shell.ps1", "config.json", "data.csv", "no_extension"
        ]
        for name in dangerous:
            self.assertFalse(allowed_file(name), f"Should have rejected disallowed file: {name}")

    def test_disallowed_file_upload_rejected_at_route(self):
        """Uploading an unauthorized file extension is rejected with an informative error and not processed."""
        data = {
            "product_image": (io.BytesIO(b"MALICIOUS CODE ECHO HELLO"), "exploit.exe"),
            "action": "scan"
        }
        res = self.client.post("/", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Invalid file format", res.data)
        self.assertIn(b"Accepted formats: JPG, JPEG, PNG, WEBP, BMP", res.data)

    # =========================================================================
    # 2. Path Traversal Protection
    # =========================================================================
    def test_path_traversal_filename_sanitized(self):
        """Uploading a file with directory traversal components is sanitized to a safe base filename."""
        traversal_name = "../../etc/malicious.png"
        data = {
            "product_image": (io.BytesIO(b"fake_image_bytes_12345"), traversal_name),
            "action": "scan"
        }
        res = self.client.post("/", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        # Verify no file was created outside UPLOAD_FOLDER
        upload_folder = app.config["UPLOAD_FOLDER"]
        escaped_path = os.path.abspath(os.path.join(upload_folder, traversal_name))
        self.assertFalse(os.path.exists(escaped_path))

    # =========================================================================
    # 3. Request Payload Size Protection
    # =========================================================================
    def test_max_content_length_configured(self):
        """App must have MAX_CONTENT_LENGTH configured to prevent DoS via massive payload upload."""
        self.assertIn("MAX_CONTENT_LENGTH", app.config)
        self.assertEqual(app.config["MAX_CONTENT_LENGTH"], 16 * 1024 * 1024)

    # =========================================================================
    # 4. Secret Scanning Across Workspace
    # =========================================================================
    def test_no_hardcoded_google_api_keys_in_repo(self):
        """No real Google/Gemini API keys (AIzaSy...) should be committed in source code or static assets."""
        gemini_pattern = re.compile(r"AIzaSy[0-9A-Za-z_-]{33}")
        violations = []

        scan_extensions = {".py", ".html", ".js", ".css", ".json", ".md", ".txt", ".env"}
        exclude_dirs = {".git", "__pycache__", "venv", ".venv", "node_modules"}

        for root, dirs, files in os.walk(self.workspace_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file == ".env":
                    continue  # .env is gitignored and stores local credentials
                ext = os.path.splitext(file)[1].lower()
                if ext in scan_extensions or file.startswith(".env"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            matches = gemini_pattern.findall(content)
                            if matches:
                                violations.append((file, matches))
                    except Exception:
                        pass

        self.assertEqual(len(violations), 0, f"Found hardcoded API keys in files: {violations}")

    def test_env_example_contains_only_placeholders(self):
        """.env.example should contain only empty placeholders without actual secret values."""
        example_path = os.path.join(self.workspace_dir, ".env.example")
        if os.path.exists(example_path):
            with open(example_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY=") or line.startswith("MISTRAL_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    self.assertEqual(val, "", f".env.example contains non-empty key value: {val}")

    # =========================================================================
    # 5. Database & API Secret Protection
    # =========================================================================
    def test_api_keys_not_persisted_in_database(self):
        """User-supplied API keys must NEVER be persisted in the SQLite inspections database."""
        post_res = self.client.post("/", data={
            "action": "check",
            "gemini_api_key": "user_temp_secret_key_9876543210",
            "mistral_api_key": "user_temp_mistral_key_1234567890",
            "product_name": "Security Audit Biscuit",
            "manufacturer": "Security Foods Ltd, Delhi 110001",
            "net_quantity": "100 g",
            "mrp": "MRP Rs. 20.00 (incl. of all taxes)",
            "unit_sale_price": "Rs. 0.20 per g",
            "manufacturing_date": "11/2025",
            "consumer_care": "care@securityfoods.com"
        })
        self.assertEqual(post_res.status_code, 200)

        # Retrieve the saved record
        recent = list_inspections(limit=1)
        self.assertTrue(len(recent) > 0)
        rec_id = recent[0]["id"]
        full_rec = get_inspection(rec_id)

        # Assert no trace of the secret keys in database columns or json blobs
        rec_str = json.dumps(full_rec)
        self.assertNotIn("user_temp_secret_key_9876543210", rec_str)
        self.assertNotIn("user_temp_mistral_key_1234567890", rec_str)

        # Assert JSON export does not contain secret
        json_res = self.client.get(f"/api/inspection/{rec_id}/json")
        self.assertEqual(json_res.status_code, 200)
        self.assertNotIn(b"user_temp_secret_key_9876543210", json_res.data)
        self.assertNotIn(b"user_temp_mistral_key_1234567890", json_res.data)

    # =========================================================================
    # 6. Frontend Key Elimination & Template Sanitization
    # =========================================================================
    def test_frontend_contains_no_api_key_inputs(self):
        """Frontend UI must contain ZERO API key password fields or configuration inputs."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        # Must not contain password input elements
        self.assertNotIn('type="password"', html)
        # Must not contain API key form field names
        self.assertNotIn('name="gemini_api_key"', html)
        self.assertNotIn('name="mistral_api_key"', html)
        # Must not contain custom API key drawers or headers
        self.assertNotIn("Click to configure custom API key", html)
        self.assertNotIn("API Key Configuration", html)

    def test_api_keys_never_appear_in_rendered_html(self):
        """Even when backend environment keys are configured, they are never exposed in rendered HTML."""
        secret_mistral = "mistral_env_secret_key_999988887777"
        secret_gemini = "gemini_env_secret_key_111122223333"

        with patch.dict(os.environ, {"MISTRAL_API_KEY": secret_mistral, "GEMINI_API_KEY": secret_gemini}):
            res = self.client.get("/")
            self.assertEqual(res.status_code, 200)
            html = res.data.decode("utf-8")

            self.assertNotIn(secret_mistral, html)
            self.assertNotIn(secret_gemini, html)

    def test_api_keys_never_appear_in_telemetry_dict(self):
        """Hybrid extraction metadata dict contains only booleans and sanitized engine names, zero keys."""
        from pipeline.hybrid_extractor import run_hybrid_extraction
        sample_image = os.path.join(self.workspace_dir, "product.jpg")
        secret_mistral = "mistral_secret_for_telemetry_check"
        secret_gemini = "gemini_secret_for_telemetry_check"

        mock_local = {
            "product_name": "Test Prod",
            "confidence": 80.0,
            "raw_text": "Sample"
        }

        with patch("pipeline.hybrid_extractor.extract_local_ocr", return_value=mock_local):
            res = run_hybrid_extraction(
                sample_image,
                engine_mode="auto",
                mistral_api_key=secret_mistral,
                gemini_api_key=secret_gemini
            )

            # Stringify the result to ensure neither key exists in any key or value
            res_json = json.dumps(res, default=str)
            self.assertNotIn(secret_mistral, res_json)
            self.assertNotIn(secret_gemini, res_json)


if __name__ == "__main__":
    unittest.main()

