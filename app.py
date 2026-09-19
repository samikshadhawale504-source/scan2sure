"""
Flask Application for SIH26034: Legal Metrology (Packaged Commodities) Rules, 2011 Compliance System.
Provides a hybrid vision/OCR extraction workflow, interactive field review, explainable compliance audits,
official certificate generation, e-commerce listing parsing, and SQLite inspection logs.
"""

import os
import json
import time
from flask import Flask, render_template, request, send_from_directory, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from config import load_project_env, get_provider_status

# Ensure environment is loaded BEFORE any provider clients or pipelines are imported
load_project_env()

from pipeline.hybrid_extractor import run_hybrid_extraction
from gemini_vision import is_gemini_available, get_api_key, validate_gemini_configuration
from mistral_ocr import is_mistral_available, get_mistral_api_key
from compliance_rules import evaluate_compliance
from validation.field_validator import validate_all_fields
from services.ecommerce_parser import parse_ecommerce_listing
from models.database import init_db, save_inspection, get_inspection, list_inspections

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "sih26034-legal-metrology-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# Initialize SQLite database
init_db()

# Safe Startup AI Provider Configuration Diagnostic (YES/NO only, NEVER exposes keys)
try:
    _provider_status = get_provider_status()
    print("[*] Startup AI Provider Configuration Status:")
    print(f"    - Mistral Configured: {'YES' if _provider_status['mistral_configured'] else 'NO'}")
    print(f"    - Gemini Available  : {'YES' if _provider_status['gemini_configured'] else 'NO'}")
except Exception:
    pass


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded images safely for side-by-side inspector preview."""
    safe_name = secure_filename(filename)
    return send_from_directory(app.config["UPLOAD_FOLDER"], safe_name)


@app.route("/", methods=["GET", "POST"])
def home():
    load_project_env()

    result = None
    extraction_meta = None
    image_filename = None
    image_filenames = []
    scan_message = None
    inspection_id = None
    fields_detail = {}

    selected_engine = request.form.get("engine", "auto") if request.method == "POST" else "auto"

    # Read credentials strictly from backend environment / .env (never from user frontend)
    effective_mistral_key = get_mistral_api_key()
    has_mistral_key = bool(effective_mistral_key and len(effective_mistral_key) > 5)

    effective_gemini_key = get_api_key()
    has_gemini_key = bool(effective_gemini_key and len(effective_gemini_key) > 5)

    fields = {
        "Product Name": "",
        "Manufacturer": "",
        "Net Quantity": "",
        "MRP": "",
        "Unit Sale Price": "",
        "Manufacturing Date": "",
        "Batch Number": "",
        "Best Before / Expiry": "",
        "Consumer Care": "",
        "Country of Origin": "India"
    }

    fields_detected_count = 0
    fields_review_count = 0

    if request.method == "POST":
        action = request.form.get("action", "")

        # -------------------------------------------------------------
        # STEP 1: Scan & Extract using Multi-Image Hybrid Pipeline
        # Priority: Mistral OCR (Primary) -> Gemini Vision (Optional) -> Local OCR (Fallback)
        # -------------------------------------------------------------
        uploaded_files = request.files.getlist("product_images")
        if not uploaded_files or not any(f.filename for f in uploaded_files):
            single_img = request.files.get("product_image")
            if single_img and single_img.filename:
                uploaded_files = [single_img]

        valid_files = [f for f in uploaded_files if f and f.filename and f.filename.strip()]

        if action in ("scan", "extract") or valid_files:
            if valid_files:
                invalid_files = [f.filename for f in valid_files if not allowed_file(f.filename)]
                if invalid_files:
                    scan_message = f"Error: Invalid file format in '{', '.join(invalid_files)}'. Accepted formats: JPG, JPEG, PNG, WEBP, BMP."
                else:
                    saved_paths = []
                    saved_filenames = []
                    # Limit to max 10 images per product inspection
                    for f in valid_files[:10]:
                        safe_name = secure_filename(f.filename)
                        timestamp_prefix = int(time.time() * 1000)
                        unique_name = f"{timestamp_prefix}_{safe_name}"
                        image_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
                        f.save(image_path)
                        saved_paths.append(image_path)
                        saved_filenames.append(unique_name)

                    image_filenames = saved_filenames
                    image_filename = saved_filenames[0] if saved_filenames else None

                    # Execute Multi-Image Hybrid Extraction (Mistral OCR -> Gemini Vision -> Local OCR)
                    extract_res = run_hybrid_extraction(
                        image_path=saved_paths,
                        engine_mode=selected_engine,
                        gemini_api_key=effective_gemini_key,
                        mistral_api_key=effective_mistral_key
                    )

                    fields = extract_res["fields"]
                    fields_detail = extract_res.get("fields_detail", {})
                    engine_used = extract_res["engine"]
                    confidence = extract_res["confidence"]
                    time_sec = extract_res["processing_time_sec"]
                    fallback_note = extract_res.get("fallback_note", "")

                    fields_detected_count = sum(1 for f in fields_detail.values() if f.get("status") == "DETECTED" or (f.get("value") and str(f.get("value")).strip() and str(f.get("value")).strip() != "CONFLICT / REQUIRES REVIEW"))
                    fields_review_count = sum(1 for f in fields_detail.values() if f.get("requires_review"))

                    extraction_meta = {
                        "engine": engine_used,
                        "confidence": confidence,
                        "time_sec": time_sec,
                        "fallback_used": extract_res.get("fallback_used", False),
                        "fallback_note": fallback_note,
                        "mistral_configured": extract_res.get("mistral_configured", has_mistral_key),
                        "mistral_attempted": extract_res.get("mistral_attempted", False),
                        "mistral_success": extract_res.get("mistral_success", False),
                        "mistral_failure_reason": extract_res.get("mistral_failure_reason", None),
                        "gemini_configured": extract_res.get("gemini_configured", has_gemini_key),
                        "gemini_attempted": extract_res.get("gemini_attempted", False),
                        "gemini_success": extract_res.get("gemini_success", False),
                        "gemini_failure_reason": extract_res.get("gemini_failure_reason", None),
                        "local_attempted": extract_res.get("local_attempted", False),
                        "images_uploaded": extract_res.get("images_uploaded", len(saved_filenames)),
                        "images_processed": extract_res.get("images_processed", len(saved_filenames)),
                        "images_failed": extract_res.get("images_failed", 0),
                        "partial_success_note": extract_res.get("partial_success_note", ""),
                        "fields_detected_count": fields_detected_count,
                        "fields_review_count": fields_review_count
                    }

                    if extract_res.get("error"):
                        scan_message = f"Extraction Warning: {extract_res.get('error')}"
                    elif extract_res.get("partial_success_note"):
                        scan_message = f"{extract_res.get('partial_success_note')} Extracted declarations across {len(saved_filenames)} packaging view(s) in {time_sec}s using {engine_used}."
                    else:
                        scan_message = (
                            f"Successfully extracted declarations across {len(saved_filenames)} packaging view(s) in {time_sec}s using {engine_used}. "
                            "Review or refine the unified product record below, then execute the Legal Metrology audit."
                        )

        # -------------------------------------------------------------
        # STEP 2: Execute Legal Metrology PCR 2011 Audit
        # -------------------------------------------------------------
        elif action == "check":
            image_filename = request.form.get("current_image", None)
            current_images_raw = request.form.get("current_images_json", "")
            image_filenames = []
            if current_images_raw:
                try:
                    image_filenames = json.loads(current_images_raw)
                except Exception:
                    image_filenames = []
            if not image_filenames and image_filename:
                image_filenames = [image_filename]
            if not image_filename and image_filenames:
                image_filename = image_filenames[0]

            engine_used = request.form.get("engine_used", "Manual Verification")
            conf_str = request.form.get("confidence", "")
            try:
                conf_val = float(conf_str) if conf_str and conf_str != "None" else 0.0
            except ValueError:
                conf_val = 0.0

            fields = {
                "Product Name": request.form.get("product_name", "").strip(),
                "Manufacturer": request.form.get("manufacturer", "").strip(),
                "Address": request.form.get("address", "").strip(),
                "Net Quantity": request.form.get("net_quantity", "").strip(),
                "MRP": request.form.get("mrp", "").strip(),
                "Unit Sale Price": request.form.get("unit_sale_price", "").strip(),
                "Manufacturing Date": request.form.get("manufacturing_date", "").strip(),
                "Batch Number": request.form.get("batch_number", "").strip(),
                "Best Before / Expiry": request.form.get("expiry_or_best_before", "").strip(),
                "Consumer Care": request.form.get("consumer_care", "").strip(),
                "Country of Origin": request.form.get("country_of_origin", "India").strip()
            }

            commodity_category = request.form.get("commodity_category", "").strip() or None

            # Retrieve preserved fields_detail from extraction step
            fields_detail_raw = request.form.get("fields_detail_json", "")
            if fields_detail_raw:
                try:
                    fields_detail = json.loads(fields_detail_raw)
                except Exception:
                    fields_detail = {}

            # If inspector modified or manually entered fields, track provenance
            for k, current_v in fields.items():
                if k not in fields_detail:
                    fields_detail[k] = {
                        "value": current_v,
                        "evidence": "Manually entered by inspector" if current_v else "",
                        "confidence": None,
                        "source": "Manual Input",
                        "has_conflict": False,
                        "conflict_detail": None,
                        "requires_review": not bool(current_v)
                    }
                else:
                    prev_v = fields_detail[k].get("value", "")
                    if current_v != prev_v:
                        fields_detail[k]["value"] = current_v
                        fields_detail[k]["evidence"] = f"Inspector modified: '{prev_v}' -> '{current_v}'"
                        fields_detail[k]["source"] = "Inspector Refined"
                        fields_detail[k]["confidence"] = None  # Human entered/verified
                        fields_detail[k]["has_conflict"] = False
                        fields_detail[k]["conflict_detail"] = None

            # Run Field Validation & Statutory Compliance Rules
            val_results = validate_all_fields(fields, fields_detail)
            result = evaluate_compliance(fields, fields_detail, commodity_category=commodity_category)

            # Persist Unified Inspection Record to SQLite
            saved_image_field = json.dumps(image_filenames) if len(image_filenames) > 1 else (image_filename or None)
            inspection_id = save_inspection(
                product_name=fields.get("Product Name"),
                image_filename=saved_image_field,
                engine_used=engine_used,
                confidence=conf_val,
                overall_status=result["status"],
                compliance_score=result["score"],
                input_source="image" if image_filenames else "manual",
                fields=fields,
                validation_results=val_results,
                compliance_results=result
            )

    if fields_detail and not fields_detected_count:
        fields_detected_count = sum(1 for f in fields_detail.values() if f.get("status") == "DETECTED" or (f.get("value") and str(f.get("value")).strip() and str(f.get("value")).strip() != "CONFLICT / REQUIRES REVIEW"))
        fields_review_count = sum(1 for f in fields_detail.values() if f.get("requires_review"))

    return render_template(
        "index.html",
        result=result,
        fields=fields,
        fields_detail=fields_detail,
        image_filename=image_filename,
        image_filenames=image_filenames,
        extraction_meta=extraction_meta,
        scan_message=scan_message,
        selected_engine=selected_engine,
        has_gemini_key=has_gemini_key,
        has_mistral_key=has_mistral_key,
        fields_detected_count=fields_detected_count,
        fields_review_count=fields_review_count,
        inspection_id=inspection_id
    )


@app.route("/ecommerce", methods=["GET", "POST"])
def ecommerce():
    """E-Commerce product listing text audit mode."""
    result = None
    raw_text = ""
    inspection_id = None

    if request.method == "POST":
        raw_text = request.form.get("listing_text", "").strip()
        parsed = parse_ecommerce_listing(raw_text)

        if parsed.get("success"):
            fields = parsed["fields"]
            fields_detail = parsed.get("fields_detail", {})
            commodity_category = request.form.get("commodity_category", "").strip() or None
            val_results = validate_all_fields(fields, fields_detail)
            result = evaluate_compliance(fields, fields_detail, commodity_category=commodity_category)

            inspection_id = save_inspection(
                product_name=fields.get("Product Name"),
                image_filename=None,
                engine_used="E-Commerce Text Parser",
                confidence=0.0,
                overall_status=result["status"],
                compliance_score=result["score"],
                input_source="ecommerce",
                fields=fields,
                validation_results=val_results,
                compliance_results=result
            )

    return render_template(
        "ecommerce.html",
        result=result,
        raw_text=raw_text,
        inspection_id=inspection_id
    )


@app.route("/history")
def history():
    """Historical audit log of all inspections."""
    inspections = list_inspections(limit=100)
    return render_template("history.html", inspections=inspections)


@app.route("/certificates")
def certificates():
    """Certificates registry for verified legal metrology inspection records."""
    inspections = list_inspections(limit=100)
    return render_template("certificates.html", inspections=inspections)


@app.route("/analytics")
def analytics():
    """Enterprise compliance analytics and diagnostic metrics."""
    inspections = list_inspections(limit=500)
    total_scans = len(inspections)
    compliant_count = sum(1 for i in inspections if i.get("overall_status") == "COMPLIANT")
    non_compliant_count = sum(1 for i in inspections if i.get("overall_status") in ("NON-COMPLIANT", "NON_COMPLIANT"))
    review_count = sum(1 for i in inspections if i.get("overall_status") == "REQUIRES_REVIEW")
    compliance_rate = round((compliant_count / total_scans * 100), 1) if total_scans > 0 else 100.0

    avg_score = round(sum(float(i.get("compliance_score") or 0) for i in inspections) / total_scans, 1) if total_scans > 0 else 100.0
    metrics = {
        "total": total_scans,
        "compliant": compliant_count,
        "non_compliant": non_compliant_count,
        "requires_review": review_count,
        "compliant_pct": compliance_rate,
        "avg_score": avg_score
    }

    return render_template(
        "analytics.html",
        total_scans=total_scans,
        compliant_count=compliant_count,
        non_compliant_count=non_compliant_count,
        review_count=review_count,
        compliance_rate=compliance_rate,
        metrics=metrics,
        inspections=inspections[:10]
    )


@app.route("/settings")
def settings():
    """System configuration and diagnostic settings."""
    status = get_provider_status()
    return render_template("settings.html", status=status, provider_status=status)


@app.route("/help")
def help_support():
    """Inspector handbook and Legal Metrology PCR 2011 compliance guidelines."""
    return render_template("help.html")


@app.route("/inspection/<inspection_id>")
def view_inspection(inspection_id):
    """View details of a past inspection."""
    data = get_inspection(inspection_id)
    if not data:
        return redirect(url_for("history"))

    effective_key = get_api_key()
    fields = data.get("fields", {})
    audit_table = data.get("compliance", {}).get("audit_table", [])
    fields_detail = {}
    rule_to_field = {
        "PCR-01": "Product Name",
        "PCR-02": "Manufacturer",
        "PCR-03": "Net Quantity",
        "PCR-04": "MRP",
        "PCR-05": "Unit Sale Price",
        "PCR-06": "Manufacturing Date",
        "PCR-07": "Consumer Care",
        "PCR-08": "Country of Origin",
        "PCR-09": "Best Before / Expiry",
        "PCR-10": "Batch Number",
    }
    for row in audit_table:
        rid = row.get("rule_id")
        fname = rule_to_field.get(rid)
        if fname:
            fields_detail[fname] = {
                "value": row.get("extracted_value") or fields.get(fname, ""),
                "evidence": row.get("extracted_evidence", ""),
                "confidence": row.get("confidence"),
                "source": row.get("extraction_source", data.get("engine_used", "Local OCR")),
                "source_images": row.get("source_images") or ([row.get("source_image")] if row.get("source_image") else []),
                "source_image": row.get("source_image", ""),
                "has_conflict": row.get("has_conflict", False),
                "conflict_detail": row.get("conflict_detail")
            }

    raw_img_val = data.get("image_filename")
    image_filenames = []
    primary_image = ""
    if raw_img_val:
        if raw_img_val.strip().startswith("["):
            try:
                parsed_imgs = json.loads(raw_img_val)
                if isinstance(parsed_imgs, list):
                    image_filenames = parsed_imgs
                    if image_filenames:
                        primary_image = image_filenames[0]
            except Exception:
                image_filenames = [raw_img_val]
                primary_image = raw_img_val
        else:
            image_filenames = [raw_img_val]
            primary_image = raw_img_val

    fields_detected_count = sum(1 for f in fields_detail.values() if (f.get("value") and str(f.get("value")).strip() and str(f.get("value")).strip() not in ["NOT_DETECTED", "REQUIRES_REVIEW", "CONFLICT / REQUIRES REVIEW"]))
    fields_review_count = sum(1 for f in fields_detail.values() if f.get("has_conflict") or f.get("requires_review"))

    return render_template(
        "index.html",
        result=data.get("compliance"),
        fields=fields,
        fields_detail=fields_detail,
        image_filename=primary_image,
        image_filenames=image_filenames,
        extraction_meta={
            "engine": data.get("engine_used", ""),
            "confidence": data.get("confidence", 0.0),
            "time_sec": 0.0,
            "fallback_note": "",
            "images_uploaded": len(image_filenames),
            "images_processed": len(image_filenames),
            "images_failed": 0
        },
        selected_engine="auto",
        has_gemini_key=bool(effective_key),
        has_mistral_key=bool(os.environ.get("MISTRAL_API_KEY")),
        fields_detected_count=fields_detected_count,
        fields_review_count=fields_review_count,
        inspection_id=data.get("id")
    )


@app.route("/inspection/<inspection_id>/report")
def official_report(inspection_id):
    """Official printable Legal Metrology statutory compliance certificate."""
    data = get_inspection(inspection_id)
    if not data:
        return redirect(url_for("history"))

    raw_img_val = data.get("image_filename")
    if raw_img_val and raw_img_val.strip().startswith("["):
        try:
            parsed_imgs = json.loads(raw_img_val)
            if isinstance(parsed_imgs, list) and parsed_imgs:
                data["image_filenames"] = parsed_imgs
                data["image_filename"] = parsed_imgs[0]
        except Exception:
            pass
    return render_template("report.html", inspection=data)


@app.route("/api/inspection/<inspection_id>/json")
def api_inspection_json(inspection_id):
    """JSON API export for inspection report."""
    data = get_inspection(inspection_id)
    if not data:
        return jsonify({"error": "Inspection not found"}), 404
    raw_img_val = data.get("image_filename")
    if raw_img_val and raw_img_val.strip().startswith("["):
        try:
            parsed_imgs = json.loads(raw_img_val)
            if isinstance(parsed_imgs, list):
                data["image_filenames"] = parsed_imgs
        except Exception:
            pass
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))