"""
Hybrid Extraction Orchestrator for Packaged Commodity Compliance.
Intelligently coordinates Gemini Multimodal Vision AI and Local Tesseract OCR.
Guarantees automatic, graceful fallback to offline processing when API is unavailable.
"""

import os
import sys
import time
import re
from typing import Dict, Any, Optional, List, Tuple, Union

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_extraction_confidence_threshold
from gemini_vision import extract_with_gemini, is_gemini_available, get_api_key
from mistral_ocr import extract_with_mistral, is_mistral_available, get_mistral_api_key
from ocr_test import extract_product_info as extract_local_ocr
from validation.field_validator import validate_date, calculate_and_validate_usp, validate_mrp, validate_net_quantity

# Backwards compatibility alias
get_gemini_api_key = get_api_key

MANDATORY_FIELDS = [
    "Product Name",
    "Manufacturer",
    "Address",
    "Net Quantity",
    "MRP",
    "Unit Sale Price",
    "Manufacturing Date",
    "Batch Number",
    "Best Before / Expiry",
    "Consumer Care",
    "Country of Origin"
]


def _run_single_hybrid_extraction(
    image_path: str,
    engine_mode: str = "auto",
    gemini_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes packaging extraction according to the selected engine mode.
    
    Modes:
    - 'auto': 3-Tier cascade: Gemini Vision -> Mistral OCR -> Local OCR.
    - 'gemini': Gemini Vision AI only (fails cleanly without Local OCR fallback).
    - 'mistral': Mistral OCR only (fails cleanly without Local OCR fallback).
    - 'local': Local OpenCV + Tesseract OCR only (strictly offline).
    - 'dual': Cross-engine comparison between Gemini Vision and Local OCR.
    """
    t0 = time.time()
    mode = (engine_mode or "auto").lower()
    effective_gemini_key = get_api_key(gemini_api_key)
    effective_mistral_key = get_mistral_api_key(mistral_api_key)

    engine_used = ""
    fields = {}
    fields_detail = {}
    confidence = None
    time_sec = 0.0
    fallback_used = False
    fallback_note = ""
    error_message = None
    raw_text = ""
    gemini_fields = None
    local_fields = None

    gemini_configured = bool(effective_gemini_key and len(effective_gemini_key) > 5)
    gemini_attempted = False
    gemini_success = False
    gemini_failure_reason = None

    mistral_configured = bool(effective_mistral_key and len(effective_mistral_key) > 5)
    mistral_attempted = False
    mistral_success = False
    mistral_failure_reason = None
    local_attempted = False

    if mode == "dual":
        # Dual-engine comparison mode: executes both Gemini Vision and Local OCR to identify discrepancies
        if gemini_configured:
            gemini_attempted = True
            gemini_res = extract_with_gemini(image_path, api_key=effective_gemini_key)
            if gemini_res.get("success"):
                gemini_success = True
            else:
                gemini_failure_reason = gemini_res.get("error")
        else:
            gemini_res = {"success": False, "error": "Gemini API key not configured."}
            gemini_failure_reason = "Gemini API key not configured."

        local_attempted = True
        local_res = extract_local_ocr(image_path)
        time_sec = round(time.time() - t0, 2)

        local_fields = {
            "Product Name": local_res.get("product_name", ""),
            "Manufacturer": local_res.get("manufacturer", ""),
            "Address": local_res.get("address", ""),
            "Net Quantity": local_res.get("net_quantity", ""),
            "MRP": local_res.get("mrp", ""),
            "Unit Sale Price": local_res.get("unit_sale_price", ""),
            "Manufacturing Date": local_res.get("manufacturing_date", ""),
            "Batch Number": local_res.get("batch_number", ""),
            "Best Before / Expiry": local_res.get("expiry_or_best_before", ""),
            "Consumer Care": local_res.get("consumer_care", ""),
            "Country of Origin": local_res.get("country_of_origin", "India")
        }

        if gemini_res.get("success"):
            gemini_fields = gemini_res["fields"]
            fields = dict(gemini_fields)
            fields_detail = gemini_res.get("fields_detail", {})
            confidence = gemini_res.get("confidence", 95.0)
            engine_used = f"Dual-Engine Audit: Gemini Vision + Local OCR"
            raw_text = gemini_res.get("raw_text", "") + "\n--- OCR ---\n" + local_res.get("raw_text", "")

            # Detect conflicts between Gemini and Local OCR
            conflicts = detect_extraction_conflicts(gemini_fields, local_fields)
            for fname, cinfo in conflicts.items():
                if fname in fields_detail:
                    fields_detail[fname]["has_conflict"] = True
                    fields_detail[fname]["conflict_detail"] = cinfo["conflict_detail"]
                    fields_detail[fname]["requires_review"] = True
        else:
            fallback_used = True
            fallback_note = f"Gemini failed in dual mode ({gemini_res.get('error')}). Using Local OCR results."
            engine_used = "Local OCR (OpenCV + Tesseract - Fallback)"
            fields = local_fields
            confidence = local_res.get("confidence")
            raw_text = local_res.get("raw_text", "")
            fields_detail = _build_local_fields_detail(fields, confidence or 0.0)

    elif mode == "auto":
        # 3-Tier Cascade: Tier 1 (Mistral OCR - Primary) -> Tier 2 (Gemini Vision - Optional) -> Tier 3 (Local OCR - Offline Fallback)
        mistral_handled = False
        if mistral_configured:
            mistral_attempted = True
            mistral_res = extract_with_mistral(image_path, api_key=effective_mistral_key)
            if mistral_res.get("success"):
                mistral_success = True
                mistral_handled = True
                engine_used = f"Mistral OCR — Active ({mistral_res.get('model', 'mistral-ocr-latest')})"
                fields = mistral_res["fields"]
                fields_detail = mistral_res.get("fields_detail", {})
                confidence = mistral_res.get("confidence", 90.0)
                time_sec = mistral_res.get("processing_time_sec", round(time.time() - t0, 2))
                raw_text = mistral_res.get("raw_text", "")
                fallback_used = False
                fallback_note = ""
            else:
                mistral_success = False
                mistral_failure_reason = mistral_res.get("error", "Unknown Mistral error")
        else:
            mistral_attempted = False
            mistral_failure_reason = "Mistral API key not configured."

        # If Mistral did not succeed, attempt Tier 2: Gemini Vision (Optional)
        gemini_handled = False
        if not mistral_handled:
            if gemini_configured:
                gemini_attempted = True
                gemini_res = extract_with_gemini(image_path, api_key=effective_gemini_key)
                if gemini_res.get("success"):
                    gemini_success = True
                    gemini_handled = True
                    fallback_used = True
                    fallback_note = f"Mistral OCR failed ({mistral_failure_reason}) — Gemini Vision fallback used."
                    engine_used = f"Gemini Vision — Active ({gemini_res.get('model', 'gemini-3.6-flash')})"
                    fields = gemini_res["fields"]
                    fields_detail = gemini_res.get("fields_detail", {})
                    confidence = gemini_res.get("confidence", 95.0)
                    time_sec = gemini_res.get("processing_time_sec", round(time.time() - t0, 2))
                    raw_text = gemini_res.get("raw_text", "")
                else:
                    gemini_success = False
                    gemini_failure_reason = gemini_res.get("error", "Unknown Gemini error")
            else:
                gemini_attempted = False
                gemini_failure_reason = "Gemini API key not configured (Optional)."

        # If neither Mistral nor Gemini succeeded, engage Tier 3: Local OCR (Offline Fallback)
        if not mistral_handled and not gemini_handled:
            fallback_used = True
            local_attempted = True
            if mistral_attempted and gemini_attempted:
                fallback_note = f"Mistral OCR failed ({mistral_failure_reason}) — Gemini Vision failed ({gemini_failure_reason}) — Local OCR fallback used."
                engine_used = "Local OCR (OpenCV + Tesseract - Fallback)"
            elif mistral_attempted and not gemini_configured:
                fallback_note = f"Mistral OCR failed ({mistral_failure_reason}) — Gemini unavailable — Local OCR fallback used."
                engine_used = "Local OCR (OpenCV + Tesseract - Fallback)"
            elif not mistral_configured and gemini_attempted:
                fallback_note = f"Mistral unavailable — Gemini Vision failed ({gemini_failure_reason}) — Local OCR fallback used."
                engine_used = "Local OCR (OpenCV + Tesseract - Fallback)"
            else:
                fallback_note = "No AI API keys configured. Automatically using Local OCR (OpenCV + Tesseract)."
                engine_used = "Local OCR (OpenCV + Tesseract - Fallback)"

            local_res = extract_local_ocr(image_path)
            raw_conf = local_res.get("confidence", 0.0)
            confidence = raw_conf if raw_conf > 0 else None
            time_sec = local_res.get("processing_time_sec", round(time.time() - t0, 2))
            raw_text = local_res.get("raw_text", "")
            fields = {
                "Product Name": local_res.get("product_name", ""),
                "Manufacturer": local_res.get("manufacturer", ""),
                "Address": local_res.get("address", ""),
                "Net Quantity": local_res.get("net_quantity", ""),
                "MRP": local_res.get("mrp", ""),
                "Unit Sale Price": local_res.get("unit_sale_price", ""),
                "Manufacturing Date": local_res.get("manufacturing_date", ""),
                "Batch Number": local_res.get("batch_number", ""),
                "Best Before / Expiry": local_res.get("expiry_or_best_before", ""),
                "Consumer Care": local_res.get("consumer_care", ""),
                "Country of Origin": local_res.get("country_of_origin", "India")
            }
            fields_detail = _build_local_fields_detail(fields, raw_conf)

    elif mode == "mistral":
        # Mistral OCR Only: strictly does NOT fall back to Local OCR
        if mistral_configured:
            mistral_attempted = True
            mistral_res = extract_with_mistral(image_path, api_key=effective_mistral_key)
            if mistral_res.get("success"):
                mistral_success = True
                engine_used = f"Mistral OCR — Active ({mistral_res.get('model', 'mistral-ocr-latest')})"
                fields = mistral_res["fields"]
                fields_detail = mistral_res.get("fields_detail", {})
                confidence = mistral_res.get("confidence", 90.0)
                time_sec = mistral_res.get("processing_time_sec", round(time.time() - t0, 2))
                raw_text = mistral_res.get("raw_text", "")
            else:
                mistral_success = False
                mistral_failure_reason = mistral_res.get("error", "Unknown error")
                error_message = mistral_failure_reason
                engine_used = "Mistral OCR (Failed)"
                time_sec = mistral_res.get("processing_time_sec", round(time.time() - t0, 2))
        else:
            mistral_attempted = False
            mistral_success = False
            error_message = "Mistral API key is not configured in backend environment (.env). Configure MISTRAL_API_KEY to use Mistral OCR Only mode."
            engine_used = "Mistral OCR (Key Required)"

    elif mode == "gemini":
        # Gemini Vision AI Only: strictly does NOT fall back to Local OCR or Mistral
        if gemini_configured:
            gemini_attempted = True
            gemini_res = extract_with_gemini(image_path, api_key=effective_gemini_key)
            if gemini_res.get("success"):
                gemini_success = True
                engine_used = f"Gemini Vision — Active ({gemini_res.get('model', 'gemini-3.6-flash')})"
                fields = gemini_res["fields"]
                fields_detail = gemini_res.get("fields_detail", {})
                confidence = gemini_res.get("confidence", 95.0)
                time_sec = gemini_res.get("processing_time_sec", round(time.time() - t0, 2))
                raw_text = gemini_res.get("raw_text", "")
            else:
                gemini_success = False
                gemini_failure_reason = gemini_res.get("error", "Unknown error")
                error_message = gemini_failure_reason
                engine_used = "Gemini Vision AI (Failed)"
                time_sec = gemini_res.get("processing_time_sec", round(time.time() - t0, 2))
        else:
            gemini_attempted = False
            gemini_success = False
            error_message = "Gemini API key is not configured in backend environment (.env). Configure GEMINI_API_KEY to use Gemini Vision AI Only mode."
            engine_used = "Gemini Vision AI (Key Required)"

    else:
        # Local OCR Only mode: strictly offline, never attempts Gemini or Mistral
        gemini_attempted = False
        gemini_success = False
        mistral_attempted = False
        mistral_success = False
        local_attempted = True
        local_res = extract_local_ocr(image_path)
        engine_used = "Local OCR (OpenCV + Tesseract)"
        raw_conf = local_res.get("confidence", 0.0)
        confidence = raw_conf if raw_conf > 0 else None
        time_sec = local_res.get("processing_time_sec", round(time.time() - t0, 2))
        raw_text = local_res.get("raw_text", "")
        fields = {
            "Product Name": local_res.get("product_name", ""),
            "Manufacturer": local_res.get("manufacturer", ""),
            "Address": local_res.get("address", ""),
            "Net Quantity": local_res.get("net_quantity", ""),
            "MRP": local_res.get("mrp", ""),
            "Unit Sale Price": local_res.get("unit_sale_price", ""),
            "Manufacturing Date": local_res.get("manufacturing_date", ""),
            "Batch Number": local_res.get("batch_number", ""),
            "Best Before / Expiry": local_res.get("expiry_or_best_before", ""),
            "Consumer Care": local_res.get("consumer_care", ""),
            "Country of Origin": local_res.get("country_of_origin", "India")
        }
        fields_detail = _build_local_fields_detail(fields, raw_conf)

    # -------------------------------------------------------------
    # Automated Statutory Unit Sale Price (USP) Calculation & Verification
    # -------------------------------------------------------------
    if fields:
        raw_mrp = fields.get("MRP", "")
        raw_qty = fields.get("Net Quantity", "")
        printed_usp_raw = fields.get("Unit Sale Price", "")

        mrp_val = validate_mrp(raw_mrp)
        qty_val = validate_net_quantity(raw_qty, commodity_name=fields.get("Product Name", ""))

        usp_res = calculate_and_validate_usp(
            mrp=mrp_val.get("numeric_price"),
            quantity=qty_val.get("numeric_value"),
            unit=qty_val.get("normalized_unit"),
            declared_usp_str=printed_usp_raw,
            raw_quantity_str=raw_qty
        )

        # Ensure Unit Sale Price entry in fields_detail
        if "Unit Sale Price" not in fields_detail:
            fields_detail["Unit Sale Price"] = {
                "value": printed_usp_raw,
                "evidence": f"Detected printed USP: '{printed_usp_raw}'" if printed_usp_raw else "",
                "confidence": None,
                "source": "Local OCR" if "Local OCR" in engine_used else ("Mistral OCR" if "Mistral" in engine_used else "Gemini Vision AI"),
                "has_conflict": False,
                "conflict_detail": None,
                "requires_review": False
            }

        usp_fd = fields_detail["Unit Sale Price"]
        usp_fd["calculated_usp"] = usp_res.get("calculated_usp_formatted")
        usp_fd["calculated_usp_num"] = usp_res.get("calculated_usp")
        usp_fd["calculated_unit"] = usp_res.get("calculated_unit")
        usp_fd["evidence_string"] = usp_res.get("evidence_string")
        usp_fd["printed_usp"] = printed_usp_raw
        usp_fd["printed_usp_num"] = usp_res.get("printed_usp_num")
        usp_fd["has_mismatch"] = usp_res.get("has_mismatch", False)
        usp_fd["mismatch_message"] = usp_res.get("mismatch_message")
        usp_fd["is_exempt"] = usp_res.get("is_exempt", False)
        usp_fd["exempt_reason"] = usp_res.get("exempt_reason", "")
        usp_fd["status"] = usp_res.get("status", "CALCULATED")

        if usp_res.get("has_mismatch"):
            usp_fd["requires_review"] = True

        calc_fmt = usp_res.get("calculated_usp_formatted")
        if calc_fmt == "NOT_CALCULABLE" or not usp_res.get("calculated_usp"):
            calc_fmt = None
        calculated_usp_str = calc_fmt or "Not Calculable"
        printed_usp_str = printed_usp_raw or "Not Printed / Not Detected"

        fields["Calculated USP"] = calculated_usp_str
        fields["Printed USP"] = printed_usp_str

        usp_eng = usp_fd.get("engine", "Local OCR" if "Local" in engine_used else ("Mistral OCR" if "Mistral" in engine_used else "Gemini Vision"))

        fields_detail["Calculated USP"] = {
            "value": calculated_usp_str,
            "evidence": usp_res.get("evidence_string") or f"Computed from MRP ({raw_mrp}) and Net Qty ({raw_qty})",
            "confidence": 100.0 if calc_fmt else None,
            "source": "Auto-Calculated (PCR 2011)",
            "engine": usp_eng,
            "status": "CALCULATED" if calc_fmt else "NOT_CALCULABLE",
            "has_conflict": False,
            "conflict_detail": None,
            "requires_review": (calc_fmt is None)
        }

        fields_detail["Printed USP"] = {
            "value": printed_usp_str,
            "evidence": f"Detected printed declaration: '{printed_usp_raw}'" if printed_usp_raw else "No printed USP declaration detected",
            "confidence": 90.0 if printed_usp_raw else None,
            "source": usp_fd.get("source", "Packaging Label"),
            "engine": usp_eng,
            "status": "DETECTED" if printed_usp_raw else "NOT_DETECTED",
            "has_conflict": False,
            "conflict_detail": None,
            "requires_review": False
        }

        # If printed USP was not detected on packaging, auto-populate with statutory calculated USP
        if not printed_usp_raw and usp_res.get("calculated_usp_formatted") and usp_res.get("calculated_usp_formatted") != "NOT_CALCULABLE":
            fields["Unit Sale Price"] = usp_res["calculated_usp_formatted"]
            usp_fd["value"] = usp_res["calculated_usp_formatted"]
            usp_fd["evidence"] = usp_res["evidence_string"]
            usp_fd["source"] = "Auto-Calculated (PCR 2011)"

    return {
        "success": bool(fields and not error_message),
        "engine": engine_used,
        "mode_requested": mode,
        "confidence": confidence,
        "processing_time_sec": time_sec,
        "fallback_used": fallback_used,
        "fallback_note": fallback_note,
        "gemini_configured": gemini_configured,
        "gemini_attempted": gemini_attempted,
        "gemini_success": gemini_success,
        "gemini_failure_reason": gemini_failure_reason,
        "mistral_configured": mistral_configured,
        "mistral_attempted": mistral_attempted,
        "mistral_success": mistral_success,
        "mistral_failure_reason": mistral_failure_reason,
        "local_attempted": local_attempted,
        "error": error_message,
        "fields": fields,
        "fields_detail": fields_detail,
        "raw_text": raw_text,
        "gemini_fields": gemini_fields,
        "local_fields": local_fields
    }


def merge_multi_image_records(
    per_image_results: Optional[List[Dict[str, Any]]] = None,
    total_time_sec: float = 0.0,
    successful_results: Optional[List[Dict[str, Any]]] = None,
    total_images_uploaded: Optional[int] = None,
    images_processed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Combines extractions from multiple package views of the same physical product into ONE unified record.
    - Preserves field-level provenance (source image, engine, confidence).
    - Deduplicates identical declarations across views.
    - Flags conflicting declarations as CONFLICT / REQUIRES REVIEW without guessing.
    - Calculates deterministic USP from the merged record.
    - Runs once for the physical product.
    """
    if per_image_results is None:
        per_image_results = successful_results or []
    elif successful_results is not None and not per_image_results:
        per_image_results = successful_results

    # Ensure each item in per_image_results has default success flag if fields are present
    for r in per_image_results:
        if "success" not in r:
            r["success"] = bool(r.get("fields"))

    images_uploaded = total_images_uploaded if total_images_uploaded is not None else len(per_image_results)
    successful_results = [r for r in per_image_results if r.get("success") and r.get("fields")]
    failed_results = [r for r in per_image_results if not (r.get("success") and r.get("fields"))]
    
    if images_processed is None:
        images_processed = len(successful_results)
    images_failed = len(failed_results)

    failed_labels = [r.get("image_label", f"Image {r.get('image_index', '?')}") for r in failed_results]

    partial_success_note = ""
    if failed_results and successful_results:
        partial_success_note = (
            f"Partial multi-view extraction: {images_processed} of {images_uploaded} images processed successfully. "
            f"{', '.join(failed_labels)} requires review."
        )

    if images_processed == 0:
        return {
            "success": False,
            "engine": "Failed",
            "confidence": 0.0,
            "processing_time_sec": total_time_sec,
            "images_uploaded": images_uploaded,
            "images_processed": 0,
            "images_failed": images_failed,
            "failed_image_labels": failed_labels,
            "partial_success_note": "All uploaded images failed extraction.",
            "fallback_used": False,
            "fallback_note": "",
            "error": "All uploaded product images failed extraction. Please verify image clarity and file format.",
            "fields": {f: "" for f in MANDATORY_FIELDS},
            "fields_detail": {},
            "raw_text": "",
            "per_image_results": per_image_results
        }

    merged_fields = {}
    merged_fields_detail = {}

    for field_name in MANDATORY_FIELDS:
        candidates = []
        for img_res in successful_results:
            raw_val = img_res.get("fields", {}).get(field_name, "")
            val = str(raw_val).strip() if raw_val is not None else ""
            if val and val.upper() not in ["NOT_DETECTED", "REQUIRES_REVIEW", "NONE", "UNREADABLE", "UNAVAILABLE"]:
                c_detail = img_res.get("fields_detail", {}).get(field_name, {})
                c_conf = c_detail.get("confidence") if c_detail.get("confidence") is not None else img_res.get("confidence")
                c_ev = c_detail.get("evidence") or val
                c_eng = c_detail.get("engine") or img_res.get("engine", "OCR")
                norm_val = normalize_field_for_comparison(field_name, val)
                candidates.append({
                    "image_label": img_res.get("image_label", f"Image {img_res.get('image_index', 1)}"),
                    "image_index": img_res.get("image_index", 1),
                    "image_filename": img_res.get("image_filename", ""),
                    "engine": c_eng,
                    "value": val,
                    "normalized": norm_val,
                    "confidence": c_conf,
                    "evidence": c_ev
                })

        if not candidates:
            merged_fields[field_name] = ""
            merged_fields_detail[field_name] = {
                "value": "",
                "evidence": f"Not detected across {images_processed} uploaded view(s)",
                "confidence": None,
                "engine": "Multi-Image Inspection",
                "source": "Multi-Image Inspection",
                "source_image": None,
                "source_images": [],
                "has_conflict": False,
                "conflict_detail": None,
                "status": "NOT_DETECTED",
                "requires_review": False if field_name == "Address" else True
            }
        elif len(candidates) == 1:
            c = candidates[0]
            merged_fields[field_name] = c["value"]
            merged_fields_detail[field_name] = {
                "value": c["value"],
                "evidence": c["evidence"] if c["evidence"] != c["value"] else f"Detected on {c['image_label']}: '{c['value']}'",
                "confidence": c["confidence"],
                "engine": c["engine"],
                "source": f"{c['image_label']} ({c['engine']})",
                "source_image": c["image_label"],
                "source_images": [c["image_label"]],
                "has_conflict": False,
                "conflict_detail": None,
                "status": "DETECTED",
                "requires_review": False
            }
        else:
            # Candidates detected across multiple views: check deduplication vs conflict
            unique_norms = {}
            for c in candidates:
                unique_norms.setdefault(c["normalized"], []).append(c)

            is_consistent = len(unique_norms) == 1
            if not is_consistent and field_name in ["Product Name", "Manufacturer", "Consumer Care"]:
                norm_keys = list(unique_norms.keys())
                longest_key = max(norm_keys, key=len)
                if all(k in longest_key for k in norm_keys):
                    is_consistent = True

            if is_consistent:
                # DEDUPLICATION: All views agree
                chosen = max(candidates, key=lambda x: (x["confidence"] if x["confidence"] is not None else 0, len(x["value"])))
                all_labels = list(dict.fromkeys(c["image_label"] for c in candidates))
                sources_str = ", ".join(all_labels)

                merged_fields[field_name] = chosen["value"]
                merged_fields_detail[field_name] = {
                    "value": chosen["value"],
                    "evidence": f"Verified across multiple views: {sources_str} ('{chosen['value']}')",
                    "confidence": max((c["confidence"] for c in candidates if c["confidence"] is not None), default=chosen["confidence"]),
                    "engine": chosen["engine"],
                    "source": f"Multi-View ({sources_str})",
                    "source_image": all_labels[0],
                    "source_images": all_labels,
                    "has_conflict": False,
                    "conflict_detail": None,
                    "status": "DETECTED",
                    "requires_review": False
                }
            else:
                # CONFLICT: Different views disagree!
                conflict_summary = " vs ".join(f"{c['image_label']} ('{c['value']}')" for c in candidates)
                all_labels = list(dict.fromkeys(c["image_label"] for c in candidates))

                merged_fields[field_name] = "CONFLICT / REQUIRES REVIEW"
                merged_fields_detail[field_name] = {
                    "value": "CONFLICT / REQUIRES REVIEW",
                    "evidence": f"Conflicting declarations: {conflict_summary}",
                    "confidence": None,
                    "engine": candidates[0]["engine"],
                    "source": f"Conflicting Views ({', '.join(all_labels)})",
                    "source_image": all_labels[0],
                    "source_images": all_labels,
                    "has_conflict": True,
                    "conflict_detail": f"Conflicting declarations: {conflict_summary}",
                    "conflicting_candidates": candidates,
                    "status": "CONFLICT",
                    "requires_review": True
                }

    # -------------------------------------------------------------
    # Automated USP Calculation on Unified Product Record
    # -------------------------------------------------------------
    raw_mrp = merged_fields.get("MRP", "")
    raw_qty = merged_fields.get("Net Quantity", "")

    # Check if any view detected printed USP
    printed_usp_candidate = None
    usp_val = merged_fields.get("Unit Sale Price", "")
    if usp_val and usp_val != "CONFLICT / REQUIRES REVIEW":
        printed_usp_candidate = usp_val
    else:
        usp_candidates = [
            c for img_res in successful_results
            for c in [img_res.get("fields", {}).get("Unit Sale Price", "").strip()]
            if c and c.upper() not in ["NOT_DETECTED", "REQUIRES_REVIEW", "NONE"]
        ]
        if usp_candidates:
            printed_usp_candidate = usp_candidates[0]

    mrp_val = validate_mrp(raw_mrp) if raw_mrp and "CONFLICT" not in raw_mrp else {"numeric_price": None}
    qty_val = validate_net_quantity(raw_qty, commodity_name=merged_fields.get("Product Name", "")) if raw_qty and "CONFLICT" not in raw_qty else {"numeric_value": None, "normalized_unit": None}

    usp_res = calculate_and_validate_usp(
        mrp=mrp_val.get("numeric_price"),
        quantity=qty_val.get("numeric_value"),
        unit=qty_val.get("normalized_unit"),
        declared_usp_str=printed_usp_candidate,
        raw_quantity_str=raw_qty if "CONFLICT" not in raw_qty else None
    )

    usp_fd = merged_fields_detail.get("Unit Sale Price", {})
    calc_fmt = usp_res.get("calculated_usp_formatted")
    if calc_fmt == "NOT_CALCULABLE" or not usp_res.get("calculated_usp"):
        calc_fmt = None
    usp_fd["calculated_usp"] = calc_fmt
    usp_fd["calculated_usp_num"] = usp_res.get("calculated_usp")
    usp_fd["calculated_unit"] = usp_res.get("calculated_unit")
    usp_fd["evidence_string"] = usp_res.get("evidence_string")
    usp_fd["printed_usp"] = printed_usp_candidate or ""
    usp_fd["printed_usp_num"] = usp_res.get("printed_usp_num")
    usp_fd["is_exempt"] = usp_res.get("is_exempt", False)
    usp_fd["exempt_reason"] = usp_res.get("exempt_reason", "")

    if usp_res.get("has_mismatch"):
        usp_fd["has_mismatch"] = True
        usp_fd["mismatch_message"] = usp_res.get("mismatch_message")
        usp_fd["requires_review"] = True

    calculated_usp_str = calc_fmt or "Not Calculable"
    printed_usp_str = printed_usp_candidate or "Not Printed / Not Detected"

    # Aggregate telemetry
    engines_used = list(dict.fromkeys(r.get("engine", "") for r in successful_results if r.get("engine")))
    if any("Mistral" in e for e in engines_used):
        overall_engine = "Mistral OCR (Multi-View)"
    elif any("Gemini" in e for e in engines_used):
        overall_engine = "Gemini Vision (Multi-View Fallback)"
    elif engines_used:
        overall_engine = "Local OCR (Multi-View Fallback)"
    else:
        overall_engine = "Multi-Image Inspection"

    confidences = [r.get("confidence") for r in successful_results if r.get("confidence") is not None]
    avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else None

    merged_fields["Calculated USP"] = calculated_usp_str
    merged_fields["Printed USP"] = printed_usp_str

    usp_merge_eng = usp_fd.get("engine", overall_engine)

    merged_fields_detail["Calculated USP"] = {
        "value": calculated_usp_str,
        "evidence": usp_res.get("evidence_string") or f"Computed from MRP ({raw_mrp}) and Net Qty ({raw_qty})",
        "confidence": 100.0 if calc_fmt else None,
        "source": "Auto-Calculated (PCR 2011 Rule 6(1)(m))",
        "engine": usp_merge_eng,
        "status": "CALCULATED" if calc_fmt else "NOT_CALCULABLE",
        "has_conflict": False,
        "conflict_detail": None,
        "requires_review": (calc_fmt is None)
    }

    merged_fields_detail["Printed USP"] = {
        "value": printed_usp_str,
        "evidence": f"Detected printed declaration: '{printed_usp_candidate}'" if printed_usp_candidate else "No printed USP declaration detected across packaging views",
        "confidence": 90.0 if printed_usp_candidate else None,
        "source": usp_fd.get("source", "Packaging Label"),
        "engine": usp_merge_eng,
        "status": "DETECTED" if printed_usp_candidate else "NOT_DETECTED",
        "has_conflict": False,
        "conflict_detail": None,
        "requires_review": False
    }

    if not printed_usp_candidate and usp_res.get("calculated_usp_formatted") and usp_res.get("calculated_usp_formatted") != "NOT_CALCULABLE":
        merged_fields["Unit Sale Price"] = usp_res["calculated_usp_formatted"]
        usp_fd["value"] = usp_res["calculated_usp_formatted"]
        usp_fd["evidence"] = usp_res["evidence_string"]
        usp_fd["source"] = "Auto-Calculated (PCR 2011)"
        usp_fd["status"] = "CALCULATED"
        usp_fd["requires_review"] = False
    elif usp_res.get("status") == "NOT_CALCULABLE" and not printed_usp_candidate:
        usp_fd["status"] = "NOT_CALCULABLE"
        usp_fd["requires_review"] = True
        merged_fields["Unit Sale Price"] = "REQUIRES_REVIEW"

    # Telemetry flags
    mistral_configured = any(r.get("mistral_configured", False) for r in per_image_results)
    mistral_attempted = any(r.get("mistral_attempted", False) for r in per_image_results)
    mistral_success = any(r.get("mistral_success", False) for r in per_image_results)

    gemini_configured = any(r.get("gemini_configured", False) for r in per_image_results)
    gemini_attempted = any(r.get("gemini_attempted", False) for r in per_image_results)
    gemini_success = any(r.get("gemini_success", False) for r in per_image_results)

    local_attempted = any(r.get("local_attempted", False) for r in per_image_results)
    fallback_used = any(r.get("fallback_used", False) for r in per_image_results)

    fallback_notes = [r.get("fallback_note", "") for r in per_image_results if r.get("fallback_note")]
    combined_fallback_note = "; ".join(list(dict.fromkeys(fallback_notes))) if fallback_notes else ""

    raw_texts = [f"=== {r.get('image_label', 'Image')} ({r.get('image_filename', '')}) ===\n" + r.get("raw_text", "") for r in successful_results if r.get("raw_text")]
    combined_raw_text = "\n\n".join(raw_texts)

    return {
        "success": True,
        "engine": overall_engine,
        "confidence": avg_confidence,
        "processing_time_sec": total_time_sec,
        "fallback_used": fallback_used,
        "fallback_note": combined_fallback_note,
        "mistral_configured": mistral_configured,
        "mistral_attempted": mistral_attempted,
        "mistral_success": mistral_success,
        "gemini_configured": gemini_configured,
        "gemini_attempted": gemini_attempted,
        "gemini_success": gemini_success,
        "local_attempted": local_attempted,
        "images_uploaded": images_uploaded,
        "images_processed": images_processed,
        "images_failed": images_failed,
        "failed_image_labels": failed_labels,
        "partial_success_note": partial_success_note,
        "fields": merged_fields,
        "fields_detail": merged_fields_detail,
        "raw_text": combined_raw_text,
        "per_image_results": per_image_results
    }


def run_multi_image_extraction(
    image_paths: List[str],
    engine_mode: str = "auto",
    gemini_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Processes all uploaded product images and merges results into a single product record.
    """
    t0 = time.time()
    per_image_results = []

    if len(image_paths) == 1:
        path = image_paths[0]
        filename = os.path.basename(path)
        try:
            single_res = _run_single_hybrid_extraction(
                image_path=path,
                engine_mode=engine_mode,
                gemini_api_key=gemini_api_key,
                mistral_api_key=mistral_api_key
            )
            single_res["image_index"] = 1
            single_res["image_label"] = "Image 1"
            single_res["image_filename"] = filename
            single_res["images_uploaded"] = 1
            single_res["images_processed"] = 1 if single_res.get("success") else 0
            single_res["images_failed"] = 0 if single_res.get("success") else 1
            for fd in single_res.get("fields_detail", {}).values():
                if isinstance(fd, dict):
                    if "source_images" not in fd:
                        fd["source_images"] = ["Image 1"]
                    if "source_image" not in fd:
                        fd["source_image"] = "Image 1"
                    if "has_conflict" not in fd:
                        fd["has_conflict"] = False
                    if "conflict_detail" not in fd:
                        fd["conflict_detail"] = None
            per_image_info = dict(single_res)
            single_res["per_image_results"] = [per_image_info]
            return single_res
        except Exception as ex:
            return {
                "success": False,
                "engine": "Failed",
                "confidence": 0.0,
                "processing_time_sec": round(time.time() - t0, 2),
                "images_uploaded": 1,
                "images_processed": 0,
                "images_failed": 1,
                "failed_image_labels": ["Image 1"],
                "partial_success_note": "Image failed extraction.",
                "fallback_used": False,
                "fallback_note": "",
                "error": str(ex),
                "fields": {f: "" for f in MANDATORY_FIELDS},
                "fields_detail": {},
                "raw_text": "",
                "per_image_results": []
            }

    for idx, path in enumerate(image_paths):
        label = f"Image {idx + 1}"
        filename = os.path.basename(path)
        try:
            res = _run_single_hybrid_extraction(
                image_path=path,
                engine_mode=engine_mode,
                gemini_api_key=gemini_api_key,
                mistral_api_key=mistral_api_key
            )
            res["image_index"] = idx + 1
            res["image_label"] = label
            res["image_filename"] = filename
            per_image_results.append(res)
        except Exception as ex:
            per_image_results.append({
                "image_index": idx + 1,
                "image_label": label,
                "image_filename": filename,
                "success": False,
                "error": str(ex),
                "engine": "Failed",
                "confidence": None,
                "fields": {},
                "fields_detail": {}
            })

    total_time = round(time.time() - t0, 2)
    return merge_multi_image_records(per_image_results, total_time_sec=total_time)


def run_hybrid_extraction(
    image_path: Union[str, List[str]],
    engine_mode: str = "auto",
    gemini_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Entrypoint for packaging extraction. Supports single image path or list of image paths.
    Delegates to run_multi_image_extraction for unified multi-view merge and provenance.
    """
    if isinstance(image_path, (list, tuple)):
        paths = [str(p) for p in image_path if p]
    elif image_path:
        paths = [str(image_path)]
    else:
        paths = []

    if not paths:
        return {
            "success": False,
            "error": "No product images provided.",
            "engine": "None",
            "confidence": 0.0,
            "processing_time_sec": 0.0,
            "fields": {f: "" for f in MANDATORY_FIELDS},
            "fields_detail": {},
            "images_uploaded": 0,
            "images_processed": 0,
            "images_failed": 0
        }

    return run_multi_image_extraction(
        image_paths=paths,
        engine_mode=engine_mode,
        gemini_api_key=gemini_api_key,
        mistral_api_key=mistral_api_key
    )


def normalize_field_for_comparison(field_name: str, val: str) -> str:
    """Normalizes field text to compare semantic equality rather than superficial formatting."""
    if not val:
        return ""
    text = str(val).strip()

    if field_name == "MRP":
        m = re.search(r"(\d+(?:\.\d{1,2})?)", text.replace(",", ""))
        if m:
            try:
                return f"{float(m.group(1)):.2f}"
            except ValueError:
                pass
        return text.lower()

    if field_name == "Net Quantity":
        m = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", text)
        if m:
            num = m.group(1)
            unit = m.group(2).lower()
            return f"{num}_{unit}"
        return re.sub(r"\s+", "", text).lower()

    if field_name == "Manufacturing Date":
        dval = validate_date(text)
        if dval.get("parsed_month") and dval.get("parsed_year"):
            return f"{dval['parsed_month']:02d}/{dval['parsed_year']}"
        m = re.search(r"(\d{1,2})[/\-.](\d{1,2})?[/\-.]?(\d{2,4})", text)
        if m:
            parts = [p for p in m.groups() if p]
            return "/".join(parts)
        return re.sub(r"\s+", "", text).lower()

    if field_name == "Batch Number":
        return re.sub(r"[^a-zA-Z0-9]", "", text).upper()

    clean = re.sub(r"[^\w\s]", " ", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def detect_extraction_conflicts(
    gemini_fields: Dict[str, str],
    local_fields: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Compares extractions from Gemini Vision AI and Local OCR.
    Identifies semantic divergence on packaging declarations.
    """
    conflicts = {}
    for field_name in gemini_fields.keys():
        val_g = (gemini_fields.get(field_name) or "").strip()
        val_l = (local_fields.get(field_name) or "").strip()

        # If either engine did not detect the field, there is no direct conflicting assertion
        if not val_g or not val_l:
            continue

        norm_g = normalize_field_for_comparison(field_name, val_g)
        norm_l = normalize_field_for_comparison(field_name, val_l)

        if norm_g == norm_l:
            continue

        # Substring tolerance for descriptive entity fields
        if field_name in ["Product Name", "Manufacturer", "Consumer Care"]:
            if norm_g in norm_l or norm_l in norm_g:
                continue

        conflicts[field_name] = {
            "has_conflict": True,
            "gemini_value": val_g,
            "local_value": val_l,
            "conflict_detail": f"Gemini detected '{val_g}' vs Local OCR detected '{val_l}'"
        }

    return conflicts


def _build_local_fields_detail(fields: Dict[str, str], overall_conf: float) -> Dict[str, Any]:
    """Constructs explainability metadata for locally extracted OCR fields."""
    threshold = get_extraction_confidence_threshold()
    details = {}
    for name, val in fields.items():
        val_str = (val or "").strip()
        has_val = bool(val_str)
        conf = round(overall_conf, 1) if (has_val and overall_conf > 0) else None
        details[name] = {
            "value": val_str,
            "evidence": f"Detected in OCR stream: '{val_str[:40]}...'" if len(val_str) > 40 else f"Detected in OCR stream: '{val_str}'" if has_val else "No reliable declaration detected in OCR stream",
            "confidence": conf,
            "requires_review": not has_val or (conf is not None and conf < threshold),
            "source": "Local OCR (Tesseract)",
            "engine": "Local OCR",
            "has_conflict": False,
            "conflict_detail": None
        }
    return details
