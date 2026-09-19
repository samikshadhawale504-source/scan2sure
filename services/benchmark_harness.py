"""
Multi-Image Benchmark Evaluation Harness for SIH26034.
Evaluates packaged commodity images through the EXACT production Legal Metrology pipeline:
  Image -> Preprocessing -> Hybrid Extraction (Gemini/Local OCR) -> Field Validation -> Compliance Rules

Key Design Principles:
1. Reusable & Image-Independent: Zero product-specific hardcoding or sample-specific rules.
2. Production Pipeline Fidelity: Calls the exact same functions as the web application.
3. Resilient Batch Processing: Error-isolated execution ensures corrupt images do not halt the batch.
4. Ground Truth Integrity: Explicitly labels missing annotations as "ground truth unavailable".
   When annotations are provided, computes field-level accuracy and compliance concordance.
5. Multi-Format Reporting: Exports detailed JSON, tabular CSV, and human-readable text summaries.
"""

import os
import sys
import time
import json
import csv
import argparse
from typing import Dict, Any, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.hybrid_extractor import run_hybrid_extraction, normalize_field_for_comparison
from validation.field_validator import validate_all_fields
from compliance_rules import evaluate_compliance

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def process_single_image(
    image_path: str,
    engine_mode: str = "auto",
    gemini_api_key: Optional[str] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    commodity_category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a single image through the complete production compliance pipeline.
    Captures all statutory declarations, validation issues, confidence ratings, and conflicts.
    """
    t0 = time.time()
    filename = os.path.basename(image_path)
    abs_path = os.path.abspath(image_path)

    record: Dict[str, Any] = {
        "filename": filename,
        "image_path": abs_path,
        "image_processing_success": False,
        "error_message": None,
        "engine_used": "Unknown",
        "fallback_used": False,
        "fallback_note": "",
        "processing_time_sec": 0.0,
        "confidence_values": {},
        "overall_confidence": None,
        "local_ocr_extraction": {},
        "gemini_extraction": {},
        "hybrid_final_fields": {},
        "field_validation_results": {},
        "final_compliance_status": "UNKNOWN",
        "compliance_score": 0.0,
        "failed_rules": [],
        "not_detected_rules": [],
        "requires_review_rules": [],
        "not_applicable_rules": [],
        "compliant_rules": [],
        "requires_review_state": False,
        "conflicts": {},
        "rule_status_counts": {
            "COMPLIANT": 0,
            "NON_COMPLIANT": 0,
            "REQUIRES_REVIEW": 0,
            "NOT_DETECTED": 0,
            "NOT_APPLICABLE": 0
        },
        "ground_truth_status": "ground truth unavailable",
        "accuracy_metrics": None
    }

    if not os.path.exists(abs_path):
        record["error_message"] = f"Image file not found: {abs_path}"
        record["processing_time_sec"] = round(time.time() - t0, 2)
        return record

    try:
        # STEP 1: Hybrid Extraction (Gemini Multimodal AI with Local OCR fallback)
        extract_res = run_hybrid_extraction(
            image_path=abs_path,
            engine_mode=engine_mode,
            gemini_api_key=gemini_api_key
        )

        record["image_processing_success"] = bool(extract_res.get("success"))
        record["error_message"] = extract_res.get("error")
        record["engine_used"] = extract_res.get("engine", "")
        record["fallback_used"] = extract_res.get("fallback_used", False)
        record["fallback_note"] = extract_res.get("fallback_note", "")
        record["overall_confidence"] = extract_res.get("confidence")

        fields = extract_res.get("fields", {})
        fields_detail = extract_res.get("fields_detail", {})
        record["hybrid_final_fields"] = fields

        # Split engine components for provenance comparison
        record["local_ocr_extraction"] = extract_res.get("local_fields") or (
            fields if "Local OCR" in record["engine_used"] else {}
        )
        record["gemini_extraction"] = extract_res.get("gemini_fields") or (
            fields if "Gemini" in record["engine_used"] else {}
        )

        # Field-level confidence
        record["confidence_values"] = {
            k: v.get("confidence") for k, v in fields_detail.items() if isinstance(v, dict)
        }

        # Conflict extraction
        conflicts = {}
        for fname, fdetail in fields_detail.items():
            if isinstance(fdetail, dict) and fdetail.get("has_conflict"):
                conflicts[fname] = fdetail.get("conflict_detail") or "Discrepancy detected"
        record["conflicts"] = conflicts

        # STEP 2: Field Validation Layer
        val_res = validate_all_fields(fields, fields_detail)
        record["field_validation_results"] = {
            k: {
                "valid": v.get("valid"),
                "issues": v.get("issues", []),
                "warnings": v.get("warnings", [])
            }
            for k, v in val_res.items()
            if isinstance(v, dict) and "valid" in v
        }

        # STEP 3: Statutory Compliance Engine
        compliance_res = evaluate_compliance(
            fields,
            fields_detail=fields_detail,
            commodity_category=commodity_category
        )

        record["final_compliance_status"] = compliance_res.get("status", "UNKNOWN")
        record["compliance_score"] = compliance_res.get("score", 0.0)

        # Audit breakdown
        audit_table = compliance_res.get("audit_table", [])
        status_counts = {"COMPLIANT": 0, "NON_COMPLIANT": 0, "REQUIRES_REVIEW": 0, "NOT_DETECTED": 0, "NOT_APPLICABLE": 0}

        for row in audit_table:
            st = row.get("status", "")
            if st in status_counts:
                status_counts[st] += 1

            entry_summary = {
                "rule_id": row.get("rule_id"),
                "rule": row.get("rule"),
                "requirement": row.get("requirement"),
                "status": st,
                "reason": row.get("reason"),
                "action": row.get("recommended_action")
            }

            if st == "NON_COMPLIANT":
                record["failed_rules"].append(entry_summary)
            elif st == "NOT_DETECTED":
                record["not_detected_rules"].append(entry_summary)
            elif st == "REQUIRES_REVIEW":
                record["requires_review_rules"].append(entry_summary)
            elif st == "NOT_APPLICABLE":
                record["not_applicable_rules"].append(entry_summary)
            elif st == "COMPLIANT":
                record["compliant_rules"].append(entry_summary)

        record["rule_status_counts"] = status_counts
        record["requires_review_state"] = (
            record["final_compliance_status"] == "REQUIRES_REVIEW"
            or len(record["requires_review_rules"]) > 0
            or len(record["conflicts"]) > 0
        )

        # STEP 4: Ground Truth Evaluation (if supplied)
        if ground_truth and isinstance(ground_truth, dict):
            record["ground_truth_status"] = "evaluated"
            gt_fields = ground_truth.get("fields", ground_truth)
            matches = 0
            total_eval = 0
            field_accuracies = {}

            for target_field, expected_val in gt_fields.items():
                if target_field in ["expected_status", "commodity_category"]:
                    continue
                total_eval += 1
                extracted_val = fields.get(target_field, "")

                norm_ext = normalize_field_for_comparison(target_field, str(extracted_val))
                norm_exp = normalize_field_for_comparison(target_field, str(expected_val))

                is_match = (norm_ext == norm_exp) if (norm_ext and norm_exp) else (not norm_ext and not norm_exp)
                if is_match:
                    matches += 1

                field_accuracies[target_field] = {
                    "expected": expected_val,
                    "extracted": extracted_val,
                    "matched": is_match
                }

            expected_status = ground_truth.get("expected_status")
            status_matched = (expected_status == record["final_compliance_status"]) if expected_status else None

            field_accuracy_pct = round((matches / total_eval * 100.0), 1) if total_eval > 0 else 0.0

            record["accuracy_metrics"] = {
                "total_fields_evaluated": total_eval,
                "fields_matched": matches,
                "field_accuracy_pct": field_accuracy_pct,
                "field_breakdown": field_accuracies,
                "expected_status": expected_status,
                "status_matched": status_matched
            }
        else:
            record["ground_truth_status"] = "ground truth unavailable"
            record["accuracy_metrics"] = None

    except Exception as e:
        record["image_processing_success"] = False
        record["error_message"] = f"Pipeline exception: {str(e)}"

    record["processing_time_sec"] = round(time.time() - t0, 2)
    return record


def run_benchmark(
    image_folder: str,
    output_dir: Optional[str] = None,
    engine_mode: str = "auto",
    gemini_api_key: Optional[str] = None,
    ground_truth_file: Optional[str] = None,
    target_filenames: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes a multi-image benchmark batch over a folder of packaged commodity images.
    Returns comprehensive metrics and writes machine-readable (JSON/CSV) and text summaries.
    """
    start_time = time.time()
    folder_abs = os.path.abspath(image_folder)

    if not os.path.exists(folder_abs):
        raise FileNotFoundError(f"Benchmark folder not found: {folder_abs}")

    # Discover supported image files
    all_files = sorted(os.listdir(folder_abs))
    image_files = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if target_filenames:
        target_set = set(target_filenames)
        image_files = [f for f in image_files if f in target_set]

    # Load Ground Truth annotations if provided
    gt_map: Dict[str, Any] = {}
    if ground_truth_file and os.path.exists(ground_truth_file):
        try:
            with open(ground_truth_file, "r", encoding="utf-8") as f:
                gt_map = json.load(f)
        except Exception as e:
            print(f"[BENCHMARK WARNING] Failed to load ground truth file ({e}). Marking ground truth unavailable.")
            gt_map = {}

    results = []
    status_summary = {
        "COMPLIANT": 0,
        "NON-COMPLIANT": 0,
        "REQUIRES_REVIEW": 0,
        "ERROR": 0
    }
    total_conflicts = 0
    total_time = 0.0
    valid_confidences = []

    gt_evaluated_count = 0
    gt_total_accuracy_sum = 0.0

    print(f"\n{'=' * 75}")
    print(f"SIH26034 LEGAL METROLOGY MULTI-IMAGE BENCHMARK EVALUATION")
    print(f"Directory   : {folder_abs}")
    print(f"Images Found: {len(image_files)}")
    print(f"Engine Mode : {engine_mode}")
    print(f"Ground Truth: {'Provided (' + ground_truth_file + ')' if gt_map else 'Unavailable'}")
    print(f"{'=' * 75}\n")

    for idx, fname in enumerate(image_files, 1):
        fpath = os.path.join(folder_abs, fname)
        gt_item = gt_map.get(fname)
        cat = gt_item.get("commodity_category") if isinstance(gt_item, dict) else None

        print(f"[{idx:02d}/{len(image_files):02d}] Processing: {fname} ...", end="", flush=True)

        record = process_single_image(
            image_path=fpath,
            engine_mode=engine_mode,
            gemini_api_key=gemini_api_key,
            ground_truth=gt_item,
            commodity_category=cat
        )

        results.append(record)
        total_time += record["processing_time_sec"]

        if record["overall_confidence"] is not None:
            valid_confidences.append(record["overall_confidence"])

        total_conflicts += len(record.get("conflicts", {}))

        st = record["final_compliance_status"]
        if not record["image_processing_success"]:
            status_summary["ERROR"] += 1
        elif st in status_summary:
            status_summary[st] += 1
        else:
            status_summary[st] = status_summary.get(st, 0) + 1

        acc_str = ""
        if record["accuracy_metrics"]:
            gt_evaluated_count += 1
            pct = record["accuracy_metrics"]["field_accuracy_pct"]
            gt_total_accuracy_sum += pct
            acc_str = f" | Accuracy: {pct}%"

        print(f" DONE ({record['processing_time_sec']}s) -> [{st}]{acc_str}")

    elapsed_total = round(time.time() - start_time, 2)
    avg_time = round(total_time / len(image_files), 2) if image_files else 0.0
    avg_conf = round(sum(valid_confidences) / len(valid_confidences), 1) if valid_confidences else None
    avg_accuracy = round(gt_total_accuracy_sum / gt_evaluated_count, 1) if gt_evaluated_count > 0 else None

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_folder": folder_abs,
        "total_images": len(image_files),
        "successful_extractions": sum(1 for r in results if r["image_processing_success"]),
        "failed_extractions": sum(1 for r in results if not r["image_processing_success"]),
        "status_distribution": status_summary,
        "total_conflicts_detected": total_conflicts,
        "average_processing_time_sec": avg_time,
        "total_elapsed_sec": elapsed_total,
        "average_confidence_pct": avg_conf,
        "ground_truth_evaluated_count": gt_evaluated_count,
        "ground_truth_accuracy_pct": avg_accuracy if avg_accuracy is not None else "ground truth unavailable"
    }

    full_report = {
        "benchmark_summary": benchmark_summary,
        "image_results": results
    }

    # Destination directory
    out_dir = os.path.abspath(output_dir or folder_abs)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Export JSON
    json_path = os.path.join(out_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # 2. Export CSV
    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    csv_headers = [
        "filename", "success", "engine", "time_sec", "confidence",
        "compliance_status", "compliance_score", "failed_rules_count",
        "not_detected_count", "requires_review_count", "conflicts_count",
        "product_name", "net_quantity", "mrp", "manufacturing_date", "batch_number",
        "ground_truth_status", "ground_truth_accuracy_pct"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for r in results:
            fields = r.get("hybrid_final_fields", {})
            acc = r["accuracy_metrics"]["field_accuracy_pct"] if r["accuracy_metrics"] else "N/A"
            writer.writerow({
                "filename": r["filename"],
                "success": r["image_processing_success"],
                "engine": r["engine_used"],
                "time_sec": r["processing_time_sec"],
                "confidence": r["overall_confidence"],
                "compliance_status": r["final_compliance_status"],
                "compliance_score": r["compliance_score"],
                "failed_rules_count": len(r["failed_rules"]),
                "not_detected_count": len(r["not_detected_rules"]),
                "requires_review_count": len(r["requires_review_rules"]),
                "conflicts_count": len(r.get("conflicts", {})),
                "product_name": fields.get("Product Name", ""),
                "net_quantity": fields.get("Net Quantity", ""),
                "mrp": fields.get("MRP", ""),
                "manufacturing_date": fields.get("Manufacturing Date", ""),
                "batch_number": fields.get("Batch Number", ""),
                "ground_truth_status": r["ground_truth_status"],
                "ground_truth_accuracy_pct": acc
            })

    # 3. Export Human-Readable Summary Text
    summary_text = _format_text_summary(benchmark_summary, results)
    txt_path = os.path.join(out_dir, "benchmark_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\n[BENCHMARK EXPORTS SAVED]")
    print(f" - JSON Report : {json_path}")
    print(f" - CSV Table   : {csv_path}")
    print(f" - Text Summary: {txt_path}\n")

    return full_report


def _format_text_summary(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    """Generates an official Legal Metrology benchmark audit scorecard banner."""
    lines = []
    lines.append("\n" + "=" * 75)
    lines.append("        DIRECTORATE OF LEGAL METROLOGY - BENCHMARK EVALUATION REPORT")
    lines.append("=" * 75)
    lines.append(f"Timestamp            : {summary['timestamp']}")
    lines.append(f"Source Folder        : {summary['benchmark_folder']}")
    lines.append(f"Total Images Scanned : {summary['total_images']}")
    lines.append(f"Successful Parses    : {summary['successful_extractions']}")
    lines.append(f"Failed / Corrupted   : {summary['failed_extractions']}")
    lines.append(f"Average Latency      : {summary['average_processing_time_sec']} sec / image")
    lines.append(f"Total Elapsed Time   : {summary['total_elapsed_sec']} sec")
    lines.append(f"Average Confidence   : {summary['average_confidence_pct'] or 'Unmeasured'}%")
    lines.append(f"Total Discrepancies  : {summary['total_conflicts_detected']} cross-engine conflicts")

    gt_acc = summary["ground_truth_accuracy_pct"]
    if isinstance(gt_acc, (int, float)):
        lines.append(f"Ground Truth Accuracy: {gt_acc}% (Evaluated on {summary['ground_truth_evaluated_count']} labeled images)")
    else:
        lines.append(f"Ground Truth Accuracy: Ground Truth Unavailable (Requires labeled ground_truth.json)")

    lines.append("-" * 75)
    lines.append("COMPLIANCE STATUS DISTRIBUTION:")
    for status, count in summary["status_distribution"].items():
        pct = round((count / summary['total_images'] * 100.0), 1) if summary['total_images'] > 0 else 0
        lines.append(f"  * {status:<18}: {count:>3} packages ({pct}%)")

    lines.append("-" * 75)
    lines.append(f"{'Filename':<24} | {'Status':<15} | {'Score':<6} | {'Time':<6} | {'Conflicts':<9} | {'Ground Truth'}")
    lines.append("-" * 75)
    for r in results:
        gt_disp = (
            f"{r['accuracy_metrics']['field_accuracy_pct']}%"
            if r.get("accuracy_metrics")
            else "Unavailable"
        )
        lines.append(
            f"{r['filename'][:23]:<24} | "
            f"{r['final_compliance_status']:<15} | "
            f"{r['compliance_score']:>5.1f}% | "
            f"{r['processing_time_sec']:>4.1f}s | "
            f"{len(r.get('conflicts', {})):>9} | "
            f"{gt_disp}"
        )
    lines.append("=" * 75 + "\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SIH26034 Legal Metrology Multi-Image Benchmark Harness")
    parser.add_argument("--folder", "-f", default=".", help="Directory containing packaging images")
    parser.add_argument("--output", "-o", default=None, help="Directory to save JSON/CSV benchmark exports")
    parser.add_argument("--engine", "-e", default="auto", choices=["auto", "gemini", "local", "dual"], help="Extraction engine mode")
    parser.add_argument("--key", "-k", default=None, help="Optional Gemini API key")
    parser.add_argument("--ground-truth", "-gt", default=None, help="Optional path to ground_truth.json annotations")
    parser.add_argument("--images", "-i", nargs="*", default=None, help="Optional specific list of image filenames")
    args = parser.parse_args()

    run_benchmark(
        image_folder=args.folder,
        output_dir=args.output,
        engine_mode=args.engine,
        gemini_api_key=args.key,
        ground_truth_file=args.ground_truth,
        target_filenames=args.images
    )


if __name__ == "__main__":
    main()
