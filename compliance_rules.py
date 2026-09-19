"""
Legal Metrology (Packaged Commodities) Rules, 2011 - Comprehensive Compliance Verification Engine.
Implements statutory compliance checks under Rule 6 and recent amendments (including USP under Rule 6(1)(m)).
Produces explainable, rule-by-rule inspector audit logs, category scores, and recommended statutory actions.

Stage 1 Updates:
- Differentiates NOT_DETECTED from NON_COMPLIANT.
- Supports commodity-category conditioning (Food/Perishable vs Non-Food under Rule 6(1)(g)).
- Implements Rule 6(1)(m) USP exemption for 1kg/1L/1m/1unit packages where MRP = USP.
- Implements Rule 26(a) small-package exemption for <= 10g / <= 10ml.
- Validates metric unit case sensitivity (Rule 13) and prohibited expressions (Rule 11(2)).
"""

import re
from typing import Dict, Any, List, Optional
from config import get_extraction_confidence_threshold
from validation.field_validator import (
    validate_mrp,
    validate_net_quantity,
    calculate_and_validate_usp,
    validate_date,
    validate_consumer_care,
    validate_manufacturer_address,
    PERMITTED_METRIC_UNITS,
    PROHIBITED_UNIT_ABBREVIATIONS
)


def evaluate_product_name(product_name: str, evidence: Optional[str] = None) -> Dict[str, Any]:
    """Rule 6(1)(a): Common or generic name of commodity on the Principal Display Panel."""
    name = (product_name or "").strip()
    ev = evidence or name

    if not name:
        return {
            "rule_id": "PCR-01",
            "rule": "Rule 6(1)(a)",
            "requirement": "Generic / Common Commodity Name",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(a)",
            "category": "Mandatory PDP",
            "extracted_evidence": "Not detected in scanned label",
            "status": "NOT_DETECTED",
            "confidence": 75.0,
            "score": 0,
            "weight": 15,
            "reason": "Generic commodity name was not detected in the provided image/input. Physical packaging inspection required to confirm presence on Principal Display Panel.",
            "recommended_action": "Verify physical package PDP or upload complete multi-panel image."
        }

    if len(name) < 3:
        return {
            "rule_id": "PCR-01",
            "rule": "Rule 6(1)(a)",
            "requirement": "Generic / Common Commodity Name",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(a)",
            "category": "Mandatory PDP",
            "extracted_evidence": ev,
            "status": "REQUIRES_REVIEW",
            "confidence": 70.0,
            "score": 8,
            "weight": 15,
            "reason": f"Product name '{name}' is unusually short; verify that generic commodity descriptor is fully stated.",
            "recommended_action": "Ensure that the common noun or descriptor representing the nature of the product is clearly printed."
        }

    return {
        "rule_id": "PCR-01",
        "rule": "Rule 6(1)(a)",
        "requirement": "Generic / Common Commodity Name",
        "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(a)",
        "category": "Mandatory PDP",
        "extracted_evidence": ev,
        "status": "COMPLIANT",
        "confidence": 95.0,
        "score": 15,
        "weight": 15,
        "reason": f"Compliant generic/common name declared on package: '{name}'.",
        "recommended_action": "None. Requirement satisfied."
    }


def evaluate_manufacturer_address(manufacturer: str, evidence: Optional[str] = None) -> Dict[str, Any]:
    """Rule 6(1)(b): Name and complete postal address of manufacturer, packer, or importer."""
    mfg = (manufacturer or "").strip()
    ev = evidence or mfg

    if not mfg:
        return {
            "rule_id": "PCR-02",
            "rule": "Rule 6(1)(b)",
            "requirement": "Manufacturer / Packer Identity & Complete Postal Address",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(b)",
            "category": "Manufacturer & Address",
            "extracted_evidence": "Not detected in scanned label",
            "status": "NOT_DETECTED",
            "confidence": 75.0,
            "score": 0,
            "weight": 15,
            "reason": "Manufacturer / packer / importer identity and address were not detected in the uploaded image/input.",
            "recommended_action": "Inspect packaging to confirm presence of manufacturer/packer name and complete postal address."
        }

    val = validate_manufacturer_address(mfg)

    if not val["valid"]:
        return {
            "rule_id": "PCR-02",
            "rule": "Rule 6(1)(b)",
            "requirement": "Manufacturer / Packer Identity & Complete Postal Address",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(b)",
            "category": "Manufacturer & Address",
            "extracted_evidence": ev,
            "status": "NON_COMPLIANT",
            "confidence": 85.0,
            "score": 0,
            "weight": 15,
            "reason": "; ".join(val["issues"]),
            "recommended_action": "Ensure legal entity name is stated without ambiguity."
        }

    if val["has_pincode"] or (len(mfg) > 30 and val["has_locality"]):
        return {
            "rule_id": "PCR-02",
            "rule": "Rule 6(1)(b)",
            "requirement": "Manufacturer / Packer Identity & Complete Postal Address",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(b)",
            "category": "Manufacturer & Address",
            "extracted_evidence": ev,
            "status": "COMPLIANT",
            "confidence": 95.0,
            "score": 15,
            "weight": 15,
            "reason": f"Complete postal address with location indicators verified: '{mfg}'.",
            "recommended_action": "None. Requirement satisfied."
        }
    else:
        return {
            "rule_id": "PCR-02",
            "rule": "Rule 6(1)(b)",
            "requirement": "Manufacturer / Packer Identity & Complete Postal Address",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(b)",
            "category": "Manufacturer & Address",
            "extracted_evidence": ev,
            "status": "REQUIRES_REVIEW",
            "confidence": 75.0,
            "score": 10,
            "weight": 15,
            "reason": f"Manufacturer name present ('{mfg}'), but complete postal address/PIN code could not be confirmed.",
            "recommended_action": "Verify physical package to confirm whether 6-digit postal PIN code is stamped or printed."
        }


def evaluate_net_quantity(
    net_quantity: str,
    evidence: Optional[str] = None,
    commodity_name: Optional[str] = None,
    commodity_category: Optional[str] = None,
    confidence: Optional[float] = None,
    extraction_source: Optional[str] = None,
    has_conflict: bool = False,
    conflict_detail: Optional[str] = None
) -> Dict[str, Any]:
    """Rule 6(1)(c): Net quantity in legal standard metric units."""
    qty = (net_quantity or "").strip()
    ev = evidence or qty

    if not qty:
        res = {
            "rule_id": "PCR-03",
            "rule": "Rule 6(1)(c)",
            "requirement": "Net Quantity & Metric Units",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(c) & First Schedule",
            "category": "Quantity & Metric Units",
            "extracted_evidence": "Not detected in scanned label",
            "status": "NOT_DETECTED",
            "confidence": None,
            "score": 0,
            "weight": 15,
            "reason": "Net quantity declaration was not detected in the uploaded image/input.",
            "recommended_action": "Verify physical package PDP to confirm presence of net quantity in standard metric units."
        }
    else:
        val = validate_net_quantity(qty, commodity_name=commodity_name, commodity_category=commodity_category)

        # 1. Affirmative Legal Violations -> NON_COMPLIANT
        if val.get("is_illegal_unit") or val.get("has_prohibited_qualifier") or val.get("is_vulgar_fraction") or (val.get("numeric_value") is not None and val["numeric_value"] <= 0):
            res = {
                "rule_id": "PCR-03",
                "rule": "Rule 6(1)(c)",
                "requirement": "Net Quantity & Metric Units",
                "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(c), Rule 11(2) & Rule 13",
                "category": "Quantity & Metric Units",
                "extracted_evidence": ev,
                "status": "NON_COMPLIANT",
                "confidence": 95.0,
                "score": 0,
                "weight": 15,
                "reason": "; ".join(val["issues"]),
                "recommended_action": (
                    f"Replace non-compliant expression/symbol with legal statutory symbol '{val.get('normalized_unit', 'standard unit')}' "
                    "and remove any prohibited qualifiers (Rule 11(2))."
                )
            }
        # 2. Rule 13 Formatting Advisory (e.g. upper-case symbol G / KG) -> REQUIRES_REVIEW (not NON_COMPLIANT)
        elif val.get("is_case_violation"):
            res = {
                "rule_id": "PCR-03",
                "rule": "Rule 6(1)(c)",
                "requirement": "Net Quantity & Metric Units",
                "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(c) & Rule 13",
                "category": "Quantity & Metric Units",
                "extracted_evidence": ev,
                "status": "REQUIRES_REVIEW",
                "confidence": 80.0,
                "score": 10,
                "weight": 15,
                "reason": (
                    f"Formatting Advisory under Rule 13: Metric symbol for mass/length should be lower-case ('g', 'kg'). "
                    f"Detected upper-case '{val['unit']}'. Verify whether packaging typography is all-caps or OCR capitalization."
                ),
                "recommended_action": f"Inspect physical packaging to ensure standard lower-case symbol '{val['normalized_unit']}' is used on label artwork."
            }
        # 3. Valid Metric Quantity -> COMPLIANT
        elif val["valid"]:
            reason = f"Net quantity complies with metric standards: {val['numeric_value']} {val['normalized_unit']}."
            if val.get("is_permitted_climatic_qualifier"):
                reason += " Permitted under Proviso to Rule 11(2) for commodities susceptible to climatic variation."
            if val.get("has_spacing_advisory"):
                reason += " (Advisory: standard space recommended between numerical value and symbol)."

            res = {
                "rule_id": "PCR-03",
                "rule": "Rule 6(1)(c)",
                "requirement": "Net Quantity & Metric Units",
                "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(c)",
                "category": "Quantity & Metric Units",
                "extracted_evidence": ev,
                "status": "COMPLIANT",
                "confidence": 95.0,
                "score": 15,
                "weight": 15,
                "reason": reason,
                "recommended_action": "None. Requirement satisfied."
            }
        else:
            res = {
                "rule_id": "PCR-03",
                "rule": "Rule 6(1)(c)",
                "requirement": "Net Quantity & Metric Units",
                "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(c)",
                "category": "Quantity & Metric Units",
                "extracted_evidence": ev,
                "status": "REQUIRES_REVIEW",
                "confidence": 65.0,
                "score": 8,
                "weight": 15,
                "reason": f"Unable to verify standard metric formatting for quantity '{qty}'.",
                "recommended_action": "Verify that numeric magnitude and metric symbol conform to First Schedule specifications."
            }

    # Threshold & Conflict adjustments
    threshold = get_extraction_confidence_threshold()
    if has_conflict:
        res["status"] = "REQUIRES_REVIEW"
        res["score"] = 8
        res["reason"] = f"Extraction discrepancy detected: {conflict_detail}. Physical package inspection required."
        res["recommended_action"] = "Inspect physical packaging to resolve discrepancy between extraction engines."
    elif confidence is not None and confidence < threshold:
        if res["status"] == "COMPLIANT":
            res["status"] = "REQUIRES_REVIEW"
            res["score"] = 10
            res["reason"] = f"Low extraction confidence ({confidence:.1f}% < {threshold:.1f}% threshold). Declaration format appears valid ('{qty}'), but requires physical verification."
            res["recommended_action"] = "Inspect physical packaging to verify accuracy of low-confidence reading."
        elif res["status"] == "NON_COMPLIANT":
            res["status"] = "REQUIRES_REVIEW"
            res["score"] = 5
            res["reason"] = f"Potential non-compliance detected ('{qty}'), but extraction confidence ({confidence:.1f}%) is below threshold ({threshold:.1f}%). Must be verified before enforcement."
            res["recommended_action"] = "Inspect physical package PDP to confirm whether suspected defect exists."

    res["extracted_value"] = qty
    res["confidence"] = confidence if qty else None
    res["extraction_source"] = extraction_source or ("Manual Input" if confidence is None else "Local OCR (Tesseract)")
    res["has_conflict"] = has_conflict
    res["conflict_detail"] = conflict_detail
    return res


def evaluate_mrp(mrp_str: str, evidence: Optional[str] = None) -> Dict[str, Any]:
    """Rule 6(1)(da): Retail Sale Price (MRP) including 'inclusive of all taxes'."""
    raw = (mrp_str or "").strip()
    ev = evidence or raw

    if not raw:
        return {
            "rule_id": "PCR-04",
            "rule": "Rule 6(1)(da)",
            "requirement": "Retail Sale Price (MRP & Taxes)",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(da)",
            "category": "Price & USP",
            "extracted_evidence": "Not detected in scanned label",
            "status": "NOT_DETECTED",
            "confidence": 75.0,
            "score": 0,
            "weight": 15,
            "reason": "Retail Sale Price (MRP) was not detected in the uploaded image/input.",
            "recommended_action": "Verify packaging to confirm presence of Maximum Retail Price (MRP) declaration."
        }

    val = validate_mrp(raw)

    if val["numeric_price"] is None:
        return {
            "rule_id": "PCR-04",
            "rule": "Rule 6(1)(da)",
            "requirement": "Retail Sale Price (MRP & Taxes)",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(da)",
            "category": "Price & USP",
            "extracted_evidence": ev,
            "status": "NOT_DETECTED",
            "confidence": 75.0,
            "score": 0,
            "weight": 15,
            "reason": f"No valid numeric price identified in text: '{raw}'.",
            "recommended_action": "Check whether price is clearly printed or stamped on package."
        }

    if val["numeric_price"] <= 0:
        return {
            "rule_id": "PCR-04",
            "rule": "Rule 6(1)(da)",
            "requirement": "Retail Sale Price (MRP & Taxes)",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(da)",
            "category": "Price & USP",
            "extracted_evidence": ev,
            "status": "NON_COMPLIANT",
            "confidence": 95.0,
            "score": 0,
            "weight": 15,
            "reason": f"Invalid Retail Sale Price: ₹{val['numeric_price']:.2f}. Price cannot be zero or negative.",
            "recommended_action": "Declare valid Maximum Retail Price in INR."
        }

    if not val.get("has_mrp_prefix"):
        return {
            "rule_id": "PCR-04",
            "rule": "Rule 6(1)(da)",
            "requirement": "Retail Sale Price (MRP & Taxes)",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(da)",
            "category": "Price & USP",
            "extracted_evidence": ev,
            "status": "REQUIRES_REVIEW",
            "confidence": 85.0,
            "score": 10,
            "weight": 15,
            "reason": f"Price found (₹{val['numeric_price']:.2f}), but statutory prefix 'MRP' or 'Maximum Retail Price' is missing before the numeric figure (Rule 6(1)(da)).",
            "recommended_action": "Ensure price is preceded by the words 'MRP' or 'Maximum Retail Price'."
        }

    if val["has_tax_clause"]:
        return {
            "rule_id": "PCR-04",
            "rule": "Rule 6(1)(da)",
            "requirement": "Retail Sale Price (MRP & Taxes)",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(da)",
            "category": "Price & USP",
            "extracted_evidence": ev,
            "status": "COMPLIANT",
            "confidence": 95.0,
            "score": 15,
            "weight": 15,
            "reason": f"Compliant MRP declared: ₹{val['numeric_price']:.2f} with statutory prefix and tax inclusion clause.",
            "recommended_action": "None. Requirement satisfied."
        }

    return {
        "rule_id": "PCR-04",
        "rule": "Rule 6(1)(da)",
        "requirement": "Retail Sale Price (MRP & Taxes)",
        "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(da)",
        "category": "Price & USP",
        "extracted_evidence": ev,
        "status": "REQUIRES_REVIEW",
        "confidence": 85.0,
        "score": 10,
        "weight": 15,
        "reason": f"Price found (₹{val['numeric_price']:.2f}), but mandatory phrase 'inclusive of all taxes' or 'incl. of all taxes' is missing or unverified.",
        "recommended_action": "Ensure 'inclusive of all taxes' or 'incl. of all taxes' is printed immediately adjoining MRP."
    }


def evaluate_unit_sale_price(
    mrp_str: str,
    net_quantity_str: str,
    declared_usp_str: Optional[str] = None,
    evidence: Optional[str] = None
) -> Dict[str, Any]:
    """
    Rule 6(1)(m): Unit Sale Price declaration for packaged commodities.
    Implements statutory exemptions:
    - First Proviso to Rule 6(1)(m): MRP = USP (1kg, 1L, 1m, 1 unit) -> NOT_APPLICABLE.
    - Rule 26(a): Net quantity <= 10g or <= 10ml -> NOT_APPLICABLE.
    """
    mrp_val = validate_mrp(mrp_str)
    qty_val = validate_net_quantity(net_quantity_str)
    
    usp_res = calculate_and_validate_usp(
        mrp_val.get("numeric_price"),
        qty_val.get("numeric_value"),
        qty_val.get("normalized_unit"),
        declared_usp_str,
        raw_quantity_str=net_quantity_str
    )

    ev = evidence or declared_usp_str or usp_res.get("evidence_string") or "Not declared on packaging"

    # Statutory Exemption Check
    if usp_res.get("is_exempt"):
        return {
            "rule_id": "PCR-05",
            "rule": "Rule 6(1)(m)",
            "requirement": "Unit Sale Price (USP) Calculation & Declaration",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(m)",
            "category": "Price & USP",
            "extracted_evidence": ev if declared_usp_str else "Statutorily Exempt",
            "status": "NOT_APPLICABLE",
            "confidence": 95.0,
            "score": 10,
            "weight": 10,
            "reason": usp_res.get("exempt_reason", "Exempt from Unit Sale Price declaration under PCR 2011."),
            "recommended_action": "None. Package qualifies for statutory exemption from Unit Sale Price declaration."
        }

    if not usp_res["applicable"] or usp_res.get("status") == "NOT_CALCULABLE":
        return {
            "rule_id": "PCR-05",
            "rule": "Rule 6(1)(m)",
            "requirement": "Unit Sale Price (USP) Calculation & Declaration",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(m)",
            "category": "Price & USP",
            "extracted_evidence": ev,
            "status": "NOT_DETECTED",
            "confidence": 60.0,
            "score": 0,
            "weight": 10,
            "reason": "USP is NOT_CALCULABLE due to missing or invalid MRP / Net Quantity. Inspector review required.",
            "recommended_action": "Verify physical package to confirm whether Unit Sale Price is declared alongside MRP."
        }

    expected_usp = usp_res["expected_usp"]
    expected_unit = usp_res["expected_unit"]
    calculated_formatted = usp_res.get("calculated_usp_formatted", f"₹{expected_usp:.2f}/{expected_unit}")
    evidence_formula = usp_res.get("evidence_string", f"USP = ₹{expected_usp:.2f}/{expected_unit}")
    declared_usp = usp_res["declared_usp"]

    if declared_usp is None:
        return {
            "rule_id": "PCR-05",
            "rule": "Rule 6(1)(m)",
            "requirement": "Unit Sale Price (USP) Calculation & Declaration",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(m)",
            "category": "Price & USP",
            "extracted_evidence": f"Calculated: {calculated_formatted} ({evidence_formula})",
            "status": "NOT_DETECTED",
            "confidence": 80.0,
            "score": 0,
            "weight": 10,
            "reason": f"Unit Sale Price was not detected on label. Statutory calculated USP: {calculated_formatted}.",
            "recommended_action": f"Declare Unit Sale Price as '{calculated_formatted}' alongside MRP."
        }

    if usp_res["matches"]:
        return {
            "rule_id": "PCR-05",
            "rule": "Rule 6(1)(m)",
            "requirement": "Unit Sale Price (USP) Calculation & Declaration",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(m)",
            "category": "Price & USP",
            "extracted_evidence": ev,
            "status": "COMPLIANT",
            "confidence": 95.0,
            "score": 10,
            "weight": 10,
            "reason": f"Declared USP (₹{declared_usp:.2f}) matches statutory calculated rate ({calculated_formatted}) [{evidence_formula}].",
            "recommended_action": "None. Requirement satisfied."
        }
    else:
        return {
            "rule_id": "PCR-05",
            "rule": "Rule 6(1)(m)",
            "requirement": "Unit Sale Price (USP) Calculation & Declaration",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(m)",
            "category": "Price & USP",
            "extracted_evidence": ev,
            "status": "NON_COMPLIANT",
            "confidence": 90.0,
            "score": 0,
            "weight": 10,
            "reason": f"Declared USP (₹{declared_usp:.2f}) diverges from calculated rate ({calculated_formatted}) [{evidence_formula}]: USP mismatch — Inspector Review Required.",
            "recommended_action": f"Correct Unit Sale Price to statutory rate: {calculated_formatted} ({evidence_formula})."
        }


def evaluate_manufacturing_date(mfg_date_str: str, evidence: Optional[str] = None) -> Dict[str, Any]:
    """Rule 6(1)(d): Month and year of manufacture, pre-packing, or import."""
    raw = (mfg_date_str or "").strip()
    ev = evidence or raw

    if not raw:
        return {
            "rule_id": "PCR-06",
            "rule": "Rule 6(1)(d)",
            "requirement": "Month & Year of Manufacture / Pre-packing",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(d)",
            "category": "Dates & Shelf Life",
            "extracted_evidence": "Not detected in scanned label",
            "status": "NOT_DETECTED",
            "confidence": 75.0,
            "score": 0,
            "weight": 10,
            "reason": "Date of manufacture, pre-packing, or import was not detected in the uploaded image/input.",
            "recommended_action": "Verify physical package to confirm whether Month and Year of manufacture is stamped or printed."
        }

    val = validate_date(raw)

    if val["issues"]:
        return {
            "rule_id": "PCR-06",
            "rule": "Rule 6(1)(d)",
            "requirement": "Month & Year of Manufacture / Pre-packing",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(d)",
            "category": "Dates & Shelf Life",
            "extracted_evidence": ev,
            "status": "NON_COMPLIANT",
            "confidence": 95.0,
            "score": 0,
            "weight": 10,
            "reason": "; ".join(val["issues"]),
            "recommended_action": "Declare valid calendar month and year (e.g. MM/YYYY)."
        }

    if val["valid"]:
        return {
            "rule_id": "PCR-06",
            "rule": "Rule 6(1)(d)",
            "requirement": "Month & Year of Manufacture / Pre-packing",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(d)",
            "category": "Dates & Shelf Life",
            "extracted_evidence": ev,
            "status": "COMPLIANT",
            "confidence": 95.0,
            "score": 10,
            "weight": 10,
            "reason": f"Valid manufacturing/packing date declared: {val['parsed_month']:02d}/{val['parsed_year']}.",
            "recommended_action": "None. Requirement satisfied."
        }

    return {
        "rule_id": "PCR-06",
        "rule": "Rule 6(1)(d)",
        "requirement": "Month & Year of Manufacture / Pre-packing",
        "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(d)",
        "category": "Dates & Shelf Life",
        "extracted_evidence": ev,
        "status": "REQUIRES_REVIEW",
        "confidence": 70.0,
        "score": 5,
        "weight": 10,
        "reason": f"Date value '{raw}' does not clearly follow standard Month/Year format.",
        "recommended_action": "Ensure manufacturing date follows standard MM/YYYY or Month YYYY format."
    }


def evaluate_consumer_care(care_str: str, evidence: Optional[str] = None) -> Dict[str, Any]:
    """Rule 6(1)(e): Consumer grievance redressal details (phone, email, postal address)."""
    raw = (care_str or "").strip()
    ev = evidence or raw

    if not raw:
        return {
            "rule_id": "PCR-07",
            "rule": "Rule 6(1)(e)",
            "requirement": "Consumer Care Redressal Details",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(e)",
            "category": "Consumer Grievance",
            "extracted_evidence": "Not detected in scanned label",
            "status": "NOT_DETECTED",
            "confidence": 75.0,
            "score": 0,
            "weight": 15,
            "reason": "Consumer grievance redressal details were not detected in the uploaded image/input.",
            "recommended_action": "Verify package to confirm presence of consumer grievance contact details (phone/email/address)."
        }

    val = validate_consumer_care(raw)

    if not val["valid"]:
        return {
            "rule_id": "PCR-07",
            "rule": "Rule 6(1)(e)",
            "requirement": "Consumer Care Redressal Details",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(e)",
            "category": "Consumer Grievance",
            "extracted_evidence": ev,
            "status": "NON_COMPLIANT",
            "confidence": 95.0,
            "score": 0,
            "weight": 15,
            "reason": "No valid telephone number, email, or postal redressal address found in consumer care text.",
            "recommended_action": "Provide telephone number, email address, or postal address of consumer grievance officer."
        }

    if val["channel_count"] >= 2:
        return {
            "rule_id": "PCR-07",
            "rule": "Rule 6(1)(e)",
            "requirement": "Consumer Care Redressal Details",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(e)",
            "category": "Consumer Grievance",
            "extracted_evidence": ev,
            "status": "COMPLIANT",
            "confidence": 95.0,
            "score": 15,
            "weight": 15,
            "reason": f"Multi-channel grievance mechanism established ({val['channel_count']} contact points detected).",
            "recommended_action": "None. Requirement satisfied."
        }

    return {
        "rule_id": "PCR-07",
        "rule": "Rule 6(1)(e)",
        "requirement": "Consumer Care Redressal Details",
        "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(e)",
        "category": "Consumer Grievance",
        "extracted_evidence": ev,
        "status": "REQUIRES_REVIEW",
        "confidence": 80.0,
        "score": 10,
        "weight": 15,
        "reason": "Only a single contact channel detected. Legal Metrology mandates phone, email, or postal grievance access.",
        "recommended_action": "Provide both telephone number and email address for consumer complaints."
    }


def evaluate_best_before(
    best_before_str: str,
    commodity_category: Optional[str] = None,
    evidence: Optional[str] = None
) -> Dict[str, Any]:
    """
    Rule 6(1)(g) / FSSAI & PCR: Expiry or Best Before for food/perishable commodities.
    Implements conditional statutory check:
    - Non-food / non-perishable commodities (hardware, apparel, electronics, stationery) -> NOT_APPLICABLE.
    - Food / perishable / cosmetic commodities -> Checked.
    """
    raw = (best_before_str or "").strip()
    ev = evidence or raw

    cat = (commodity_category or "").lower()
    is_non_food = cat in ["non_food", "hardware", "stationery", "apparel", "electronics", "durable", "textile", "industrial"]

    if is_non_food:
        return {
            "rule_id": "PCR-08",
            "rule": "Rule 6(1)(g)",
            "requirement": "Best Before / Expiry Period",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(g)",
            "category": "Dates & Shelf Life",
            "extracted_evidence": ev if raw else "Exempt (Non-food/General Merchandise)",
            "status": "NOT_APPLICABLE",
            "confidence": 95.0,
            "score": 5,
            "weight": 5,
            "reason": "Exempt under Rule 6(1)(g): Commodity is non-perishable / general merchandise and not subject to mandatory expiry or best-before declaration.",
            "recommended_action": "None. Commodity is exempt from shelf-life declaration."
        }

    if not raw:
        return {
            "rule_id": "PCR-08",
            "rule": "Rule 6(1)(g)",
            "requirement": "Best Before / Expiry Period",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(g)",
            "category": "Dates & Shelf Life",
            "extracted_evidence": "Not detected in scanned label",
            "status": "NOT_DETECTED",
            "confidence": 75.0,
            "score": 0,
            "weight": 5,
            "reason": "Best Before / Expiry declaration not detected. Mandatory for food and perishable commodities under Rule 6(1)(g).",
            "recommended_action": "If package contains perishable or food commodity, declare 'Best before [X] months from packaging'."
        }

    return {
        "rule_id": "PCR-08",
        "rule": "Rule 6(1)(g)",
        "requirement": "Best Before / Expiry Period",
        "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(g)",
        "category": "Dates & Shelf Life",
        "extracted_evidence": ev,
        "status": "COMPLIANT",
        "confidence": 95.0,
        "score": 5,
        "weight": 5,
        "reason": f"Shelf-life declaration present: '{raw}'.",
        "recommended_action": "None. Requirement satisfied."
    }


def evaluate_country_of_origin(country_str: str, is_imported: bool = False, evidence: Optional[str] = None) -> Dict[str, Any]:
    """Rule 6(10) amendment: Country of origin declaration."""
    raw = (country_str or "").strip()
    ev = evidence or raw

    if not raw:
        if is_imported:
            return {
                "rule_id": "PCR-09",
                "rule": "Rule 6(10)",
                "requirement": "Country of Origin",
                "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(10)",
                "category": "Origin & Traceability",
                "extracted_evidence": "Not declared for imported product",
                "status": "NON_COMPLIANT",
                "confidence": 95.0,
                "score": 0,
                "weight": 5,
                "reason": "Country of origin is strictly mandatory for imported packaged goods.",
                "recommended_action": "Declare 'Country of Origin: [Country Name]' conspicuously on label."
            }
        else:
            return {
                "rule_id": "PCR-09",
                "rule": "Rule 6(10)",
                "requirement": "Country of Origin",
                "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(10)",
                "category": "Origin & Traceability",
                "extracted_evidence": "Not explicitly stated (Domestic manufacturer address present)",
                "status": "REQUIRES_REVIEW",
                "confidence": 75.0,
                "score": 3,
                "weight": 5,
                "reason": "Country of origin not explicitly declared (recommended for domestic, mandatory for imported).",
                "recommended_action": "Declare country of origin or manufacture explicitly."
            }

    return {
        "rule_id": "PCR-09",
        "rule": "Rule 6(10)",
        "requirement": "Country of Origin",
        "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(10)",
        "category": "Origin & Traceability",
        "extracted_evidence": ev,
        "status": "COMPLIANT",
        "confidence": 95.0,
        "score": 5,
        "weight": 5,
        "reason": f"Country of origin declared: '{raw}'.",
        "recommended_action": "None. Requirement satisfied."
    }


def evaluate_batch_number(batch_str: str, evidence: Optional[str] = None) -> Dict[str, Any]:
    """Traceability & Lot Identification under Packaged Commodities Rules."""
    raw = (batch_str or "").strip()
    ev = evidence or raw

    if not raw:
        return {
            "rule_id": "PCR-10",
            "rule": "PCR Traceability",
            "requirement": "Batch / Lot Identification Code",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(d) & Traceability",
            "category": "Origin & Traceability",
            "extracted_evidence": "Not detected in scanned label",
            "status": "NOT_DETECTED",
            "confidence": 75.0,
            "score": 0,
            "weight": 5,
            "reason": "Batch / Lot number not detected. Required for package traceability and quality control.",
            "recommended_action": "Print or emboss Batch/Lot number (e.g. 'B.No. [Number]')."
        }

    return {
        "rule_id": "PCR-10",
        "rule": "PCR Traceability",
        "requirement": "Batch / Lot Identification Code",
        "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(d) & Traceability",
        "category": "Origin & Traceability",
        "extracted_evidence": ev,
        "status": "COMPLIANT",
        "confidence": 95.0,
        "score": 5,
        "weight": 5,
        "reason": f"Batch identifier declared: '{raw}'.",
        "recommended_action": "None. Requirement satisfied."
    }


def evaluate_compliance(
    fields: Dict[str, str],
    fields_detail: Optional[Dict[str, Any]] = None,
    commodity_category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates all packaging declarations against Legal Metrology (Packaged Commodities) Rules, 2011.
    Calculates explainable category scores, overall status, and statutory inspector audit logs.

    Stage 1:
    - Distinguishes NOT_DETECTED from NON_COMPLIANT.
    - Conditionally evaluates shelf life (Rule 6(1)(g)) based on commodity category.
    - Applies statutory exemptions for Unit Sale Price (Rule 6(1)(m)) and small packages (Rule 26).
    """
    fd = fields_detail or {}

    # Category auto-detection heuristic if not explicitly passed
    resolved_category = commodity_category
    if not resolved_category:
        p_name = (fields.get("Product Name") or "").lower()
        food_tokens = ["powder", "salt", "tea", "coffee", "oil", "biscuit", "flour", "atta", "rice", "sugar", "masala", "food", "snack", "juice", "drink", "milk", "spice", "dal", "cookie"]
        non_food_tokens = ["screwdriver", "bulb", "led", "wire", "cable", "socket", "shirt", "towel", "cloth", "textile", "screw", "pen", "notebook", "stapler", "battery", "soap", "detergent", "shampoo"]
        if any(t in p_name for t in non_food_tokens):
            resolved_category = "non_food"
        elif any(t in p_name for t in food_tokens):
            resolved_category = "food"
        else:
            resolved_category = "food"

    def get_ev(field_name):
        item = fd.get(field_name, {})
        if isinstance(item, dict):
            return item.get("evidence") or fields.get(field_name, "")
        return fields.get(field_name, "")

    evaluations = [
        evaluate_product_name(fields.get("Product Name", ""), get_ev("Product Name")),
        evaluate_manufacturer_address(fields.get("Manufacturer", ""), get_ev("Manufacturer")),
        evaluate_net_quantity(
            fields.get("Net Quantity", ""),
            get_ev("Net Quantity"),
            commodity_name=fields.get("Product Name", ""),
            commodity_category=resolved_category
        ),
        evaluate_mrp(fields.get("MRP", ""), get_ev("MRP")),
        evaluate_unit_sale_price(
            fields.get("MRP", ""),
            fields.get("Net Quantity", ""),
            fields.get("Unit Sale Price", ""),
            get_ev("Unit Sale Price")
        ),
        evaluate_manufacturing_date(fields.get("Manufacturing Date", ""), get_ev("Manufacturing Date")),
        evaluate_consumer_care(fields.get("Consumer Care", ""), get_ev("Consumer Care")),
        evaluate_best_before(fields.get("Best Before / Expiry", ""), resolved_category, get_ev("Best Before / Expiry")),
        evaluate_country_of_origin(fields.get("Country of Origin", ""), evidence=get_ev("Country of Origin")),
        evaluate_batch_number(fields.get("Batch Number", ""), get_ev("Batch Number"))
    ]

    FIELD_NAMES = [
        "Product Name",
        "Manufacturer",
        "Net Quantity",
        "MRP",
        "Unit Sale Price",
        "Manufacturing Date",
        "Consumer Care",
        "Best Before / Expiry",
        "Country of Origin",
        "Batch Number"
    ]

    threshold = get_extraction_confidence_threshold()

    # Stage 3: Extraction Confidence & Status Integrity post-processing
    for idx, e in enumerate(evaluations):
        fname = FIELD_NAMES[idx]
        detail = fd.get(fname, {}) if isinstance(fd.get(fname), dict) else {}
        raw_val = (fields.get(fname) or "").strip()

        # Determine extraction source
        if detail.get("source"):
            source = detail.get("source")
        elif fd:
            source = "Local OCR (Tesseract)"
        else:
            source = "Manual Input"

        # Determine confidence
        conf = detail.get("confidence")
        if conf is not None:
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = None

        has_conflict = bool(detail.get("has_conflict"))
        conflict_detail = detail.get("conflict_detail") or ""

        # Status integrity & threshold adjustments:
        if has_conflict and e["status"] != "NOT_APPLICABLE":
            e["status"] = "REQUIRES_REVIEW"
            e["score"] = round(e["weight"] * 0.5)
            e["reason"] = f"Extraction discrepancy detected ({conflict_detail}). Physical label inspection required."
            e["recommended_action"] = "Inspect physical packaging to resolve discrepancy between extraction engines."
        elif conf is not None and conf < threshold:
            if e["status"] == "COMPLIANT":
                e["status"] = "REQUIRES_REVIEW"
                e["score"] = round(e["weight"] * 0.6)
                e["reason"] = (
                    f"Low extraction confidence ({conf:.1f}% < {threshold:.1f}% threshold). "
                    f"Declaration format appears valid ('{raw_val}'), but requires physical verification."
                )
                e["recommended_action"] = "Inspect physical packaging to verify accuracy of low-confidence reading."
            elif e["status"] == "NON_COMPLIANT":
                # Crucial Legal Safeguard: Do NOT convict on low confidence / OCR noise
                e["status"] = "REQUIRES_REVIEW"
                e["score"] = round(e["weight"] * 0.3)
                e["reason"] = (
                    f"Potential statutory irregularity detected ('{raw_val}'), but extraction confidence "
                    f"({conf:.1f}%) is below verification threshold ({threshold:.1f}%). "
                    "Physical verification required before statutory enforcement."
                )
                e["recommended_action"] = "Inspect physical packaging to confirm whether suspected defect exists."

        # Attach metadata to evaluation entry
        e["extracted_value"] = raw_val
        e["extraction_source"] = source
        e["confidence"] = conf
        e["has_conflict"] = has_conflict
        e["conflict_detail"] = conflict_detail
        e["source_images"] = detail.get("source_images", [])
        e["source_image"] = detail.get("source_image", "")

    total_score = sum(e["score"] for e in evaluations)
    max_score = sum(e["weight"] for e in evaluations)
    score_percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 100.0

    fails = [e for e in evaluations if e["status"] == "NON_COMPLIANT"]
    not_detected = [e for e in evaluations if e["status"] == "NOT_DETECTED"]
    reviews = [e for e in evaluations if e["status"] == "REQUIRES_REVIEW"]
    passes = [e for e in evaluations if e["status"] == "COMPLIANT"]
    not_applicable = [e for e in evaluations if e["status"] == "NOT_APPLICABLE"]

    # Critical Legal Distinction:
    # A product is NON-COMPLIANT ONLY when there is an affirmative statutory violation (e.g. prohibited symbols, price <= 0, math mismatch).
    # If fields are merely NOT_DETECTED or require review, overall status is REQUIRES_REVIEW (not NON-COMPLIANT).
    if len(fails) > 0:
        overall_status = "NON-COMPLIANT"
    elif len(not_detected) > 0 or len(reviews) > 0:
        overall_status = "REQUIRES_REVIEW"
    else:
        overall_status = "COMPLIANT"

    # Explainable Category Breakdown
    categories = {}
    for e in evaluations:
        cat = e["category"]
        if cat not in categories:
            categories[cat] = {"score": 0, "max_score": 0, "passed": 0, "failed": 0, "warnings": 0}
        categories[cat]["score"] += e["score"]
        categories[cat]["max_score"] += e["weight"]
        if e["status"] in ["COMPLIANT", "NOT_APPLICABLE"]:
            categories[cat]["passed"] += 1
        elif e["status"] == "NON_COMPLIANT":
            categories[cat]["failed"] += 1
        else:
            categories[cat]["warnings"] += 1

    category_scores = {}
    for cat, data in categories.items():
        pct = round((data["score"] / data["max_score"]) * 100, 1) if data["max_score"] > 0 else 100.0
        category_scores[cat] = {
            "score": data["score"],
            "max_score": data["max_score"],
            "percentage": pct,
            "passed": data["passed"],
            "failed": data["failed"],
            "warnings": data["warnings"]
        }

    score_explanation = {
        "basis": "Explainable scoring weighted across Legal Metrology PCR 2011 statutory rules.",
        "total_score": total_score,
        "max_score": max_score,
        "score_percentage": score_percentage,
        "statutory_violations": len(fails),
        "not_detected_count": len(not_detected),
        "requires_review_count": len(reviews),
        "compliant_count": len(passes),
        "exempt_count": len(not_applicable),
        "summary": (
            "Affirmative statutory violations detected. Package is legally non-compliant."
            if len(fails) > 0
            else (
                "One or more declarations were not detected or have low extraction confidence. "
                "Physical package review is required to complete certification; no affirmative legal violations detected."
                if (len(not_detected) > 0 or len(reviews) > 0)
                else "All mandatory statutory declarations verified and fully compliant."
            )
        )
    }

    return {
        "status": overall_status,
        "score": score_percentage,
        "total_rules": len(evaluations),
        "passed_count": len(passes),
        "warning_count": len(reviews),
        "failed_count": len(fails),
        "not_detected_count": len(not_detected),
        "not_applicable_count": len(not_applicable),
        "category_scores": category_scores,
        "score_explanation": score_explanation,
        "audit_table": evaluations,
        "commodity_category": commodity_category or "general",
        "missing": [f"{e['requirement']} ({e['rule']})" for e in fails],
        "not_detected_items": [f"{e['requirement']} ({e['rule']})" for e in not_detected],
        "requires_review_items": [f"{e['requirement']} ({e['rule']})" for e in reviews]
    }
