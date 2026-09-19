"""
Field Validation Layer for Packaged Commodity Compliance.
Validates extracted raw data before passing to the Legal Metrology PCR 2011 compliance engine.
Detects illegal metric unit symbols, mathematical unit price discrepancies, impossible dates,
and incomplete addresses or contact channels.
"""

import os
import sys
import re
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_extraction_confidence_threshold

# Legal metric units permitted under the Legal Metrology Act, 2009 & PCR 2011 First Schedule
PERMITTED_METRIC_UNITS = {
    # Mass
    "g": "g", "gram": "g", "grams": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    # Volume
    "ml": "ml", "millilitre": "ml", "milliliter": "ml", "millilitres": "ml",
    "l": "l", "litre": "l", "liter": "l", "litres": "l", "liters": "l", "kl": "kl", "cl": "cl",
    # Length & Area
    "m": "m", "metre": "m", "meter": "m", "metres": "m", "meters": "m",
    "cm": "cm", "centimetre": "cm", "centimeter": "cm", "centimetres": "cm",
    "mm": "mm", "millimetre": "mm", "millimeter": "mm",
    "sq m": "sq m", "m2": "sq m", "m²": "sq m",
    "sq cm": "sq cm", "cm2": "sq cm", "cm²": "sq cm",
    "sq mm": "sq mm", "mm2": "sq mm", "mm²": "sq mm",
    # Count / Number
    "n": "number", "u": "unit", "unit": "unit", "units": "unit",
    "piece": "piece", "pieces": "piece", "number": "number", "count": "number", "item": "number", "items": "number"
}

# Explicitly illegal / prohibited abbreviations under PCR 2011 Rule 13(2), (3) and Section 11 of the Act
PROHIBITED_UNIT_ABBREVIATIONS = {
    # Mass prohibited abbreviations & punctuation
    "gm": "g",
    "gms": "g",
    "g.": "g",
    "gm.": "g",
    "gms.": "g",
    "kg.": "kg",
    "kgs": "kg",
    "kgs.": "kg",
    "kilo": "kg",
    "kilos": "kg",
    "mg.": "mg",
    "mgs": "mg",
    "mgs.": "mg",
    # Volume prohibited abbreviations & punctuation
    "ml.": "ml",
    "mls": "ml",
    "mls.": "ml",
    "ltr": "l",
    "ltr.": "l",
    "ltrs": "l",
    "ltrs.": "l",
    "lt": "l",
    "lt.": "l",
    "l.": "l",
    # Length prohibited abbreviations
    "mt": "m",
    "mts": "m",
    "mtr": "m",
    "mtrs": "m",
    "m.": "m",
    "cm.": "cm",
    "mm.": "mm",
    # Imperial / Non-Metric Units barred under Legal Metrology Act, 2009
    "lb": "kg",
    "lbs": "kg",
    "pound": "kg",
    "pounds": "kg",
    "oz": "g",
    "ounce": "g",
    "ounces": "g",
    "fl oz": "ml",
    "fluid ounce": "ml"
}

# Rule 13: Standard metric symbols for mass & length must strictly be lower-case in SI
STRICTLY_LOWERCASE_UNITS = {
    "G": "g",
    "KG": "kg",
    "MG": "mg",
    "M": "m",
    "CM": "cm",
    "MM": "mm",
    "ML": "ml"
}

# Volume symbols where capital L is standard and legally recognized (SI / BIPM / Rule 13)
STANDARD_VOLUME_SYMBOLS = {"ml", "mL", "l", "L", "cl", "cL", "kl", "kL"}

# Commodities susceptible to climatic variation under Proviso to Rule 11(2)
CLIMATIC_VARIATION_COMMODITIES = [
    "soap", "toilet soap", "bathing soap", "bathing bar", "toilet bar",
    "detergent bar", "laundry soap", "laundry bar", "camphor", "naphthalene"
]


def is_climatic_variation_commodity(
    commodity_name: Optional[str] = None,
    commodity_category: Optional[str] = None
) -> bool:
    """
    Identifies commodities susceptible to climatic variations (e.g. moisture loss/evaporation)
    under the Proviso to Rule 11(2), such as soaps, camphor, etc.
    """
    target = f"{commodity_name or ''} {commodity_category or ''}".lower()
    return any(token in target for token in CLIMATIC_VARIATION_COMMODITIES)


def validate_mrp(mrp_str: str) -> Dict[str, Any]:
    """
    Validates Retail Sale Price declaration.
    Returns parsed numeric price, tax clause presence, and validity flags.
    """
    raw = (mrp_str or "").strip()
    if not raw:
        return {
            "valid": False,
            "numeric_price": None,
            "has_tax_clause": False,
            "issues": ["Missing MRP declaration."],
            "warnings": [],
            "requires_review": True
        }

    issues = []
    warnings = []
    
    # Clean string and look for numeric price
    cleaned = raw.replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    numeric_price = float(match.group(1)) if match else None

    if numeric_price is None:
        issues.append(f"No numeric price found in MRP declaration '{raw}'.")
    elif numeric_price <= 0:
        issues.append(f"Price cannot be zero or negative (found: {numeric_price}).")
    elif numeric_price > 500000:
        warnings.append(f"Suspiciously high retail price detected: ₹{numeric_price:.2f}.")

    lower = raw.lower()
    has_tax_clause = any(k in lower for k in ["incl", "all taxes", "inclusive of all taxes", "tax"])
    if not has_tax_clause and numeric_price is not None:
        warnings.append("Statutory phrase 'inclusive of all taxes' or 'incl. of all taxes' is missing.")

    has_mrp_prefix = bool(re.search(r"(?:mrp|max(?:imum)?\.?\s*retail\s*price)", lower))
    if not has_mrp_prefix and numeric_price is not None:
        warnings.append("Statutory prefix 'MRP' or 'Maximum Retail Price' is missing before the numeric price (Rule 6(1)(da)).")

    is_valid = bool(numeric_price is not None and numeric_price > 0)

    return {
        "valid": is_valid,
        "numeric_price": numeric_price,
        "has_tax_clause": has_tax_clause,
        "has_mrp_prefix": has_mrp_prefix,
        "issues": issues,
        "warnings": warnings,
        "requires_review": bool(warnings or not is_valid)
    }


def validate_net_quantity(
    quantity_str: str,
    commodity_name: Optional[str] = None,
    commodity_category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validates declared net quantity against Legal Metrology metric standards (PCR 2011).
    Enforces:
    - Rule 11(2): Prohibits qualifying words ('approx', 'gross', 'min', 'not less than');
                  permits 'when packed' under statutory Proviso for climatic commodities (soaps/camphor).
    - Rule 12(3): Prohibits vulgar fractions (e.g. '1/2 kg'); mandates decimal notation ('0.5 kg').
    - Rule 13(1): Enforces standard SI symbols; allows standard volume symbols ('ml', 'mL', 'l', 'L').
    - Rule 13(2): Prohibits period/dot after unit symbol ('g.', 'kg.').
    - Rule 13(3): Prohibits plural symbols ('gms', 'kgs', 'mls', 'ltrs').
    - Rule 13 Formatting: Issues advisory when numerical value lacks separating space ('50g' vs '50 g').
    """
    raw = (quantity_str or "").strip()
    if not raw:
        return {
            "valid": False,
            "numeric_value": None,
            "unit": None,
            "normalized_unit": None,
            "is_illegal_unit": False,
            "is_case_violation": False,
            "is_vulgar_fraction": False,
            "has_prohibited_qualifier": False,
            "is_permitted_climatic_qualifier": False,
            "has_spacing_advisory": False,
            "issues": ["Missing net quantity declaration."],
            "warnings": [],
            "requires_review": True
        }

    issues = []
    warnings = []
    raw_lower = raw.lower()

    # 1. Rule 12(3) Check for prohibited vulgar fractions (e.g. '1/2 kg', '3/4 L')
    is_vulgar_fraction = False
    vulgar_match = re.search(r"\b(\d+/\d+)\b", raw)
    if vulgar_match:
        is_vulgar_fraction = True
        issues.append(
            f"Prohibited vulgar fraction '{vulgar_match.group(1)}' used in net quantity. "
            f"Legal Metrology Rule 12(3) mandates decimal representation (e.g. '0.5' instead of '1/2')."
        )

    # 2. Rule 11(2) Check for strictly prohibited qualifying/negating words (All commodities)
    strictly_prohibited_qualifiers = [
        "approx", "approximate", "approximately", "approx.",
        "gross", "minimum", "not less than", "min", "min.",
        "around", "nearly", "estimated"
    ]
    has_prohibited_qualifier = False
    for q in strictly_prohibited_qualifiers:
        if re.search(r"\b" + re.escape(q) + r"\b", raw_lower):
            has_prohibited_qualifier = True
            issues.append(
                f"Prohibited qualifying expression '{q}' used in net quantity. "
                f"Legal Metrology Rule 11(2) strictly prohibits qualifying phrases like 'approximate' or 'not less than' "
                f"that tend to negate or make indefinite the net quantity declaration."
            )
            break

    # 3. Rule 11(2) Proviso Check for 'when packed' on commodities susceptible to climatic variation
    is_climatic = is_climatic_variation_commodity(commodity_name, commodity_category)
    is_permitted_climatic_qualifier = False
    has_when_packed = bool(re.search(r"\b(?:net\s+weight\s+)?when\s+(?:packed|pkd)\b", raw_lower))

    if has_when_packed:
        if is_climatic:
            is_permitted_climatic_qualifier = True
            warnings.append(
                "Permitted under Proviso to Rule 11(2): The expression 'when packed' is legally permissible "
                "for commodities susceptible to climatic variation resulting in substantial change in weight (e.g. soaps, camphor)."
            )
        else:
            has_prohibited_qualifier = True
            issues.append(
                "Prohibited qualifying expression 'when packed' used in net quantity. "
                "Under Legal Metrology Rule 11(2), qualifying expressions like 'when packed' are prohibited "
                "except for commodities susceptible to climatic variations (such as soaps)."
            )

    # 4. Rule 13 Formatting Spacing Check (e.g. '50g' vs '50 g')
    has_spacing_advisory = False
    if re.search(r"\d[A-Za-z]", raw):
        has_spacing_advisory = True
        warnings.append(
            "Formatting Recommendation under Rule 13: Standard metric typography recommends a space between "
            "the numerical value and the unit symbol (e.g., '50 g' rather than '50g')."
        )

    # 5. Extract numeric magnitude and unit symbol
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([A-Za-z.]+)", raw)
    if not match:
        return {
            "valid": False,
            "numeric_value": None,
            "unit": None,
            "normalized_unit": None,
            "is_illegal_unit": False,
            "is_case_violation": False,
            "is_vulgar_fraction": is_vulgar_fraction,
            "has_prohibited_qualifier": has_prohibited_qualifier,
            "is_permitted_climatic_qualifier": is_permitted_climatic_qualifier,
            "has_spacing_advisory": has_spacing_advisory,
            "has_free_quantity": False,
            "free_magnitude": None,
            "free_unit": None,
            "base_numeric_value": None,
            "issues": issues + [f"Unable to parse standard numeric magnitude and unit from '{raw}'."],
            "warnings": warnings,
            "requires_review": True
        }

    magnitude = float(match.group(1))
    unit_raw = match.group(2)
    unit_declared = unit_raw.lower()

    # Detect promotional free quantity: e.g. "+ 20 g free", "extra 50 ml free", "(20g free)"
    has_free_quantity = False
    free_magnitude = None
    free_unit = None
    free_match = re.search(r"(?:\+|\band\b|\(extra|\(free)\s*(\d+(?:\.\d+)?)\s*([A-Za-z.]+)?\s*(?:free|extra|complimentary)?", raw_lower)
    if not free_match:
        free_match = re.search(r"\+\s*(\d+(?:\.\d+)?)\s*([A-Za-z.]+)", raw_lower)
    if free_match and ("free" in raw_lower or "extra" in raw_lower or "+" in raw):
        try:
            free_magnitude = float(free_match.group(1))
            free_unit = free_match.group(2) if free_match.group(2) else unit_raw
            has_free_quantity = True
            warnings.append(
                f"Promotional free quantity detected: '{free_magnitude:g} {free_unit} free'. "
                f"Under PCR 2011 Rule 6(1)(m), statutory Unit Sale Price is computed on base quantity ({magnitude:g} {unit_raw})."
            )
        except (ValueError, TypeError):
            pass

    if magnitude <= 0:
        issues.append(f"Quantity magnitude must be positive (found: {magnitude}).")

    # 6. Rule 13 Case Sensitivity Evaluation
    is_case_violation = False
    if unit_raw in STRICTLY_LOWERCASE_UNITS:
        is_case_violation = True
        warnings.append(
            f"Formatting Advisory under Rule 13: Metric symbol for mass/length should be lower-case ('{STRICTLY_LOWERCASE_UNITS[unit_raw]}'). "
            f"Detected upper-case '{unit_raw}'; verify whether packaging typography is all-caps or OCR capitalization."
        )

    # 7. Rule 13(2), (3) Check for prohibited / non-standard symbols
    is_illegal = unit_declared in PROHIBITED_UNIT_ABBREVIATIONS
    normalized_unit = None

    if is_illegal:
        correct_symbol = PROHIBITED_UNIT_ABBREVIATIONS[unit_declared]
        issues.append(
            f"Prohibited unit symbol '{unit_raw}' used. Legal Metrology PCR 2011 mandates '{correct_symbol}'."
        )
        normalized_unit = correct_symbol
    elif unit_declared in PERMITTED_METRIC_UNITS:
        normalized_unit = PERMITTED_METRIC_UNITS[unit_declared]
    else:
        warnings.append(f"Unrecognized metric unit symbol '{unit_raw}'. Verify statutory schedule.")

    # 8. Rule 12(1) & (2) Magnitude Guidance Advisories
    if normalized_unit == "g" and magnitude >= 1000.0:
        warnings.append(f"Advisory under Rule 12(1): Quantities of 1000g or more should preferably be expressed in 'kg' (e.g. '{magnitude/1000:g} kg').")
    elif normalized_unit == "kg" and magnitude < 1.0:
        warnings.append(f"Advisory under Rule 12(1): Quantities under 1 kg should preferably be expressed in 'g' (e.g. '{magnitude*1000:g} g').")
    elif normalized_unit == "ml" and magnitude >= 1000.0:
        warnings.append(f"Advisory under Rule 12(2): Volumes of 1000ml or more should preferably be expressed in 'l' (e.g. '{magnitude/1000:g} L').")
    elif normalized_unit == "l" and magnitude < 1.0:
        warnings.append(f"Advisory under Rule 12(2): Volumes under 1 L should preferably be expressed in 'ml' (e.g. '{magnitude*1000:g} ml').")

    is_valid = bool(
        magnitude > 0
        and normalized_unit
        and not is_illegal
        and not is_case_violation
        and not has_prohibited_qualifier
        and not is_vulgar_fraction
    )

    return {
        "valid": is_valid,
        "numeric_value": magnitude,
        "base_numeric_value": magnitude,
        "unit": unit_raw,
        "normalized_unit": normalized_unit,
        "is_illegal_unit": is_illegal,
        "is_case_violation": is_case_violation,
        "is_vulgar_fraction": is_vulgar_fraction,
        "has_prohibited_qualifier": has_prohibited_qualifier,
        "is_permitted_climatic_qualifier": is_permitted_climatic_qualifier,
        "has_spacing_advisory": has_spacing_advisory,
        "has_free_quantity": has_free_quantity,
        "free_magnitude": free_magnitude,
        "free_unit": free_unit,
        "issues": issues,
        "warnings": warnings,
        "requires_review": bool(warnings or not is_valid)
    }


def calculate_and_validate_usp(
    mrp: Optional[float],
    quantity: Optional[float],
    unit: Optional[str],
    declared_usp_str: Optional[str] = None,
    raw_quantity_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates Unit Sale Price (USP) under Rule 6(1)(m) of Legal Metrology PCR 2011.
    Statutory Calculation Logic:
    - Weight < 1 kg  -> USP = MRP / quantity_in_grams  -> Display: ₹X.XX/g
    - Weight >= 1 kg -> USP = MRP / quantity_in_kg     -> Display: ₹X.XX/kg
    - Volume < 1 L   -> USP = MRP / quantity_in_ml     -> Display: ₹X.XX/ml
    - Volume >= 1 L  -> USP = MRP / quantity_in_litres -> Display: ₹X.XX/L
    - Unit / Count   -> USP = MRP / number_of_units    -> Display: ₹X.XX/unit
    - Length < 1 m   -> USP = MRP / quantity_in_cm     -> Display: ₹X.XX/cm
    - Length >= 1 m  -> USP = MRP / quantity_in_metres -> Display: ₹X.XX/m
    - Strict Rounding: exactly 2 decimal places.
    - Free Additional Quantity: excluded from statutory base USP calculation as per PCR 2011.
    - Missing / Invalid: returns status 'NOT_CALCULABLE' with requires_review=True.
    - Statutory Exemptions: Proviso to Rule 6(1)(m) (1kg, 1L, 1m, 1unit) and Rule 26(a) (<=10g, <=10ml).
    - Preserves printed USP and flags mismatch if divergent.
    """
    if mrp is None or mrp <= 0 or quantity is None or quantity <= 0 or not unit:
        return {
            "status": "NOT_CALCULABLE",
            "applicable": False,
            "is_exempt": False,
            "is_exempt_equal_mrp": False,
            "is_small_pack_exempt": False,
            "exempt_reason": "",
            "expected_usp": None,
            "expected_unit": None,
            "calculated_usp": None,
            "calculated_usp_formatted": "NOT_CALCULABLE",
            "calculated_unit": None,
            "evidence_string": "USP is NOT_CALCULABLE: Valid MRP and Net Quantity are required for statutory calculation.",
            "declared_usp": None,
            "printed_usp": declared_usp_str,
            "printed_usp_num": None,
            "matches": None,
            "has_mismatch": False,
            "mismatch_message": None,
            "issues": ["Missing or invalid MRP / Net Quantity"],
            "warnings": [],
            "requires_review": True
        }

    u = str(unit).lower().strip().rstrip(".")
    calculated_usp = None
    calculated_unit = None
    base_qty_display = ""
    issues = []
    warnings = []

    # Check promotional free quantity
    free_info = None
    if raw_quantity_str:
        q_eval = validate_net_quantity(raw_quantity_str)
        if q_eval.get("has_free_quantity") and q_eval.get("free_magnitude"):
            free_info = {
                "free_magnitude": q_eval["free_magnitude"],
                "free_unit": q_eval.get("free_unit") or unit
            }

    # Statutory normalization & calculation
    # 1. Weight
    if u in ["g", "gram", "grams", "gm", "gms"]:
        if quantity < 1000.0:
            calculated_usp = round(mrp / quantity, 2)
            calculated_unit = "g"
            base_qty_display = f"{quantity:g} g"
        else:
            qty_in_kg = quantity / 1000.0
            calculated_usp = round(mrp / qty_in_kg, 2)
            calculated_unit = "kg"
            base_qty_display = f"{qty_in_kg:g} kg"
    elif u in ["kg", "kilogram", "kilograms", "kgs"]:
        if quantity < 1.0:
            qty_in_g = quantity * 1000.0
            calculated_usp = round(mrp / qty_in_g, 2)
            calculated_unit = "g"
            base_qty_display = f"{qty_in_g:g} g"
        else:
            calculated_usp = round(mrp / quantity, 2)
            calculated_unit = "kg"
            base_qty_display = f"{quantity:g} kg"
    elif u in ["mg", "milligram", "milligrams"]:
        grams = quantity / 1000.0
        if grams < 1000.0:
            calculated_usp = round(mrp / grams, 2) if grams > 0 else None
            calculated_unit = "g"
            base_qty_display = f"{grams:g} g"
        else:
            kg = grams / 1000.0
            calculated_usp = round(mrp / kg, 2) if kg > 0 else None
            calculated_unit = "kg"
            base_qty_display = f"{kg:g} kg"

    # 2. Volume
    elif u in ["ml", "millilitre", "milliliter", "mls", "ml."]:
        if quantity < 1000.0:
            calculated_usp = round(mrp / quantity, 2)
            calculated_unit = "ml"
            base_qty_display = f"{quantity:g} ml"
        else:
            qty_in_l = quantity / 1000.0
            calculated_usp = round(mrp / qty_in_l, 2)
            calculated_unit = "L"
            base_qty_display = f"{qty_in_l:g} L"
    elif u in ["l", "litre", "liter", "ltr", "ltrs", "l."]:
        if quantity < 1.0:
            qty_in_ml = quantity * 1000.0
            calculated_usp = round(mrp / qty_in_ml, 2)
            calculated_unit = "ml"
            base_qty_display = f"{qty_in_ml:g} ml"
        else:
            calculated_usp = round(mrp / quantity, 2)
            calculated_unit = "L"
            base_qty_display = f"{quantity:g} L"

    # 3. Unit / Count / Number
    elif u in ["number", "unit", "units", "piece", "pieces", "item", "items", "u", "n", "count"]:
        calculated_usp = round(mrp / quantity, 2)
        calculated_unit = "unit"
        base_qty_display = f"{quantity:g} unit" if quantity == 1.0 else f"{quantity:g} units"

    # 4. Length
    elif u in ["cm", "centimetre", "centimeter"]:
        if quantity < 100.0:
            calculated_usp = round(mrp / quantity, 2)
            calculated_unit = "cm"
            base_qty_display = f"{quantity:g} cm"
        else:
            qty_in_m = quantity / 100.0
            calculated_usp = round(mrp / qty_in_m, 2)
            calculated_unit = "m"
            base_qty_display = f"{qty_in_m:g} m"
    elif u in ["m", "meter", "metre"]:
        if quantity < 1.0:
            qty_in_cm = quantity * 100.0
            calculated_usp = round(mrp / qty_in_cm, 2)
            calculated_unit = "cm"
            base_qty_display = f"{qty_in_cm:g} cm"
        else:
            calculated_usp = round(mrp / quantity, 2)
            calculated_unit = "m"
            base_qty_display = f"{quantity:g} m"
    else:
        return {
            "status": "NOT_CALCULABLE",
            "applicable": False,
            "is_exempt": False,
            "is_exempt_equal_mrp": False,
            "is_small_pack_exempt": False,
            "exempt_reason": "",
            "expected_usp": None,
            "expected_unit": None,
            "calculated_usp": None,
            "calculated_usp_formatted": "NOT_CALCULABLE",
            "calculated_unit": None,
            "evidence_string": f"USP is NOT_CALCULABLE: Unrecognized unit '{unit}' for statutory calculation.",
            "declared_usp": None,
            "printed_usp": declared_usp_str,
            "printed_usp_num": None,
            "matches": None,
            "has_mismatch": False,
            "mismatch_message": None,
            "issues": [f"Unrecognized unit '{unit}'"],
            "warnings": [],
            "requires_review": True
        }

    # Formatted display string
    calculated_usp_formatted = f"₹{calculated_usp:.2f}/{calculated_unit}"
    mrp_disp = f"₹{mrp:g}" if mrp == int(mrp) else f"₹{mrp:.2f}"

    if free_info:
        evidence_string = (
            f"USP = {mrp_disp} ÷ {base_qty_display} = {calculated_usp_formatted} "
            f"(Statutory base quantity {base_qty_display}; promotional {free_info['free_magnitude']:g} {free_info['free_unit']} free excluded under PCR 2011)"
        )
    else:
        evidence_string = f"USP = {mrp_disp} ÷ {base_qty_display} = {calculated_usp_formatted}"

    # Parse declared / printed USP if provided
    declared_num = None
    if declared_usp_str:
        usp_match = re.search(r"(\d+(?:\.\d+)?)", str(declared_usp_str).replace(",", ""))
        if usp_match:
            declared_num = float(usp_match.group(1))

    # Check statutory exemptions under Rule 6(1)(m) Proviso and Rule 26(a)
    is_exempt_equal_mrp = False
    is_small_pack_exempt = False
    exempt_reason = ""

    if ((quantity == 1.0 and u in ["kg", "kilogram", "kilograms", "l", "litre", "liter", "m", "meter", "metre", "number", "unit", "units", "piece", "pieces", "n", "u"])
        or (quantity == 1000.0 and u in ["g", "gram", "grams", "ml", "millilitre", "milliliter"])):
        is_exempt_equal_mrp = True
        exempt_reason = "Exempt under Proviso to Rule 6(1)(m): Retail Sale Price is equal to the Unit Sale Price for packages of exactly 1 kg, 1 litre, 1 metre, or 1 unit."
    elif (u in ["g", "gram", "grams", "ml", "millilitre", "milliliter"] and quantity <= 10.0) or (u in ["mg"] and quantity <= 10000.0):
        is_small_pack_exempt = True
        exempt_reason = "Exempt under Rule 26(a) & Rule 6(1)(m): Packages containing net quantity of 10g or 10ml or less are exempt from Unit Sale Price declaration."

    if is_exempt_equal_mrp or is_small_pack_exempt:
        return {
            "status": "EXEMPT",
            "applicable": False,
            "is_exempt": True,
            "is_exempt_equal_mrp": is_exempt_equal_mrp,
            "is_small_pack_exempt": is_small_pack_exempt,
            "exempt_reason": exempt_reason,
            "expected_usp": mrp if is_exempt_equal_mrp else calculated_usp,
            "expected_unit": u if is_exempt_equal_mrp else calculated_unit,
            "calculated_usp": calculated_usp,
            "calculated_usp_formatted": calculated_usp_formatted,
            "calculated_unit": calculated_unit,
            "evidence_string": evidence_string,
            "declared_usp": declared_num,
            "printed_usp": declared_usp_str,
            "printed_usp_num": declared_num,
            "matches": True,
            "has_mismatch": False,
            "mismatch_message": None,
            "issues": [],
            "warnings": [],
            "requires_review": False
        }

    # Compare with declared / printed USP
    matches = None
    has_mismatch = False
    mismatch_message = None

    if declared_num is not None and calculated_usp is not None:
        diff = abs(declared_num - calculated_usp)
        matches = diff <= max(0.02, calculated_usp * 0.02)
        if not matches:
            has_mismatch = True
            mismatch_message = f"USP mismatch — Inspector Review Required: Declared USP (₹{declared_num:.2f}) does not match calculated rate ({calculated_usp_formatted})."
            warnings.append(mismatch_message)

    return {
        "status": "CALCULATED",
        "applicable": True,
        "is_exempt": False,
        "is_exempt_equal_mrp": False,
        "is_small_pack_exempt": False,
        "exempt_reason": "",
        "expected_usp": calculated_usp,
        "expected_unit": calculated_unit,
        "calculated_usp": calculated_usp,
        "calculated_usp_formatted": calculated_usp_formatted,
        "calculated_unit": calculated_unit,
        "evidence_string": evidence_string,
        "declared_usp": declared_num,
        "printed_usp": declared_usp_str,
        "printed_usp_num": declared_num,
        "matches": matches,
        "has_mismatch": has_mismatch,
        "mismatch_message": mismatch_message,
        "issues": issues,
        "warnings": warnings,
        "requires_review": bool(has_mismatch or warnings)
    }


def validate_date(date_str: str) -> Dict[str, Any]:
    """
    Validates manufacturing or packing date formats (MM/YYYY or Month YYYY).
    Guards against impossible months and future calendar anomalies.
    """
    raw = (date_str or "").strip()
    if not raw:
        return {
            "valid": False,
            "parsed_month": None,
            "parsed_year": None,
            "issues": ["Missing manufacturing/packing date declaration."],
            "warnings": [],
            "requires_review": True
        }

    issues = []
    warnings = []
    month = None
    year = None

    # Check MM/YYYY or DD/MM/YYYY
    match_num = re.search(r"(?:(\d{1,2})[/-])?(\d{1,2})[/-](\d{4})", raw)
    if match_num:
        m_cand = int(match_num.group(2))
        y_cand = int(match_num.group(3))
        if 1 <= m_cand <= 12:
            month = m_cand
            year = y_cand
        else:
            issues.append(f"Invalid calendar month '{m_cand}' detected.")
    else:
        # Check month name e.g. "October 2025" or "Oct-2025"
        match_name = re.search(
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s/-]+(\d{4})",
            raw,
            re.IGNORECASE
        )
        if match_name:
            month_name = match_name.group(1).capitalize()
            try:
                dt = datetime.strptime(f"{month_name} {match_name.group(2)}", "%b %Y")
                month = dt.month
                year = dt.year
            except Exception:
                pass

    if not month or not year:
        warnings.append(f"Date declaration '{raw}' does not conform to standard Month/Year format (Rule 6(1)(d)).")
    else:
        # Guard against absurd years (e.g. earlier than 2000 or more than 5 years in future)
        current_year = datetime.now().year
        if year < 2000 or year > (current_year + 5):
            warnings.append(f"Suspicious declaration year '{year}' detected.")

    is_valid = bool(month and year and not issues)

    return {
        "valid": is_valid,
        "parsed_month": month,
        "parsed_year": year,
        "issues": issues,
        "warnings": warnings,
        "requires_review": bool(warnings or not is_valid)
    }


def validate_consumer_care(care_str: str) -> Dict[str, Any]:
    """
    Validates consumer care contact redressal mechanism under Rule 6(1)(e).
    Checks for phone, email, and postal contact points.
    """
    raw = (care_str or "").strip()
    if not raw:
        return {
            "valid": False,
            "has_phone": False,
            "has_email": False,
            "has_address_or_web": False,
            "channel_count": 0,
            "issues": ["Missing consumer grievance contact details."],
            "warnings": [],
            "requires_review": True
        }

    # Phone regex (Indian STD or mobile)
    has_phone = bool(re.search(r"\+?\d[\d\s\-]{8,}\d", raw))
    
    # Email regex (RFC 5322 compatible basic check)
    has_email = bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", raw))
    
    # Website or physical address keywords (excluding email tokens like care@)
    has_address_or_web = bool(re.search(r"(?:www\.|https?://|postal|address|grievance\s*cell|nodal\s*officer|pincode|\b\d{6}\b)", raw, re.IGNORECASE))

    channel_count = sum([has_phone, has_email, has_address_or_web])
    warnings = []

    if channel_count == 0:
        return {
            "valid": False,
            "has_phone": False,
            "has_email": False,
            "has_address_or_web": False,
            "channel_count": 0,
            "issues": ["No valid telephone number, email, or redressal address found in consumer care text."],
            "warnings": [],
            "requires_review": True
        }

    if channel_count == 1:
        warnings.append("Only a single contact channel detected. Legal Metrology mandates phone, email, or postal grievance access.")

    return {
        "valid": channel_count >= 1,
        "has_phone": has_phone,
        "has_email": has_email,
        "has_address_or_web": has_address_or_web,
        "channel_count": channel_count,
        "issues": [],
        "warnings": warnings,
        "requires_review": channel_count < 2
    }


def validate_manufacturer_address(mfg_str: str) -> Dict[str, Any]:
    """
    Validates completeness of manufacturer / packer name and postal address under Rule 6(1)(b).
    """
    raw = (mfg_str or "").strip()
    if not raw:
        return {
            "valid": False,
            "has_pincode": False,
            "has_locality": False,
            "issues": ["Missing manufacturer / packer name and address."],
            "warnings": [],
            "requires_review": True
        }

    has_pincode = bool(re.search(r"\b\d{6}\b", raw))
    locality_keywords = [
        "road", "rd", "street", "st", "lane", "nagar", "plot", "sector", "industrial",
        "post", "dist", "state", "mumbai", "delhi", "pune", "bengaluru", "chennai",
        "kolkata", "hyderabad", "ahmedabad", "gujarat", "maharashtra", "karnataka", "haryana"
    ]
    has_locality = any(k in raw.lower() for k in locality_keywords)
    warnings = []

    if len(raw) < 5:
        return {
            "valid": False,
            "has_pincode": False,
            "has_locality": False,
            "issues": ["Manufacturer declaration is too short to constitute a valid legal entity."],
            "warnings": [],
            "requires_review": True
        }

    if not has_pincode and not has_locality and len(raw) < 25:
        warnings.append("Address appears incomplete (no PIN code or postal locality identified).")

    return {
        "valid": True,
        "has_pincode": has_pincode,
        "has_locality": has_locality,
        "issues": [],
        "warnings": warnings,
        "requires_review": bool(warnings)
    }


def validate_all_fields(fields: Dict[str, str], fields_detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Executes end-to-end validation across all extracted fields.
    Produces validation status, normalized values, and inspector review warnings.
    Propagates extraction confidence and detects conflict warnings when fields_detail is provided.
    """
    fd = fields_detail or {}
    threshold = get_extraction_confidence_threshold()

    mrp_val = validate_mrp(fields.get("MRP", ""))
    qty_val = validate_net_quantity(fields.get("Net Quantity", ""), commodity_name=fields.get("Product Name", ""))
    usp_val = calculate_and_validate_usp(
        mrp_val.get("numeric_price"),
        qty_val.get("numeric_value"),
        qty_val.get("normalized_unit"),
        fields.get("Unit Sale Price"),
        raw_quantity_str=fields.get("Net Quantity", "")
    )
    date_val = validate_date(fields.get("Manufacturing Date", ""))
    care_val = validate_consumer_care(fields.get("Consumer Care", ""))
    mfg_val = validate_manufacturer_address(fields.get("Manufacturer", ""))

    # Propagate confidence, source, and conflict metadata from fields_detail
    field_components = [
        ("MRP", mrp_val),
        ("Net Quantity", qty_val),
        ("Unit Sale Price", usp_val),
        ("Manufacturing Date", date_val),
        ("Consumer Care", care_val),
        ("Manufacturer", mfg_val)
    ]

    for fname, fval in field_components:
        item = fd.get(fname, {}) if isinstance(fd.get(fname), dict) else {}
        conf = item.get("confidence")
        src = item.get("source", "Manual Input" if not fd else "Local OCR")
        has_conf = item.get("has_conflict", False)
        conf_detail = item.get("conflict_detail")

        fval["confidence"] = conf
        fval["source"] = src
        fval["has_conflict"] = has_conf
        fval["conflict_detail"] = conf_detail

        # If measurable confidence is below threshold, flag review warning
        if conf is not None and conf < threshold:
            fval["warnings"].append(
                f"Low extraction confidence ({conf:.1f}% < {threshold:.1f}% threshold). Inspector review required."
            )
            fval["requires_review"] = True

        # If extraction engines conflict, flag review warning
        if has_conf:
            fval["warnings"].append(
                f"Extraction discrepancy detected: {conf_detail}."
            )
            fval["requires_review"] = True

    all_issues = mrp_val["issues"] + qty_val["issues"] + date_val["issues"] + care_val["issues"] + mfg_val["issues"]
    all_warnings = mrp_val["warnings"] + qty_val["warnings"] + usp_val["warnings"] + date_val["warnings"] + care_val["warnings"] + mfg_val["warnings"]

    return {
        "is_valid": len(all_issues) == 0,
        "mrp": mrp_val,
        "quantity": qty_val,
        "unit_sale_price": usp_val,
        "date": date_val,
        "consumer_care": care_val,
        "manufacturer": mfg_val,
        "total_issues": len(all_issues),
        "total_warnings": len(all_warnings),
        "issues": all_issues,
        "warnings": all_warnings
    }
