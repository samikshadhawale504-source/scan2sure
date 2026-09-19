import re


def parse_price(value):
    """Extract numeric price from a text value."""

    if not value:
        return None

    cleaned = value.replace(",", "")

    match = re.search(r"\d+(?:\.\d+)?", cleaned)

    if match:
        return float(match.group())

    return None


def parse_unit_price(value):
    """
    Extract numeric unit price and declared unit.

    Examples:
    0.56
    ₹0.56/g
    Rs. 0.56 per g
    0.56 per gram
    """

    if not value:
        return None, None

    price = parse_price(value)

    if price is None:
        return None, None

    text = value.lower()

    unit = None

    if re.search(r"\bkg\b|kilogram", text):
        unit = "kg"

    elif re.search(r"\bg\b|gram", text):
        unit = "g"

    elif re.search(r"\bml\b|millilitre|milliliter", text):
        unit = "ml"

    elif re.search(r"\bl\b|litre|liter", text):
        unit = "l"

    elif re.search(r"\bcm\b|centimeter|centimetre", text):
        unit = "cm"

    elif re.search(r"\bm\b|meter|metre", text):
        unit = "m"

    elif re.search(r"\b(number|no\.?|unit|piece|pcs)\b", text):
        unit = "number"

    return price, unit


def parse_quantity(value):
    """Extract quantity and unit."""

    if not value:
        return None, None

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(mg|g|kg|ml|l|cm|mm|m)\b",
        value,
        re.IGNORECASE
    )

    if not match:
        return None, None

    number = float(match.group(1))
    unit = match.group(2).lower()

    return number, unit


def calculate_expected_unit_price(mrp, quantity, unit):
    """
    Calculate expected unit sale price.

    Returns:
        expected price
        expected unit
    """

    if mrp is None or quantity is None or unit is None:
        return None, None

    # -------------------------
    # WEIGHT
    # -------------------------

    if unit == "mg":

        grams = quantity / 1000

        if grams < 1000:
            return round(mrp / grams, 2), "g"

    elif unit == "g":

        if quantity < 1000:
            return round(mrp / quantity, 2), "g"

        return round(mrp / (quantity / 1000), 2), "kg"

    elif unit == "kg":

        return round(mrp / quantity, 2), "kg"

    # -------------------------
    # VOLUME
    # -------------------------

    elif unit == "ml":

        if quantity < 1000:
            return round(mrp / quantity, 2), "ml"

        return round(mrp / (quantity / 1000), 2), "l"

    elif unit == "l":

        return round(mrp / quantity, 2), "l"

    # -------------------------
    # LENGTH
    # -------------------------

    elif unit == "mm":

        cm = quantity / 10

        if cm < 100:
            return round(mrp / cm, 2), "cm"

        return round(mrp / (cm / 100), 2), "m"

    elif unit == "cm":

        if quantity < 100:
            return round(mrp / quantity, 2), "cm"

        return round(mrp / (quantity / 100), 2), "m"

    elif unit == "m":

        return round(mrp / quantity, 2), "m"

    return None, None


def check_compliance(fields):

    issues = []
    warnings = []

    # ==================================================
    # 1. PRODUCT NAME
    # ==================================================

    if not fields.get("Product Name", "").strip():

        issues.append({
            "field": "Product Name",
            "message": "Common/generic name of the commodity is missing."
        })

    # ==================================================
    # 2. MANUFACTURER
    # ==================================================

    manufacturer = fields.get("Manufacturer", "").strip()

    if not manufacturer:

        issues.append({
            "field": "Manufacturer",
            "message": "Manufacturer / packer / importer details are missing."
        })

    # ==================================================
    # 3. NET QUANTITY
    # ==================================================

    quantity_text = fields.get("Net Quantity", "").strip()

    quantity = None
    quantity_unit = None

    if not quantity_text:

        issues.append({
            "field": "Net Quantity",
            "message": "Net quantity is missing."
        })

    else:

        quantity, quantity_unit = parse_quantity(quantity_text)

        if quantity is None:

            warnings.append({
                "field": "Net Quantity",
                "message": "Net quantity was detected, but its unit could not be verified."
            })

    # ==================================================
    # 4. MRP
    # ==================================================

    mrp_text = fields.get("MRP", "").strip()

    mrp = None

    if not mrp_text:

        issues.append({
            "field": "MRP",
            "message": "Maximum Retail Price (MRP) is missing."
        })

    else:

        mrp = parse_price(mrp_text)

        if mrp is None:

            warnings.append({
                "field": "MRP",
                "message": "MRP was detected, but the price format could not be verified."
            })

    # ==================================================
    # 5. UNIT SALE PRICE
    # ==================================================

    unit_price_text = fields.get("Unit Sale Price", "").strip()

    declared_unit_price = None
    declared_unit = None

    if not unit_price_text:

        issues.append({
            "field": "Unit Sale Price",
            "message": "Unit sale price is missing."
        })

    else:

        declared_unit_price, declared_unit = parse_unit_price(
            unit_price_text
        )

        if declared_unit_price is None:

            warnings.append({
                "field": "Unit Sale Price",
                "message": "Unit sale price could not be read as a valid numeric value."
            })

    # ==================================================
    # 6. CALCULATE EXPECTED UNIT SALE PRICE
    # ==================================================

    expected_price = None
    expected_unit = None

    if (
        mrp is not None
        and quantity is not None
        and quantity_unit is not None
    ):

        expected_price, expected_unit = calculate_expected_unit_price(
            mrp,
            quantity,
            quantity_unit
        )
    unit_price_mismatch = False
    unit_price_unit_mismatch = False
    # ==================================================
    # 7. COMPARE DECLARED VS CALCULATED VALUE
    # ==================================================

    if (
       expected_price is not None
       and declared_unit_price is not None
    ):

    # Check numeric value
# Check numeric value
        if round(declared_unit_price, 2) != round(expected_price, 2):

            unit_price_mismatch = True

            warnings.append({
                "field": "Unit Sale Price",
                "message": (
                    f"Declared value: ₹{declared_unit_price:.2f}/{expected_unit}. "
                    f"Expected value from MRP and net quantity: "
                    f"₹{expected_price:.2f}/{expected_unit}."
                )
            })

        # Check unit when explicitly provided
        elif declared_unit is not None and declared_unit != expected_unit:

            unit_price_unit_mismatch = True

            warnings.append({
                "field": "Unit Sale Price",
                "message": (
                    f"The declared unit is '{declared_unit}', "
                    f"but the expected unit for this package is "
                    f"'{expected_unit}'."
                )
            })
    # ==================================================
    # 8. MANUFACTURING DATE
    # ==================================================

    mfg_date = fields.get("Manufacturing Date", "").strip()

    if not mfg_date:

        issues.append({
            "field": "Manufacturing Date",
            "message": "Manufacturing / packing / import month and year are missing."
        })

    elif not re.search(
        r"(0?[1-9]|1[0-2])[/-]\d{4}",
        mfg_date
    ):

        warnings.append({
            "field": "Manufacturing Date",
            "message": "Manufacturing date format could not be verified."
        })

    # ==================================================
    # 9. BEST BEFORE / EXPIRY
    # ==================================================

    expiry = fields.get("Best Before / Expiry", "").strip()

    if not expiry:

        warnings.append({
            "field": "Best Before / Expiry",
            "message": (
                "Best-before/use-by information was not detected. "
                "Verify whether it is applicable."
            )
        })

    # ==================================================
    # 10. CONSUMER CARE
    # ==================================================

    consumer = fields.get("Consumer Care", "").strip()

    if not consumer:

        issues.append({
            "field": "Consumer Care",
            "message": "Consumer care details are missing."
        })

    else:

        has_phone = bool(
            re.search(r"\+?\d[\d\s\-]{8,}", consumer)
        )

        has_email = bool(
            re.search(r"[\w\.-]+@[\w\.-]+\.\w+", consumer)
        )

        if not has_phone and not has_email:

            warnings.append({
                "field": "Consumer Care",
                "message": (
                    "Consumer care information was detected, "
                    "but a valid contact channel could not be verified."
                )
            })

    # ==================================================
    # 11. COUNTRY OF ORIGIN
    # ==================================================

    country = fields.get("Country of Origin", "").strip()

    if country:
        pass

    # ==================================================
    # FINAL STATUS
    # ==================================================

    if issues:

        status = "NON-COMPLIANT"

    elif warnings:

        status = "NEEDS REVIEW"

    else:

        status = "COMPLIANT"

    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,

        # New information for the UI
        "unit_price": {
            "declared": declared_unit_price,
            "declared_unit": declared_unit,
            "expected": expected_price,
            "expected_unit": expected_unit
        }
    }