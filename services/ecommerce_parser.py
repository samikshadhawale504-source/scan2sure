"""
E-Commerce and Product Information Parsing Service.
Normalizes raw e-commerce product listings, metadata descriptions, and structured text
into the standardized Legal Metrology packaging declaration schema.
"""

import re
from typing import Dict, Any, Optional


def parse_ecommerce_listing(text_content: str, source_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses unformatted product listing text or pasted product specifications.
    Extracts statutory Legal Metrology declarations and maps them into the standard schema.
    """
    text = (text_content or "").strip()
    if not text:
        return {
            "success": False,
            "error": "No text provided for e-commerce listing parsing.",
            "fields": {}
        }

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # 1. Product Name: Typically first line or line labeled "Product Name / Title"
    product_name = ""
    for line in lines:
        match = re.search(r"(?:title|product(?:\s*name)?|item\s*name)\s*[:\-]\s*(.+)", line, re.IGNORECASE)
        if match:
            product_name = match.group(1).strip()
            break
    if not product_name and lines:
        product_name = lines[0]

    # 2. Net Quantity
    net_quantity = ""
    qty_match = re.search(
        r"(?:net\s*(?:wt\.?|weight|qty\.?|quantity)|size|volume)\s*[:\-]?\s*(\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|cm|m|units?|pieces?|pcs|number|count)\b)",
        text,
        re.IGNORECASE
    )
    if qty_match:
        net_quantity = qty_match.group(1).strip()
    else:
        # Fallback search for standalone quantity
        standalone_qty = re.search(r"\b(\d+(?:\.\d+)?\s*(?:g|kg|ml|l)\b)", text, re.IGNORECASE)
        if standalone_qty:
            net_quantity = standalone_qty.group(1).strip()

    # 3. MRP
    mrp = ""
    mrp_match = re.search(
        r"(?:mrp|maximum\s*retail\s*price|price)\s*[:\-]?\s*(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)(.*)",
        text,
        re.IGNORECASE
    )
    if mrp_match:
        price_num = mrp_match.group(1).replace(",", "")
        trailing = mrp_match.group(2)
        has_tax = "tax" in trailing.lower() or "incl" in trailing.lower() or "tax" in text.lower()
        mrp = f"₹{price_num} (incl. of all taxes)" if has_tax else f"₹{price_num}"

    # 4. Unit Sale Price
    usp = ""
    usp_match = re.search(r"(?:unit\s*(?:sale\s*)?price|usp)\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if usp_match:
        usp = usp_match.group(1).strip()

    # 5. Manufacturer / Packer
    manufacturer = ""
    mfg_match = re.search(
        r"(?:manufactured\s*(?:by|and\s*packed\s*by)|manufacturer|packer|marketer)\s*[:\-]?\s*(.+)",
        text,
        re.IGNORECASE
    )
    if mfg_match:
        manufacturer = mfg_match.group(1).strip()

    # 6. Dates
    mfg_date = ""
    mfg_date_match = re.search(
        r"(?:mfg\.?\s*date|date\s*of\s*manufacture|packed\s*on|pkd)\s*[:\-]?\s*([0-9A-Za-z/.\-]+)",
        text,
        re.IGNORECASE
    )
    if mfg_date_match:
        mfg_date = mfg_date_match.group(1).strip()

    expiry = ""
    exp_match = re.search(
        r"(?:best\s*before|expiry|use\s*by|shelf\s*life)\s*[:\-]?\s*(.+)",
        text,
        re.IGNORECASE
    )
    if exp_match:
        expiry = exp_match.group(1).split("\n")[0].strip()

    # 7. Batch Number
    batch = ""
    batch_match = re.search(r"(?:batch\s*(?:no\.?|number)|lot\s*(?:no\.?|number))\s*[:\-]?\s*([A-Za-z0-9\-]+)", text, re.IGNORECASE)
    if batch_match:
        batch = batch_match.group(1).strip()

    # 8. Consumer Care
    care_parts = []
    phone_match = re.search(r"(?:phone|helpline|tel|toll[\s\-]free)\s*[:\-]?\s*(\+?\d[\d\s\-]{8,}\d)", text, re.IGNORECASE)
    if phone_match:
        care_parts.append(f"Phone: {phone_match.group(1).strip()}")
    email_match = re.search(r"(?:email|e[\s\-]mail|care\s*email)\s*[:\-]?\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text, re.IGNORECASE)
    if email_match:
        care_parts.append(f"Email: {email_match.group(1).strip()}")
    consumer_care = " | ".join(care_parts) if care_parts else ""

    # 9. Country of Origin
    origin = "India"
    origin_match = re.search(r"(?:country\s*of\s*origin|origin)\s*[:\-]?\s*([A-Za-z\s]+)", text, re.IGNORECASE)
    if origin_match:
        origin = origin_match.group(1).split("\n")[0].strip()

    fields = {
        "Product Name": product_name,
        "Manufacturer": manufacturer,
        "Net Quantity": net_quantity,
        "MRP": mrp,
        "Unit Sale Price": usp,
        "Manufacturing Date": mfg_date,
        "Batch Number": batch,
        "Best Before / Expiry": expiry,
        "Consumer Care": consumer_care,
        "Country of Origin": origin
    }

    fields_detail = {}
    for k, v in fields.items():
        fields_detail[k] = {
            "value": v,
            "evidence": f"Parsed from listing text: '{v[:30]}...'" if len(v) > 30 else f"Parsed from listing text: '{v}'" if v else "",
            "confidence": None,
            "requires_review": not bool(v),
            "source": "E-Commerce Listing",
            "has_conflict": False,
            "conflict_detail": None
        }

    return {
        "success": True,
        "source": "ecommerce",
        "fields": fields,
        "fields_detail": fields_detail
    }
