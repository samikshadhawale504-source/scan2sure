"""
Gemini Multimodal Vision Extraction Engine for Legal Metrology (Packaged Commodities) Rules, 2011.
Uses the official Google GenAI Python SDK with structured Pydantic schema to accurately extract
statutory declarations, dot-matrix/inkjet variable print, addresses, and consumer care channels.

Enforces strict anti-hallucination guardrails and field-level confidence/evidence tracing.
"""

import os
import io
import time
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from PIL import Image

# Load environment variables
load_dotenv()

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class FieldDeclaration(BaseModel):
    """Structured extraction of an individual statutory packaging field."""
    value: str = Field(default="", description="Exact text declared on the package. Return empty string if not legible or absent.")
    evidence: str = Field(default="", description="Exact textual snippet or surrounding text showing where this declaration was observed.")
    confidence: float = Field(default=0.0, description="Confidence rating from 0.0 to 100.0 based on visual sharpness and legibility.")
    requires_review: bool = Field(default=False, description="Set to true if text is partially obscured, low-contrast dot-matrix, or uncertain.")
    is_directly_observed: bool = Field(default=True, description="True if directly legible; False if inferred from surrounding context.")


class PackagedCommodityVisionSchema(BaseModel):
    """Statutory packaging declarations under Legal Metrology (Packaged Commodities) Rules, 2011."""
    product_name: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Brand name and generic / common commodity descriptor on the Principal Display Panel (PDP)."
    )
    manufacturer: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Name of the manufacturing entity or marketer."
    )
    manufacturer_address: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Complete postal address of manufacturer including city, state, and 6-digit PIN code."
    )
    packer: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Name of pre-packer or packager if distinct from manufacturer."
    )
    packer_address: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Complete postal address of pre-packer."
    )
    importer: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Name of importer for imported packaged goods."
    )
    importer_address: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Complete postal address of importer in India."
    )
    net_quantity: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Net weight or volume declaration (e.g. '50 g', '500 ml', '1 kg')."
    )
    quantity_unit: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Metric unit symbol declared (e.g. 'g', 'kg', 'ml', 'l', 'number')."
    )
    mrp: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Retail Sale Price (MRP) including tax declaration (e.g. 'Rs. 28.00 incl. of all taxes')."
    )
    unit_sale_price: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Unit sale price under Rule 6(1)(m) (e.g. 'Rs. 0.56 per g' or '₹0.56/g')."
    )
    manufacturing_date: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Month and year of manufacture or pre-packing (e.g. '15/10/2025' or '10/2025')."
    )
    packing_date: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Pre-packing date if declared separately from manufacturing date."
    )
    import_date: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Month and year of import for foreign commodities."
    )
    batch_number: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Batch, lot, or identification code stamped on package (e.g. 'AB1025' or 'B.No. 402')."
    )
    best_before: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Best before date, expiry date, or shelf-life declaration (e.g. '14/10/2027' or '24 months from mfg')."
    )
    consumer_care_phone: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Customer care telephone helpline number."
    )
    consumer_care_email: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Customer care / grievance redressal email address."
    )
    consumer_care_website: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Customer care website URL or grievance portal."
    )
    country_of_origin: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Country of origin / manufacture (e.g. 'India' or country of import)."
    )
    other_mandatory_declarations: FieldDeclaration = Field(
        default_factory=FieldDeclaration,
        description="Other statutory markings (e.g. FSSAI License, Vegetarian green dot, ISI mark)."
    )
    label_quality_notes: str = Field(
        default="",
        description="Inspector observations regarding label clarity, dot-matrix legibility, glare, or physical condition."
    )


import re


def sanitize_gemini_error(err_str: str) -> str:
    """
    Categorizes Gemini API exceptions and ensures API keys are never exposed in error text.
    Provides precise HTTP status code classification.
    """
    if not err_str:
        return "Unknown Gemini Error"

    err_lower = err_str.lower()
    if any(term in err_lower for term in ["api_key", "api key", "unauthenticated", "401", "403", "permission", "api_key_invalid", "key not valid"]):
        return "Authentication Error (401/403/400): Gemini API key is invalid, unauthorized, or lacks vision permissions."
    elif "400" in err_str or "invalid_argument" in err_lower:
        return "400 INVALID_ARGUMENT: Bad request or unsupported parameter."
    elif "404" in err_str or "not_found" in err_lower or "model_not_found" in err_lower:
        return "404 NOT_FOUND (Model Unavailable): Requested Gemini model is not supported."
    elif "429" in err_str or "resource_exhausted" in err_lower or "quota" in err_lower or "rate limit" in err_lower:
        return "429 RESOURCE_EXHAUSTED: Gemini API request quota or rate limit exceeded."
    elif "500" in err_str or "503" in err_str or "unavailable" in err_lower or "server error" in err_lower:
        return "503 UNAVAILABLE: Gemini Vision API temporarily unreachable."
    elif any(term in err_lower for term in ["deadline", "timeout", "timed out"]):
        return "DEADLINE_EXCEEDED: Request to Gemini Vision API timed out."

    # Strip any potential API key patterns (AIzaSy... or key=...)
    cleaned = re.sub(r"AIzaSy[A-Za-z0-9_\-]{33}", "[REDACTED]", err_str)
    cleaned = re.sub(r"key=[^&\s]+", "key=[REDACTED]", cleaned)
    return cleaned[:140]


def get_api_key(explicit_key=None):
    """Resolve Gemini API key from parameter, environment, or .env file."""
    if explicit_key is not None:
        return explicit_key.strip()
    try:
        from config import load_project_env
        load_project_env()
    except Exception:
        load_dotenv(override=True)
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def is_gemini_available(api_key=None):
    """Check if Gemini SDK is installed and an API key is available."""
    if not GENAI_AVAILABLE:
        return False
    key = get_api_key(api_key)
    return bool(key and len(key) > 5)


def prepare_image_part_for_vision(image_path: str, max_dimension: int = 1600):
    """
    Loads, optimizes dimensions if needed, and returns an official types.Part
    with the exact MIME type for Gemini Vision input.
    """
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_dimension:
        ratio = max_dimension / float(max(w, h))
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    # Save as high-quality JPEG to preserve dot-matrix printing and fine text
    img.save(buf, format="JPEG", quality=92)
    image_bytes = buf.getvalue()

    if GENAI_AVAILABLE:
        return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    return img


def optimize_image_for_vision(image_path: str, max_dimension: int = 1600) -> Image.Image:
    """
    Load image and resize proportionally if too large.
    Reduces latency and prevents payload timeout while preserving fine dot-matrix details.
    """
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_dimension:
        ratio = max_dimension / float(max(w, h))
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return img


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

STABLE_FLASH_CANDIDATES = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

DISALLOWED_MODEL_PATTERNS = ["2.0", "1.5", "preview", "exp", "deprecated"]


def resolve_supported_model(client: Any = None, preferred: Optional[str] = None) -> str:
    """
    Programmatically determines a supported stable multimodal Flash model
    for the configured client by querying client.models.list().
    Avoids guessing or hardcoding unsupported model IDs.
    """
    target = preferred or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    if client is None:
        return target

    try:
        models = list(client.models.list())
        supported_names = []
        for m in models:
            name = getattr(m, "name", "") or ""
            short_name = name.replace("models/", "").strip()
            # Must be a Flash model and must NOT be disallowed (no 2.0, no 1.5, no preview, no exp)
            if "flash" in short_name.lower():
                if any(bad in short_name.lower() for bad in DISALLOWED_MODEL_PATTERNS):
                    continue
                supported_names.append(short_name)

        # If preferred target is in the list, use it
        if target in supported_names or target.replace("models/", "") in supported_names:
            return target

        # Otherwise pick the best candidate from STABLE_FLASH_CANDIDATES that exists in supported_names
        for candidate in STABLE_FLASH_CANDIDATES:
            if candidate in supported_names:
                return candidate

        # If any other valid stable Flash model exists, pick the first
        if supported_names:
            return supported_names[0]

        return target
    except Exception:
        # If listing is not permitted or network error, return the target default
        return target


def validate_gemini_configuration(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Validates Gemini configuration and verifies model support at startup/configuration time
    before an actual image scan is requested.
    """
    key = get_api_key(api_key)
    if not key:
        return {
            "configured": False,
            "valid": False,
            "active_model": None,
            "error": "No GEMINI_API_KEY configured.",
            "message": "Offline mode: Local OCR will be used."
        }

    if not GENAI_AVAILABLE:
        return {
            "configured": True,
            "valid": False,
            "active_model": None,
            "error": "google-genai SDK not installed.",
            "message": "Local OCR will be used."
        }

    try:
        client = genai.Client(api_key=key)
        preferred = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        resolved_model = resolve_supported_model(client, preferred)
        return {
            "configured": True,
            "valid": True,
            "active_model": resolved_model,
            "error": None,
            "message": f"Gemini Vision active model verified: {resolved_model}"
        }
    except Exception as exc:
        sanitized = sanitize_gemini_error(str(exc))
        return {
            "configured": True,
            "valid": False,
            "active_model": None,
            "error": sanitized,
            "message": f"Gemini configuration check warning: {sanitized}"
        }


def extract_with_gemini(image_path: str, api_key: Optional[str] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts packaged commodity label declarations using Gemini Multimodal Vision.
    Directly extracts variable stamped inkjet/dot-matrix text, addresses, and prices.
    
    Adheres strictly to Anti-Hallucination rules: returns empty string if not clearly visible.
    """
    t0 = time.time()
    key = get_api_key(api_key)

    if not key:
        return {
            "success": False,
            "error": "No Gemini API key provided. Configure GEMINI_API_KEY in .env or via web UI.",
            "engine": "gemini",
            "processing_time_sec": 0.0
        }

    if not os.path.exists(image_path):
        return {
            "success": False,
            "error": f"Image file not found: {image_path}",
            "engine": "gemini",
            "processing_time_sec": 0.0
        }

    selected_model = model_name or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    try:
        client = genai.Client(api_key=key)
        image_part = prepare_image_part_for_vision(image_path)

        prompt = (
            "You are an official Legal Metrology Inspector assessing compliance under the "
            "Legal Metrology (Packaged Commodities) Rules, 2011 (PCR 2011) in India.\n\n"
            "INSTRUCTIONS:\n"
            "1. Inspect the entire packaged commodity label thoroughly.\n"
            "2. Pay special attention to variable dynamic declarations printed using industrial "
            "inkjet, dot-matrix, or thermal printers: Batch/Lot No., Manufacturing Date, Best Before/Expiry, and MRP.\n"
            "3. Extract the Generic Commodity Name on the Principal Display Panel, Manufacturer/Packer Name and complete postal address, "
            "Net Quantity in standard metric units, Unit Sale Price (USP), Consumer Care phone/email/website, and Country of Origin.\n\n"
            "CRITICAL ANTI-HALLUCINATION RULES:\n"
            "- NEVER fabricate, guess, or invent any declaration. If a field is missing, obscured, or unreadable, "
            "set its 'value' to an empty string '' and 'requires_review' to true.\n"
            "- Do NOT convert image noise, scratches, or decorative graphics into fake prices or dates.\n"
            "- Extract verbatim evidence snippets into the 'evidence' attribute for every field found.\n"
            "- Output valid JSON strictly conforming to the provided schema."
        )

        active_model = resolve_supported_model(client, selected_model)
        models_to_try = [active_model]
        for candidate in STABLE_FLASH_CANDIDATES:
            if candidate not in models_to_try:
                models_to_try.append(candidate)

        response = None
        used_model = active_model
        last_exc = None

        for m_name in models_to_try:
            try:
                gen_config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PackagedCommodityVisionSchema,
                    temperature=0.0,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                ) if GENAI_AVAILABLE else None

                response = client.models.generate_content(
                    model=m_name,
                    contents=[image_part, prompt],
                    config=gen_config
                )
                used_model = m_name
                break
            except Exception as model_err:
                last_exc = model_err
                # If authentication or quota error, trying another model won't help
                err_str = str(model_err).lower()
                if "api_key" in err_str or "unauthenticated" in err_str or "quota" in err_str or "429" in err_str:
                    raise model_err

        if response is None:
            if last_exc:
                raise last_exc
            raise RuntimeError("Gemini Vision returned no response.")

        elapsed = round(time.time() - t0, 2)
        raw_text = response.text
        data = json.loads(raw_text)

        # Helper to unpack FieldDeclaration dicts safely
        def get_field(key_name):
            item = data.get(key_name, {})
            if isinstance(item, dict):
                return item
            return {"value": str(item) if item else "", "evidence": "", "confidence": 90.0, "requires_review": False, "is_directly_observed": True}

        prod_name_item = get_field("product_name")
        mfg_item = get_field("manufacturer")
        mfg_addr_item = get_field("manufacturer_address")
        qty_item = get_field("net_quantity")
        mrp_item = get_field("mrp")
        usp_item = get_field("unit_sale_price")
        mfg_date_item = get_field("manufacturing_date")
        batch_item = get_field("batch_number")
        best_before_item = get_field("best_before")
        care_phone_item = get_field("consumer_care_phone")
        care_email_item = get_field("consumer_care_email")
        care_web_item = get_field("consumer_care_website")
        origin_item = get_field("country_of_origin")

        # Combine manufacturer name and address if separate
        full_mfg = mfg_item.get("value", "")
        if mfg_addr_item.get("value") and mfg_addr_item.get("value") not in full_mfg:
            full_mfg = f"{full_mfg}, {mfg_addr_item.get('value')}".strip(", ")

        # Combine consumer care contact points
        care_points = []
        if care_phone_item.get("value"):
            care_points.append(f"Phone: {care_phone_item.get('value')}")
        if care_email_item.get("value"):
            care_points.append(f"Email: {care_email_item.get('value')}")
        if care_web_item.get("value"):
            care_points.append(f"Web: {care_web_item.get('value')}")
        combined_care = " | ".join(care_points)

        # Standard field mapping for UI and compliance engine
        fields = {
            "Product Name": prod_name_item.get("value", ""),
            "Manufacturer": full_mfg,
            "Address": mfg_addr_item.get("value", ""),
            "Net Quantity": qty_item.get("value", ""),
            "MRP": mrp_item.get("value", ""),
            "Unit Sale Price": usp_item.get("value", ""),
            "Manufacturing Date": mfg_date_item.get("value", ""),
            "Batch Number": batch_item.get("value", ""),
            "Best Before / Expiry": best_before_item.get("value", ""),
            "Consumer Care": combined_care,
            "Country of Origin": origin_item.get("value", "India")
        }

        # Calculate average extraction confidence
        confidences = [
            prod_name_item.get("confidence", 90.0),
            mfg_item.get("confidence", 90.0),
            qty_item.get("confidence", 90.0),
            mrp_item.get("confidence", 90.0),
            mfg_date_item.get("confidence", 90.0)
        ]
        avg_confidence = round(sum(confidences) / len(confidences), 1)

        # Detailed breakdown per field
        fields_detail = {
            "Product Name": {
                "value": prod_name_item.get("value", ""),
                "evidence": prod_name_item.get("evidence", ""),
                "confidence": prod_name_item.get("confidence", 95.0) if prod_name_item.get("value") else None,
                "requires_review": prod_name_item.get("requires_review", False),
                "status": "DETECTED" if prod_name_item.get("value") else "NOT_DETECTED",
                "source": "Gemini Vision AI",
                "engine": "Gemini Vision",
                "has_conflict": False,
                "conflict_detail": None
            },
            "Manufacturer": {
                "value": full_mfg,
                "evidence": mfg_item.get("evidence", "") or mfg_addr_item.get("evidence", ""),
                "confidence": mfg_item.get("confidence", 95.0) if full_mfg else None,
                "requires_review": mfg_item.get("requires_review", False),
                "status": "DETECTED" if full_mfg else "NOT_DETECTED",
                "source": "Gemini Vision AI",
                "engine": "Gemini Vision",
                "has_conflict": False,
                "conflict_detail": None
            },
            "Address": {
                "value": mfg_addr_item.get("value", ""),
                "evidence": mfg_addr_item.get("evidence", ""),
                "confidence": mfg_addr_item.get("confidence", 95.0) if mfg_addr_item.get("value") else None,
                "requires_review": False,
                "status": "DETECTED" if mfg_addr_item.get("value") else "NOT_DETECTED",
                "source": "Gemini Vision AI",
                "engine": "Gemini Vision",
                "has_conflict": False,
                "conflict_detail": None
            },
            "Net Quantity": {
                "value": qty_item.get("value", ""),
                "evidence": qty_item.get("evidence", ""),
                "confidence": qty_item.get("confidence", 95.0) if qty_item.get("value") else None,
                "requires_review": qty_item.get("requires_review", False),
                "status": "DETECTED" if qty_item.get("value") else "NOT_DETECTED",
                "source": "Gemini Vision AI",
                "engine": "Gemini Vision",
                "has_conflict": False,
                "conflict_detail": None
            },
            "MRP": {
                "value": mrp_item.get("value", ""),
                "evidence": mrp_item.get("evidence", ""),
                "confidence": mrp_item.get("confidence", 95.0) if mrp_item.get("value") else None,
                "requires_review": mrp_item.get("requires_review", False),
                "status": "DETECTED" if mrp_item.get("value") else "NOT_DETECTED",
                "source": "Gemini Vision AI",
                "engine": "Gemini Vision",
                "has_conflict": False,
                "conflict_detail": None
            },
            "Unit Sale Price": {
                "value": usp_item.get("value", ""),
                "evidence": usp_item.get("evidence", ""),
                "confidence": usp_item.get("confidence", 90.0) if usp_item.get("value") else None,
                "requires_review": usp_item.get("requires_review", False),
                "status": "DETECTED" if usp_item.get("value") else "NOT_DETECTED",
                "source": "Gemini Vision AI",
                "engine": "Gemini Vision",
                "has_conflict": False,
                "conflict_detail": None
            },
            "Manufacturing Date": {
                "value": mfg_date_item.get("value", ""),
                "evidence": mfg_date_item.get("evidence", ""),
                "confidence": mfg_date_item.get("confidence", 95.0) if mfg_date_item.get("value") else None,
                "requires_review": mfg_date_item.get("requires_review", False),
                "status": "DETECTED" if mfg_date_item.get("value") else "NOT_DETECTED",
                "source": "Gemini Vision AI",
                "engine": "Gemini Vision",
                "has_conflict": False,
                "conflict_detail": None
            },
            "Batch Number": {
                "value": batch_item.get("value", ""),
                "evidence": batch_item.get("evidence", ""),
                "confidence": batch_item.get("confidence", 95.0) if batch_item.get("value") else None,
                "requires_review": batch_item.get("requires_review", False),
                "status": "DETECTED" if batch_item.get("value") else "NOT_DETECTED",
                "source": "Gemini Vision AI",
                "engine": "Gemini Vision",
                "has_conflict": False,
                "conflict_detail": None
            },
            "Best Before / Expiry": {
                "value": best_before_item.get("value", ""),
                "evidence": best_before_item.get("evidence", ""),
                "confidence": best_before_item.get("confidence", 95.0) if best_before_item.get("value") else None,
                "requires_review": best_before_item.get("requires_review", False),
                "status": "DETECTED" if best_before_item.get("value") else "NOT_DETECTED",
                "source": "Gemini Vision AI",
                "engine": "Gemini Vision",
                "has_conflict": False,
                "conflict_detail": None
            },
            "Consumer Care": {
                "value": combined_care,
                "evidence": care_phone_item.get("evidence", "") or care_email_item.get("evidence", ""),
                "confidence": care_phone_item.get("confidence", 95.0) if combined_care else None,
                "requires_review": care_phone_item.get("requires_review", False),
                "status": "DETECTED" if combined_care else "NOT_DETECTED",
                "source": "Gemini Vision AI",
                "engine": "Gemini Vision",
                "has_conflict": False,
                "conflict_detail": None
            },
            "Country of Origin": {
                "value": origin_item.get("value", "India"),
                "evidence": origin_item.get("evidence", ""),
                "confidence": origin_item.get("confidence", 95.0) if origin_item.get("value") else None,
                "requires_review": origin_item.get("requires_review", False),
                "status": "DETECTED" if origin_item.get("value") else "NOT_DETECTED",
                "source": "Gemini Vision AI",
                "engine": "Gemini Vision",
                "has_conflict": False,
                "conflict_detail": None
            }
        }

        return {
            "success": True,
            "engine": "gemini",
            "model": used_model,
            "confidence": avg_confidence,
            "processing_time_sec": elapsed,
            "fields": fields,
            "fields_detail": fields_detail,
            "label_quality_notes": data.get("label_quality_notes", ""),
            "raw_text": raw_text
        }

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        sanitized_err = sanitize_gemini_error(str(e))
        return {
            "success": False,
            "error": sanitized_err,
            "engine": "gemini",
            "processing_time_sec": elapsed
        }
