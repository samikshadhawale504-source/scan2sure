import os
import re
import sys
import time

import cv2
import numpy as np
import pytesseract


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

MONTH_PATTERN = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

VALID_SHORT_WORDS = {
    "kg", "ml", "mg", "cm", "no", "by", "at",
    "in", "on", "to", "of", "rs", "or", "is", "as", "do"
}


# ============================================================
# TEXT CLEANING & GARBAGE FILTERING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    # Strip markdown image tags (e.g. ![img-0.jpeg](img-0.jpeg)) and image references
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[img-\d+.*?\]", " ", text)
    text = re.sub(r"\b1?img-\d+\b", " ", text, flags=re.I)

    text = text.replace("\x0c", " ")
    text = text.replace("\r", "\n")

    # Normalize horizontal whitespace without destroying line structure
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def normalize_for_comparison(text):
    text = clean_text(text).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def looks_like_garbage(line):
    """
    Reject obvious OCR garbage noise while keeping legitimate
    packaging text containing numbers, units, and symbols.
    """
    line = clean_text(line)
    if not line:
        return True

    core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", line)

    # Isolated 0 or 1 character lines are noise
    if len(core) <= 1:
        return True

    # 2-character lines must be valid words or digits
    if len(core) == 2:
        if core.isdigit() or core.lower() in VALID_SHORT_WORDS:
            return False
        return True

    # If the line contains statutory packaging declarations, preserve it
    lower = line.lower()
    statutory_hints = [
        "weight", "wetweight", "netweight", "mrp", "mfg", "mfd", "batch",
        "expiry", "best before", "use by", "net", "usp", "price", "packag"
    ]
    if any(h in lower for h in statutory_hints):
        return False
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:g|kg|ml|l|mg)\b", lower):
        return False
    if re.search(r"(?:₹|rs\.?|inr|[F€])\s*\d+", line):
        return False

    # Alphanumeric density check
    alnum = sum(ch.isalnum() for ch in line)
    if alnum / max(len(line), 1) < 0.45:
        return True

    # Clean words (strip punctuation) for word ratio check
    words = [re.sub(r"[^A-Za-z0-9]", "", w) for w in line.split()]
    words = [w for w in words if w]
    if not words:
        return True
    single_char_words = sum(1 for w in words if len(w) == 1)
    if len(words) >= 3 and single_char_words / len(words) > 0.50:
        return True

    # Pure symbol noise
    if re.fullmatch(r"[_|~`^.,:;/'\\\-+=<>!?@#$%&*()]+", line):
        return True

    return False


def clean_ocr_lines(text):
    """
    Clean OCR output while preserving useful packaging lines.
    """
    lines = []
    seen = set()

    for raw_line in text.splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        if looks_like_garbage(line):
            continue

        key = normalize_for_comparison(line)
        if not key:
            continue
        if key in seen:
            continue

        seen.add(key)
        lines.append(line)

    return "\n".join(lines)


# ============================================================
# ORIENTATION CORRECTION
# ============================================================

def detect_and_correct_orientation(image):
    """
    Try orientation detection with strict confidence verification.
    Requires Tesseract OSD orientation confidence >= 3.0 to prevent flipping upright images.
    If OSD confidence is low (< 3.0) or image is small, retains original image.
    """
    try:
        h, w = image.shape[:2]
        if h < 200 or w < 200:
            return image

        scale = min(1.0, 1200 / max(h, w))
        if scale < 1.0:
            preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            preview = image

        osd = pytesseract.image_to_osd(preview, config="--psm 0")
        match = re.search(r"Rotate:\s*(\d+)", osd)
        match_conf = re.search(r"Orientation confidence:\s*([\d.]+)", osd)
        osd_conf = float(match_conf.group(1)) if match_conf else 0.0

        if match and osd_conf >= 3.0:
            angle = int(match.group(1))
            if angle == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass

    return image


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_variants(image):
    """
    Generate clean, OCR-friendly grayscale variants.
    Returns a list of 3 grayscale np.ndarray variants:
    1. contrast: CLAHE contrast-enhanced variant (preserves faint label text)
    2. gray: clean normalized grayscale
    3. otsu: Otsu threshold on Gaussian blurred contrast (crisp binarization)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]

    # Bound oversized images to prevent out-of-memory and slow Tesseract runs
    max_dimension = 2200
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Mild upscale if small image
    if max(h, w) < 1000 and max(h, w) > 150:
        scale = 1.5
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Otsu threshold on Gaussian blurred contrast
    denoised = cv2.GaussianBlur(contrast, (3, 3), 0)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [contrast, gray, otsu]


# ============================================================
# OCR CONFIGURATIONS
# ============================================================

OCR_CONFIGS = [
    "--oem 3 --psm 6",
    "--oem 3 --psm 3",
]


# ============================================================
# OCR WITH DATA & SCORING
# ============================================================

def run_ocr_pass(image, config):
    """
    Execute a Tesseract OCR pass returning cleaned text, word details, and confidence.
    """
    try:
        data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        words = []
        confidences = []

        for i in range(len(data["text"])):
            word = clean_text(data["text"][i])
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0

            if not word or conf < 0:
                continue

            words.append({
                "text": word,
                "confidence": conf,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            })

            # Exclude low-confidence noise dots (< 20) from confidence calculation
            if conf >= 20.0:
                confidences.append(conf)

        text = pytesseract.image_to_string(image, config=config)
        cleaned_text = clean_ocr_lines(text)

        mean_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        )

        useful_words = sum(1 for c in confidences if c >= 35.0)

        return {
            "text": cleaned_text,
            "raw": text,
            "words": words,
            "confidence": mean_confidence,
            "useful_words": useful_words,
        }

    except Exception:
        return {
            "text": "",
            "raw": "",
            "words": [],
            "confidence": 0.0,
            "useful_words": 0,
        }


def run_ocr_data(image, config="--oem 3 --psm 6"):
    """Compatibility wrapper for Phase 1 tests."""
    return run_ocr_pass(image, config)


def score_ocr_result(result):
    """
    Score OCR output based on confidence and statutory metrology keyword hits.
    """
    text = result.get("text", "")
    if not text:
        return -1.0

    lines = [x for x in text.splitlines() if x.strip()]
    legal_keywords = [
        "mrp", "mfg", "mfd", "manufactured", "packed", "pkd",
        "batch", "lot", "net", "weight", "quantity", "customer",
        "consumer", "care", "expiry", "best before", "use by",
        "made in", "country", "price", "usp"
    ]

    lower = text.lower()
    keyword_hits = sum(1 for kw in legal_keywords if kw in lower)

    # Penalize passes with very few words
    word_count = result.get("useful_words", 0)
    penalty = 0.3 if word_count < 5 else 1.0

    score = (
        (result["confidence"] * 0.60
        + min(word_count, 80) * 0.20
        + min(len(lines), 40) * 0.10
        + keyword_hits * 2.5) * penalty
    )
    return score


def combine_selected_results(results):
    """
    Combine unique non-garbage lines from OCR passes.
    """
    if not results:
        return ""

    ranked = sorted(results, key=score_ocr_result, reverse=True)

    output_lines = []
    seen = set()

    for result in ranked:
        for raw_line in result["text"].splitlines():
            line = clean_text(raw_line)
            if not line or looks_like_garbage(line):
                continue

            key = normalize_for_comparison(line)
            if not key or key in seen:
                continue

            seen.add(key)
            output_lines.append(line)

    return "\n".join(output_lines)


def normalize_number(value):
    if not value:
        return ""
    replacements = {
        "O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5"
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value.strip()


# ============================================================
# TABULAR GRID DECLARATION PARSER
# ============================================================

def parse_packaging_grid(text):
    """
    Detect and parse tabular grid packaging declarations.
    Indian packaging commonly groups dynamic declarations in grids:
      Header row : DATE OF PACKAGING : USE BY : BATCH NO : MRP : USP
      Value row  : AUG26 | JUL27 | E190771633 | ₹ 10.00 | ₹ 1.25/g
    Works generically across any packaged commodity layout.
    """
    found = {}
    lines = [clean_text(l) for l in text.splitlines() if clean_text(l)]

    grid_header_indices = []
    for idx, line in enumerate(lines):
        lower = line.lower()
        hits = sum(1 for kw in ["packag", "mfg", "mfd", "pkd", "use by", "best before", "expiry", "batch", "lot", "mrp", "usp"] if kw in lower)
        if hits >= 2:
            grid_header_indices.append(idx)

    candidate_lines = []
    for h_idx in grid_header_indices:
        for offset in range(1, 4):
            if h_idx + offset < len(lines):
                candidate_lines.append(lines[h_idx + offset])

    all_lines = candidate_lines + lines

    # 1. USP in grid
    for line in all_lines:
        m = re.search(r"(?:₹|rs[\.:]?|inr)?\s*(\d+(?:[\.-]\d+)?)\s*(?:/|per|\s*7\s*)\s*(g|kg|ml|l|cm|m|unit|piece)s?\b", line, re.I)
        if m and "unit_sale_price" not in found:
            found["unit_sale_price"] = f"Rs. {m.group(1).replace('-', '.')}/{m.group(2).lower()}"
            break
        m_slash = re.search(r"(?:₹|rs[\.:]?|inr)?\s*(\d+\.\d{1,2})\s*7\s*(?:g|9)\b", line, re.I)
        if m_slash and "unit_sale_price" not in found:
            found["unit_sale_price"] = f"Rs. {m_slash.group(1)}/g"
            break

    # 2. MRP in grid
    for line in all_lines:
        clean_line = re.sub(r"(?:₹|rs[\.:]?|inr)?\s*\d+(?:\.\d+)?\s*(?:/|per|7)\s*(?:g|kg|ml|l|9)\b", "", line, flags=re.I)
        m = re.search(r"(?:m\.?\s*r\.?\s*p\.?|max(?:imum)?\s*retail\s*price)[^0-9\n\r]*?(?:₹|rs[\.:]?|inr)?\s*[:\-\s]*(\d+(?:\.\d+)?)", clean_line, re.I)
        if m and "mrp" not in found:
            try:
                num = float(m.group(1))
                if 0 < num < 50000 and not (1900 <= num <= 2100):
                    found["mrp"] = f"Rs. {int(num)}" if num.is_integer() else f"Rs. {num:.2f}"
            except ValueError:
                pass
        m2 = re.search(r"(?:₹|rs[\.:]|inr)\s*[:\-\s]*(\d{1,5}(?:\.\d{1,2})?)\s*(?:;|,|\s|$)(?!\s*/\s*(?:g|kg|ml|l))", clean_line, re.I)
        if m2 and "mrp" not in found:
            try:
                num = float(m2.group(1))
                if 0 < num < 50000 and not (1900 <= num <= 2100):
                    found["mrp"] = f"Rs. {int(num)}" if num.is_integer() else f"Rs. {num:.2f}"
            except ValueError:
                pass

    # 3. Dates in grid
    dates = []
    date_regex = re.compile(r"\b(" + MONTH_PATTERN + r"[\s/\-\.]*\d{2,4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[/-]\d{2,4})\b", re.I)
    for line in all_lines:
        if re.search(r"(?:lic\b|licence|license)\b", line, re.I):
            continue
        for m in date_regex.finditer(line):
            val = m.group(1).strip()
            num_m = re.match(r"^(\d{1,2})[/-](\d{2,4})$", val)
            if num_m and int(num_m.group(1)) > 12:
                continue
            if val not in dates:
                dates.append(val)

    duration_expiry = None
    for line in all_lines:
        m = re.search(r"\b(\d{1,2}\s*(?:months?|days?|years?|m))\b", line, re.I)
        if m:
            duration_expiry = m.group(1).strip()
            break

    if dates:
        found["manufacturing_date"] = dates[0]
        if len(dates) >= 2:
            found["expiry_or_best_before"] = dates[1]
        elif duration_expiry:
            found["expiry_or_best_before"] = duration_expiry
    elif duration_expiry:
        found["expiry_or_best_before"] = duration_expiry

    # 4. Batch Number in grid
    batch_candidates = []
    for line in all_lines:
        m = re.search(r"(?:batch|lot|b\.?\s*no\.?|b/no\.?)\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Za-z0-9\-/]{3,})", line, re.I)
        if m:
            val = m.group(1).strip()
            if val.lower() not in ["number", "no", "date", "mfg", "mfd", "expiry", "before", "retail", "price", "taxes", "mrp", "usp", "use", "packag"]:
                if re.search(r"\d", val) and not any(k in val.lower() for k in ["img", "image", "table", "solution"]):
                    batch_candidates.append((val, 100))

        # Look for explicit matrix stamp tokens only if accompanied by other packaging metadata
        tokens = re.findall(r"\b([A-Z0-9][A-Za-z0-9\-/]{3,20})\b", line)
        for tok in tokens:
            if date_regex.search(tok):
                continue
            if re.match(r"^[A-Za-z]{3}\d{2,4}$", tok):
                continue
            if any(bad in tok.lower() for bad in ["img", "image", "weight", "netweight", "packaging", "packaged", "mrp", "usp", "price", "solution", "table"]):
                continue
            if re.search(r"\d", tok) and re.search(r"[A-Za-z]", tok):
                specificity = len(tok) + sum(c.isdigit() for c in tok)
                batch_candidates.append((tok, specificity))

    if batch_candidates:
        batch_candidates.sort(key=lambda x: x[1], reverse=True)
        found["batch_number"] = batch_candidates[0][0]

    return found


# ============================================================
# FIELD EXTRACTORS
# ============================================================

def extract_product_name(text):
    lines = [clean_text(x) for x in text.splitlines() if clean_text(x)]
    categories = [
        "baking powder", "food additive", "tea", "coffee", "biscuits",
        "cookies", "flour", "salt", "sugar", "spices", "detergent",
        "shampoo", "soap", "oil", "rice", "atta", "snack", "noodles",
        "powder", "syrup", "drops", "spray", "lotion", "cream", "tablets",
        "capsules", "juice", "deodorant", "perfume"
    ]

    forbidden_terms = re.compile(
        r"(mrp|batch|mfg|mfd|expiry|best before|use by|net\s*weight|net\s*wt|netweight|wetweight|"
        r"weight|quantity|contents|customer care|consumer care|directions|store in|dry place|keep|airtight|spoon|"
        r"thank you|www\.|@|tel|phone|address|fssai|lic\b|licence|license|date of|packag|"
        r"taxes?|tax\b|inclusive|incisive|flammable|inflammable|caution|warning|danger|poison|instructions|dispense|tighten|pierce|"
        r"nutrition|nutritional|energy|protein|carbohydrate|sugar|fat|cholesterol|sodium|potassium|dietary|fiber|serving\s*size|per\s*100\s*g|per\s*pack|ingredients?)",
        re.I
    )

    for line in lines:
        lower = line.lower()
        if "www." in lower or "@" in lower or "http" in lower:
            continue
        if any(term in lower for term in ["nutrition", "ingredient", "serving size", "per 100", "directions"]):
            continue

        for category in categories:
            if not re.search(r"\b" + re.escape(category) + r"\b", lower):
                continue

            match = re.search(
                r"(?:use|add|mix|take|with|of)\s+(?:\d+(?:\.\d+)?\s*)?(?:g|kg|mg|ml|l)?\s*(?:of\s+)?(.+?\b"
                + re.escape(category) + r")\b",
                line,
                re.IGNORECASE
            )
            cat_m = re.search(r"(.{0,80}\b" + re.escape(category) + r")\b", line, re.IGNORECASE)
            candidate = match.group(1).strip() if match else (cat_m.group(1).strip() if cat_m else "")
            if not candidate:
                continue

            candidate = re.sub(r"^[\s:;,\-*\"'()|]+|[\s:;,\-*\"'()|]+$", "", candidate)
            candidate = re.split(r"\b(?:for|every|mix|use|add|take|directions|direction|immediately|with|in|on)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            candidate = re.sub(r"^(?:use|add|mix|take|for|with|of)\b(?:\s+\d+(?:\.\d+)?)?(?:\s*(?:g|kg|mg|ml|l))?(?:\s+of)?\s+", "", candidate, flags=re.IGNORECASE).strip()

            if candidate.lower().startswith("see ") or re.search(r"\b\d+\s*mg\b", candidate, re.I):
                continue

            words = candidate.split()
            if words and len(candidate) <= 70 and len(words) <= 8 and not forbidden_terms.search(candidate) and re.search(r"[A-Za-z]", candidate):
                return candidate

    for line in lines[:20]:
        if looks_like_garbage(line):
            continue
        lower = line.lower()
        if forbidden_terms.search(lower):
            continue
        if re.match(r"^(use|mix|add|take|directions?|store|keep|do not|tighten|pierce|dispense|see)\b", lower):
            continue
        if any(term in lower for term in ["nutrition", "ingredient", "serving", "per 100", "table"]):
            continue

        # Disqualify candidate lines containing prices, currencies, dates, or measurements
        if re.search(r"(?:₹|rs\.?|inr|\d+\.\d{2})", line, re.I):
            continue
        if re.search(r"\b(?:\d{1,2}[/-]\d{2,4}|" + MONTH_PATTERN + r"[\s/-]*\d{2,4})\b", line, re.I):
            continue
        if re.search(r"\b\d+\s*(?:g|kg|ml|l|mg)\b|/\s*(?:g|kg|ml|l)", line, re.I):
            continue

        # Disqualify lines with multiple numbers or excessive digits (e.g. tabular value rows)
        digits_count = sum(c.isdigit() for c in line)
        if digits_count > 2 or len(re.findall(r"\d+", line)) > 1:
            continue

        alnum_ratio = sum(c.isalnum() for c in line) / max(len(line), 1)
        alpha_count = sum(c.isalpha() for c in line)
        words = [re.sub(r"[^A-Za-z0-9]", "", w) for w in line.split()]
        words = [w for w in words if w]

        # Require at least one word of length >= 3 containing a standard vowel (a, e, i, o, u)
        has_vowel_word = any(len(w) >= 3 and re.search(r"[aeiouAEIOU]", w) for w in words)
        if not has_vowel_word:
            continue

        # Clean edge noise
        clean_candidate = re.sub(r"^[\s:;,\-*\"'()|~]+|[\s:;,\-*\"'()|~]+$", "", line).strip()
        if 4 <= len(clean_candidate) <= 60 and alnum_ratio >= 0.65 and alpha_count >= 4 and len(words) >= 1:
            return clean_candidate

    return ""


def extract_manufacturer(text):
    patterns = [
        r"(?:(?:mfd\.?|mfg\.?|manufactured)\s*(?:&|and)?\s*(?:pkd\.?|packed)?\s*by)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\s&.,'()\/\-]{2,120})",
        r"(?:manufactured\s+at)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\s&.,'()\/\-]{2,120})",
        r"(?:(?:packed|pkd\.?)\s*by)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\s&.,'()\/\-]{2,120})",
        r"(?:(?:marketed|mktd\.?)\s*by)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\s&.,'()\/\-]{2,120})",
        r"(?:(?:imported|imp\.?)\s*by)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\s&.,'()\/\-]{2,120})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = clean_text(match.group(1))
            value = re.split(r"\b(?:batch|mfg|mfd|mrp|net|customer|consumer|expiry|best before)\b", value, flags=re.IGNORECASE)[0].strip()
            if len(value) >= 3:
                return value

    entity_pattern = (
        r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\s+"
        r"(?:Foods?|Products?|Beverages?|Industries|Laboratories|Enterprises|Pvt\.?\s*Ltd\.?|Limited|Corporation))\b"
    )

    for line in text.splitlines():
        line_clean = clean_text(line)
        match = re.search(entity_pattern, line_clean)
        if match:
            candidate = match.group(1).strip()
            if not re.search(r"(mrp|batch|expiry|best before|net\s*wt|customer|consumer care)", candidate, re.IGNORECASE):
                return candidate

    return ""


def extract_address(text):
    patterns = [
        r"(?:(?:mfd\.?|mfg\.?|manufactured|packed|pkd\.?)\s*(?:at|unit|factory|premises))\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\s&.,'()\/\-]{5,150}\b\d{6}\b)",
        r"(?:(?:mfd\.?|mfg\.?|manufactured|packed|pkd\.?)\s*(?:at|unit|factory|premises))\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\s&.,'()\/\-]{5,120})",
        r"(?:address|works\s+at)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\s&.,'()\/\-]{5,120})",
        r"\b([A-Za-z0-9\s,.\-]{5,100}\b(?:road|rd|street|st|lane|nagar|plot|sector|industrial|dist|state)\b[A-Za-z0-9\s,.\-]{0,60}(?:\d{6})?)\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = clean_text(match.group(1))
            value = re.split(r"\b(?:batch|mfg|mfd|mrp|net\s*wt|customer|consumer|expiry|best before|tel|phone)\b", value, flags=re.IGNORECASE)[0].strip()
            if len(value) >= 6:
                return value

    return ""


def extract_net_quantity(text):
    # Pattern 1: Explicit net weight / quantity prefix
    prefix_pattern = (
        r"(?:net\s*(?:weight|wt\.?|quantity|contents?|vol\.?|volume)|wet\s*weight|wetweight|weight|n\.?\s*w\.?|w\.?\s*t\.?|contents?)\s*[:\-=\.]*\s*"
        r"(\d+(?:\.\d+)?)\s*(kg|g|mg|l|ml)\b"
    )
    match = re.search(prefix_pattern, text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2).lower()}"

    # Pattern 2: Standalone quantity - requires clean line and realistic commodity size
    for line in text.splitlines():
        line_clean = clean_text(line)
        if looks_like_garbage(line_clean):
            continue
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(kg|g|mg|l|ml)\b", line_clean, re.IGNORECASE)
        if m:
            qty = float(m.group(1))
            unit = m.group(2).lower()
            if unit in ["kg", "l"] or (unit == "ml" and qty >= 5) or (unit == "g" and (qty >= 10 or "." in m.group(1))):
                return f"{m.group(1)} {unit}"

    # Spatial pattern: number followed by 9 or g right before MRP or currency symbol (resolves '1009 mrp' OCR misread)
    spatial_m = re.search(r"\b(\d{1,4})\s*[9g]\s*(?:mrp|rs[\.:]|₹|inr|price|incl)", text, re.IGNORECASE)
    if spatial_m:
        return f"{spatial_m.group(1)} g"

    return ""


def extract_mrp(text):
    patterns = [
        # Same-line MRP
        r"(?:m\.?\s*r\.?\s*p\.?|max(?:imum)?\s*retail\s*price)[^0-9\n\r]*?(?:₹|rs[\.:]?|inr)?\s*[:\-\s]*(\d+(?:\.\d{1,2})?)(?:/[-–])?",
        # Multiline MRP: MRP on one line (possibly followed by (inclusive of all taxes)), price on subsequent line
        r"(?:m\.?\s*r\.?\s*p\.?|max(?:imum)?\s*retail\s*price)[^\n\r]*\n(?:\s*\([^\)\n\r]*\)[^\n\r]*\n)?\s*(?:₹|rs[\.:]?|inr)?\s*[:\-\s]*(\d{1,5}(?:\.\d{1,2})?)(?:/[-–])?",
        # Explicit Indian currency prefix
        r"(?:₹|rs[\.:]|inr)\s*[:\-\s]*(\d{1,5}(?:\.\d{1,2})?)(?:/[-–])?\s*(?:;|,|\s|$)(?!\s*/\s*(?:g|kg|ml|l|unit|piece))",
    ]

    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = normalize_number(match.group(1))
            try:
                number = float(value)
                if 1900 <= number <= 2100:
                    continue
                if 0 < number < 50000:
                    candidates.append(number)
            except ValueError:
                continue

    if candidates:
        number = max(candidates)
        if number.is_integer():
            return f"Rs. {int(number)}"
        return f"Rs. {number:.2f}"

    return ""


def extract_unit_sale_price(text):
    patterns = [
        r"(?:unit\s+sale\s+price|unit\s+price|u\.?s\.?p\.?)[^0-9\n\r]*?(?:₹|rs[\.:]?|inr|[Ff])?\s*(\d+(?:[\.-]\d+)?)\s*(?:/|per|\s*7\s*)\s*(g|kg|ml|l|cm|m|unit|piece)s?\b",
        r"(?:₹|rs[\.:]|inr)\s*(\d+(?:[\.-]\d+)?)\s*(?:/|per|\s*7\s*)\s*(g|kg|ml|l|cm|m|unit|piece)s?\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            unit = match.group(2).lower() if len(match.groups()) >= 2 and match.group(2) else "g"
            if unit == "9":
                unit = "g"
            elif unit == "m" and (re.search(r"/\s*m[l1I]\b", match.group(0), re.I) or re.search(r"\b(?:ml|liquid|drops|syrup)\b", text, re.I)):
                unit = "ml"
            price_val = match.group(1).replace("-", ".")
            return f"Rs. {price_val}/{unit}"

    return ""


def extract_manufacturing_date(text):
    # Pass 1: Explicit manufacturing prefix
    mfg_prefixes = r"(?:(?:date\s*of\s*(?:packaging|pkd|mfg))|[mw][fiyl][gqdo]\w*|manufactured|manufacturing|pkd\.?|packed)\s*(?:date)?\s*[:\-=.]*\s*"
    date_body = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[/-]\d{2,4}|" + MONTH_PATTERN + r"[\s/\-\.]*\d{2,4})"

    match = re.search(mfg_prefixes + date_body, text, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        val = re.sub(r"\b41/", "11/", val)
        val = re.sub(r"/28(\d{2})\b", r"/20\1", val)
        return val

    # Pass 2: Standalone Month-Year NOT preceded by expiry words
    for m in re.finditer(r"\b(" + MONTH_PATTERN + r"[\s/\-\.]*\d{2,4})\b", text, re.IGNORECASE):
        prefix_window = text[max(0, m.start() - 25):m.start()].lower()
        if not re.search(r"(?:use\s*by|best\s*before|expiry|exp\b)", prefix_window):
            val = m.group(1).strip()
            val = re.sub(r"/28(\d{2})\b", r"/20\1", val)
            return val

    # Pass 3: General date NOT preceded by expiry or license words
    for m in re.finditer(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[/-]\d{2,4})\b", text, re.IGNORECASE):
        prefix_window = text[max(0, m.start() - 30):m.start()].lower()
        if re.search(r"(?:use\s*by|best\s*before|expiry|exp\b|lic\b|licence|license)", prefix_window):
            continue
        val = m.group(1).strip()
        num_m = re.match(r"^(\d{1,2})[/-](\d{2,4})$", val)
        if num_m and int(num_m.group(1)) > 12:
            continue
        val = re.sub(r"\b41/", "11/", val)
        val = re.sub(r"/28(\d{2})\b", r"/20\1", val)
        return val

    return ""


def extract_batch_number(text):
    stop_words = {
        "number", "no", "date", "mfg", "mfd", "expiry", "before",
        "retail", "price", "taxes", "mrp", "usp", "artificial",
        "flavouring", "agent", "popular", "vanilla", "powder",
        "chemical", "industries", "ingredient", "directions"
    }

    patterns = [
        r"(?:batch|lot|b\.?\s*no\.?|b/no\.?)\s*(?:no\.?|number|#)?\s*[:\-=\.]*\s*([A-Za-z0-9\-/]{3,})",
        r"\b([A-Za-z0-9\-/]{4,})\s+(?:mfd|mfg|date\s*of)\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = match.group(1).strip()
            if value.lower() in stop_words:
                continue
            if any(bad in value.lower() for bad in ["img", "image", "weight", "netweight", "packaging", "packaged", "table", "solution", "nutrition"]):
                continue
            if re.search(r"\d", value):
                return value

    return ""


def extract_expiry(text):
    patterns = [
        r"(?:use\s*by|best\s*before|expiry|exp\.?)\s*(?:date)?\s*[:\-=\.]*\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[/-]\d{2,4}|" + MONTH_PATTERN + r"[\s/\-\.]*\d{2,4}|\d+\s*(?:months?|days?|years?|m)\b)",
        r"\b(\d+\s*(?:months?|days?|years?))\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            num_m = re.match(r"^(\d{1,2})[/-](\d{2,4})$", val)
            if num_m and int(num_m.group(1)) > 12:
                continue
            val = re.sub(r"/28(\d{2})\b", r"/20\1", val)
            return val

    return ""


def extract_consumer_care(text):
    parts = []

    # Phone: toll-free 1800, standard landline with STD code (e.g. 0120-2564285), or mobile/international
    phone_patterns = [
        r"(?:phone|tel|mobile|contact|call|helpline|no\.)\s*(?:no\.?)?\s*[:\-]?\s*(\+?\d{2,4}[\s\-]*\d{2,5}[\s\-]*\d{6,8}|\d{10,11})",
        r"\b0\d{2,4}[\s\-]+\d{6,8}\b",
        r"\+91[\s\-]?\d{2,5}[\s\-]?\d{6,8}",
        r"1800[\s\-]?\d{3}[\s\-]?\d{3,4}",
        r"\+?\d{2,4}[\s\-]?\d{2,5}[\s\-]?\d{6,8}",
    ]

    for pattern in phone_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1) if match.groups() else match.group(0)
            parts.append(f"Phone: {val.strip()}")
            break

    # Email
    email_match = re.search(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+", text)
    if email_match:
        parts.append(f"Email: {email_match.group(0)}")

    # Website
    web_match = re.search(r"(?:https?://|www\.)[A-Za-z0-9-]+\.[A-Za-z0-9.-]+", text, re.IGNORECASE)
    if web_match:
        parts.append(f"Web: {web_match.group(0)}")

    return " | ".join(parts)


def extract_country(text):
    patterns = [
        r"(?:country\s+of\s+origin)\s*[:\-]?\s*([A-Za-z]+)",
        r"(?:made\s+in)\s*[:\-]?\s*([A-Za-z]+)",
        r"(?:product\s+of)\s*[:\-]?\s*([A-Za-z]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().title()

    if re.search(r"\bIndia\b", text, re.IGNORECASE):
        return "India"

    return ""


# ============================================================
# MAIN LOCAL OCR EXTRACTION ENTRYPOINT
# ============================================================

def extract_product_info(image_path):
    start_time = time.time()

    image = cv2.imread(image_path)
    if image is None:
        return {
            "product_name": "",
            "manufacturer": "",
            "net_quantity": "",
            "mrp": "",
            "unit_sale_price": "",
            "manufacturing_date": "",
            "batch_number": "",
            "expiry_or_best_before": "",
            "consumer_care": "",
            "country_of_origin": "",
            "confidence": 0.0,
            "processing_time_sec": 0.0,
            "bounding_boxes_count": 0,
            "raw_text": "ERROR: Unable to open image.",
        }

    image = detect_and_correct_orientation(image)
    variants = preprocess_variants(image)

    ocr_results = []
    for variant in variants:
        for config in OCR_CONFIGS:
            res = run_ocr_pass(variant, config)
            if res["text"]:
                ocr_results.append(res)

    text = combine_selected_results(ocr_results)
    grid_fields = parse_packaging_grid(text)

    # Compute confidence from top OCR passes
    if ocr_results:
        ranked = sorted(ocr_results, key=score_ocr_result, reverse=True)
        top_results = ranked[:3]
        weights = [0.50, 0.30, 0.20]
        weighted_conf = 0.0
        weight_tot = 0.0
        for idx, r in enumerate(top_results):
            w = weights[idx]
            weighted_conf += r["confidence"] * w
            weight_tot += w
        confidence = round(weighted_conf / weight_tot if weight_tot else 0.0, 2)
    else:
        confidence = 0.0

    # Count bounding boxes from best result
    bounding_boxes = 0
    if ocr_results:
        best_result = max(ocr_results, key=score_ocr_result)
        bounding_boxes = sum(1 for w in best_result["words"] if w["confidence"] >= 20.0)

    # Extract fields with grid-aware fallbacks
    product_name = extract_product_name(text)
    manufacturer = extract_manufacturer(text)
    net_quantity = extract_net_quantity(text)

    mrp = extract_mrp(text) or grid_fields.get("mrp", "")
    unit_sale_price = extract_unit_sale_price(text) or grid_fields.get("unit_sale_price", "")
    manufacturing_date = extract_manufacturing_date(text) or grid_fields.get("manufacturing_date", "")
    batch_number = extract_batch_number(text) or grid_fields.get("batch_number", "")
    expiry_or_best_before = extract_expiry(text) or grid_fields.get("expiry_or_best_before", "")
    consumer_care = extract_consumer_care(text)
    country_of_origin = extract_country(text)
    address = extract_address(text)

    elapsed_time = round(time.time() - start_time, 2)

    return {
        "product_name": product_name,
        "manufacturer": manufacturer,
        "address": address,
        "net_quantity": net_quantity,
        "mrp": mrp,
        "unit_sale_price": unit_sale_price,
        "manufacturing_date": manufacturing_date,
        "batch_number": batch_number,
        "expiry_or_best_before": expiry_or_best_before,
        "consumer_care": consumer_care,
        "country_of_origin": country_of_origin,
        "confidence": confidence,
        "processing_time_sec": elapsed_time,
        "bounding_boxes_count": bounding_boxes,
        "raw_text": text,
    }


# ============================================================
# COMMAND-LINE RUNNER
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("           SIH 26034 - OCR TEST")
    print("=" * 60)

    test_img = sys.argv[1] if len(sys.argv) > 1 else "product.jpg"

    print(f"\nAnalyzing: {test_img}")
    if not os.path.exists(test_img):
        print(f"\nERROR: {test_img} was not found.")
        sys.exit(1)

    result = extract_product_info(test_img)

    print()
    print("=" * 60)
    print("DETECTED LEGAL METROLOGY FIELDS")
    print("=" * 60)
    print(f"Product Name       : {result['product_name'] or 'Not detected'}")
    print(f"Manufacturer       : {result['manufacturer'] or 'Not detected'}")
    print(f"Net Quantity       : {result['net_quantity'] or 'Not detected'}")
    print(f"MRP                : {result['mrp'] or 'Not detected'}")
    print(f"Unit Sale Price    : {result['unit_sale_price'] or 'Not detected'}")
    print(f"Manufacturing Date : {result['manufacturing_date'] or 'Not detected'}")
    print(f"Batch Number       : {result['batch_number'] or 'Not detected'}")
    print(f"Best Before/Expiry : {result['expiry_or_best_before'] or 'Not detected'}")
    print(f"Consumer Care      : {result['consumer_care'] or 'Not detected'}")
    print(f"Country of Origin  : {result['country_of_origin'] or 'Not detected'}")

    print()
    print("=" * 60)
    print("OCR METRICS")
    print("=" * 60)
    print(f"Processing Time    : {result['processing_time_sec']} seconds")
    print(f"OCR Confidence     : {result['confidence']} %")
    print(f"Bounding Boxes     : {result['bounding_boxes_count']}")

    print()
    print("=" * 60)
    print("RAW OCR TEXT")
    print("=" * 60)
    print(result["raw_text"])

    print()
    print("=" * 60)
    print("OCR TEST COMPLETED")
    print("=" * 60)
