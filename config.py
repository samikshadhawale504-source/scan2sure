"""
Configuration Layer for Legal Metrology PCR 2011 Compliance System.
Provides central, configurable threshold settings for extraction confidence,
fallback handling, and statutory verification requirements.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Robustly find project root directory regardless of current working directory
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

def load_project_env():
    """Load .env from project root directory with override=True."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
    else:
        # Fallback to standard dotenv discovery
        load_dotenv(override=True)

# Execute initial load
load_project_env()


def get_provider_status() -> dict:
    """
    Safe configuration diagnostic that returns only boolean YES/NO status.
    NEVER displays actual keys, prefixes, or suffixes.
    """
    load_project_env()
    m_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    g_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    return {
        "mistral_configured": bool(m_key and len(m_key) > 5),
        "gemini_configured": bool(g_key and len(g_key) > 5)
    }


# Default minimum confidence threshold for automated statutory compliance determinations.
# Extractions below this threshold require human inspector verification (REQUIRES_REVIEW).
DEFAULT_EXTRACTION_CONFIDENCE_THRESHOLD = 70.0


def get_extraction_confidence_threshold() -> float:
    """
    Returns the effective extraction confidence threshold (percentage 0.0 to 100.0).
    Allows dynamic override via environment variable EXTRACTION_CONFIDENCE_THRESHOLD.
    """
    env_val = os.environ.get("EXTRACTION_CONFIDENCE_THRESHOLD")
    if env_val is not None:
        try:
            return float(env_val)
        except (ValueError, TypeError):
            pass
    return DEFAULT_EXTRACTION_CONFIDENCE_THRESHOLD


def set_extraction_confidence_threshold(threshold: float) -> None:
    """
    Programmatically sets the extraction confidence threshold.
    Useful for testing varying visual quality conditions and administrative policies.
    """
    os.environ["EXTRACTION_CONFIDENCE_THRESHOLD"] = str(float(threshold))


# Dynamic property alias
EXTRACTION_CONFIDENCE_THRESHOLD = get_extraction_confidence_threshold()
