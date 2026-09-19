"""
Data schemas and domain models for Legal Metrology (Packaged Commodities) Rules, 2011 compliance inspection.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NOT_DETECTED = "NOT_DETECTED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ExtractionEngine(str, Enum):
    GEMINI = "Gemini Vision AI"
    LOCAL_OCR = "Local OCR (OpenCV + Tesseract)"
    HYBRID_FALLBACK = "Local OCR (OpenCV + Tesseract - Fallback)"
    MANUAL = "Manual Input"
    ECOMMERCE = "E-Commerce Listing"


class FieldDetail(BaseModel):
    """Encapsulates an extracted field with statutory traceability and explainability."""
    name: str
    extracted_value: str = ""
    confidence: float = 0.0
    source: str = "unknown"
    evidence_snippet: Optional[str] = None
    bounding_box: Optional[Dict[str, Any]] = None
    is_directly_observed: bool = True
    requires_manual_verification: bool = False
    notes: Optional[str] = None


class PackagedCommodityRawData(BaseModel):
    """Standard structured model for all 20+ statutory packaging declarations."""
    product_name: Optional[str] = Field(default="", description="Brand name and generic/common name")
    manufacturer: Optional[str] = Field(default="", description="Name of the manufacturer")
    manufacturer_address: Optional[str] = Field(default="", description="Complete postal address of the manufacturer with PIN code")
    packer: Optional[str] = Field(default="", description="Name of pre-packer if distinct from manufacturer")
    packer_address: Optional[str] = Field(default="", description="Complete postal address of packer")
    importer: Optional[str] = Field(default="", description="Name of importer for foreign goods")
    importer_address: Optional[str] = Field(default="", description="Complete postal address of importer")
    net_quantity: Optional[str] = Field(default="", description="Net quantity declared (e.g. 50 g, 500 ml, 1 kg)")
    quantity_unit: Optional[str] = Field(default="", description="Unit of measure (e.g. g, kg, ml, l, piece)")
    mrp: Optional[str] = Field(default="", description="Maximum Retail Price with tax clause (e.g. Rs. 28.00 incl. of all taxes)")
    unit_sale_price: Optional[str] = Field(default="", description="Unit sale price (e.g. Rs. 0.56 / g)")
    manufacturing_date: Optional[str] = Field(default="", description="Month and year of manufacture (MM/YYYY)")
    packing_date: Optional[str] = Field(default="", description="Month and year of pre-packing if applicable")
    import_date: Optional[str] = Field(default="", description="Month and year of import if applicable")
    batch_number: Optional[str] = Field(default="", description="Batch / Lot / Identification code")
    best_before: Optional[str] = Field(default="", description="Best before / Expiry period or date")
    consumer_care_phone: Optional[str] = Field(default="", description="Helpline phone number for grievances")
    consumer_care_email: Optional[str] = Field(default="", description="Grievance officer email address")
    consumer_care_website: Optional[str] = Field(default="", description="Customer care website or grievance portal")
    country_of_origin: Optional[str] = Field(default="India", description="Country of manufacture or origin")
    other_mandatory_declarations: Optional[str] = Field(default="", description="Any other statutory declarations (e.g. FSSAI, veg/non-veg)")


class RuleEvaluationResult(BaseModel):
    """Result of an individual Legal Metrology statutory rule check."""
    rule_id: str
    rule_description: str
    statutory_reference: str
    extracted_evidence: str = ""
    status: ComplianceStatus
    confidence: float = 100.0
    reason: str = ""
    recommended_action: str = ""
    category: str = "General"
    weight: int = 10
    score: float = 0.0


class CategoryScore(BaseModel):
    category: str
    score: float
    max_score: float
    percentage: float
    passed: int
    failed: int
    warnings: int


class ComplianceInspectionResult(BaseModel):
    """Complete statutory compliance inspection result."""
    overall_status: ComplianceStatus
    compliance_score: float
    category_scores: Dict[str, CategoryScore] = {}
    rule_evaluations: List[RuleEvaluationResult] = []
    fields: Dict[str, FieldDetail] = {}
    critical_violations: List[str] = []
    warnings: List[str] = []
    requires_review_fields: List[str] = []
    summary_text: str = ""
